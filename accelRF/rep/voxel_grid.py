from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch import Tensor

from .._C.rep import _ext
from .base import Explicit3D
from .utils import discretize_points, offset_points, trilinear_interp

MAX_DEPTH = 10000.0

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
        depth_im: HW
        cam_intr: 3*3
        cam_pose: 4*4
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors=None):
    """Save a 3D mesh to a polygon .ply file.
    """
    if colors is not None:
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                verts[i, 0], verts[i, 1], verts[i, 2],
                norms[i, 0], norms[i, 1], norms[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2],
            ))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

        ply_file.close()
    else:
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f\n" % (
                verts[i, 0], verts[i, 1], verts[i, 2],
                norms[i, 0], norms[i, 1], norms[i, 2]
            ))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

        ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))


def inv_project_points_cam_coords(intrinsic, uvd, c2w,use_metric_depth=False):
    '''
    As inv_project_points but doesn't do the homogeneous transformation
    int: 3x3, uvd:Nx3, c2w:3X4
    '''
    if len(uvd.shape) > 2:
        uvd = uvd.reshape(-1,3)
    if intrinsic.shape[0]>3:
        intrinsic = intrinsic[:3,:3]
    if c2w.shape[0]>3:
        c2w = c2w[:3,:]
    n_points = uvd.shape[0]

    # creating the camera rays
    uv1 = torch.hstack((uvd[:, :2], torch.ones((n_points, 1),device=uvd.device)))
    camera_rays = torch.mm(intrinsic.inverse(), uv1.T).T

    # forming the xyz points in the camera coordinates
    temp = uvd[:, 2][:,None].repeat(1,3)
    xyz_at_cam_loc = temp * camera_rays
    # cal the metric depth
    if use_metric_depth:
        metric_dp = torch.norm(xyz_at_cam_loc,2,dim=-1)
    else:
        metric_dp = None
    xyz_at_cam_loc = torch.hstack((xyz_at_cam_loc, torch.ones((n_points, 1),device=xyz_at_cam_loc.device)))
    # project to xyz
    xyz_world = torch.mm(c2w, xyz_at_cam_loc.T).T

    return xyz_world, metric_dp # Nx3


def integrate(
        depth_im,
        cam_intr,
        cam_pose,
        obs_weight,
        world_c,
        vox_coords,
        weight_vol,
        tsdf_vol,
        sdf_trunc,
        im_h,
        im_w,
        use_metric_depth=False
):
    # Convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / pix_z) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / pix_z) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    valid_vox_x = vox_coords[valid_pix, 0]
    valid_vox_y = vox_coords[valid_pix, 1]
    valid_vox_z = vox_coords[valid_pix, 2]
    depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # get depth mask
    depth_mask = depth_im.reshape(-1,1).ne(0.)
    w = torch.arange(im_w, dtype=torch.long)
    h = torch.arange(im_h, dtype=torch.long)
    x, y= torch.meshgrid(h, w)
    xy = torch.stack([y,x]).permute(1,2,0).to(cam_pose.device) # hw2 NOTE in fact, this coord uv is 0,0 1,0 2,0
    uvd = torch.cat((xy,depth_im[:,:,None]),dim=-1)
    pixel_xyz_world, metric_dp = inv_project_points_cam_coords(cam_intr,uvd,cam_pose,use_metric_depth) #Nx3

    if metric_dp is not None:
        metric_dp = metric_dp.reshape(im_h, im_w)

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    valid_vox_x = valid_vox_x[valid_pts]
    valid_vox_y = valid_vox_y[valid_pts]
    valid_vox_z = valid_vox_z[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    return weight_vol, tsdf_vol, depth_mask,pixel_xyz_world,metric_dp

def generate_pts_full(vol_dim, vol_origin, voxel_size, pt_pervox=8):
    '''
        used for generate points to extract mesh, only used in whole xyz
    '''
    xv, yv, zv = torch.meshgrid(
        torch.arange(0, vol_dim[0]*pt_pervox),
        torch.arange(0, vol_dim[1]*pt_pervox),
        torch.arange(0, vol_dim[2]*pt_pervox),
    )
    vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long()
    pts = vol_origin + ((voxel_size/pt_pervox) * vox_coords)
    return pts

@torch.jit.script
def generate_pts_vox(center_pts: torch.Tensor, voxel_size: float=0.04, pt_per_edge:int=8):
    '''
        input: center_pts: N, 3
        sample_pts: N, 512, 3
    '''
    xv, yv, zv = torch.meshgrid(
        torch.arange(0, pt_per_edge),
        torch.arange(0, pt_per_edge),
        torch.arange(0, pt_per_edge),
    )
    vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(center_pts.device) # pt_per_edge**3, 3
    sample_pts = (center_pts - voxel_size*0.5)[:,None] + (vox_coords*voxel_size/pt_per_edge).unsqueeze(0)
    return sample_pts

class VoxelGrid(Explicit3D):
    '''
    Let's start with a simple voxel grid.
    '''
    def __init__(
        self,
        bbox: Tensor,
        voxel_size: float,
        use_corner: bool=True,
        ):
        '''
        bbox2voxel: https://github.com/facebookresearch/NSVF/fairnr/modules/encoder.py#L1053
        bbox: array [min_x,y,z, max_x,y,z]

        x represents center, O represents corner
            O O O O O
             x x x x 
            O O O O O
             x x x x 
            O O O O O
        Given a center x's coords [i,j,k]. its corners' coords are [i,j,k] + {0,1}^3
        '''
        super().__init__()
        self.use_corner = use_corner
        self.bbox = bbox
        v_min, v_max = bbox[:3], bbox[3:]
        steps = ((v_max - v_min) / voxel_size).round().long() + 1
        # note the difference between torch.meshgrid and np.meshgrid.
        center_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
        center_points = (center_coords * voxel_size + v_min).reshape(-1, 3) # start from lower bound
        # self.register_buffer('center_coords', center_coords)
        n_voxels = center_points.shape[0]
        occupancy = torch.ones(n_voxels, dtype=torch.bool) # occupancy's length unchanges unless splitting

        # corner points
        if use_corner:
            corner_shape = steps+1
            n_corners = corner_shape.prod().item()
            offset = offset_points().long() # [8, 3]
            corner1d = torch.arange(n_corners).reshape(corner_shape.tolist()) 
            center2corner = (center_coords[...,None,:] + offset).reshape(-1, 8, 3) # [..., 8,3]
            center2corner = corner1d[center2corner[...,0], center2corner[...,1], center2corner[...,2]] # [..., 8]
            self.register_buffer('center2corner', center2corner)
            self.register_buffer('n_corners', torch.tensor(n_corners))
        
        # keep min max voxels, for ray_intersection
        max_ray_hit = min(steps.sum().item(), n_voxels)
        # register_buffer for saving and loading.
        self.register_buffer('occupancy', occupancy)
        self.register_buffer('grid_shape', steps) # self.grid_shape = steps
        self.register_buffer('center_points', center_points)
        self.register_buffer('n_voxels', torch.tensor(n_voxels))
        self.register_buffer('max_ray_hit', torch.tensor(max_ray_hit))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))

    def ray_intersect(self, rays_o: Tensor, rays_d: Tensor):
        '''
        Args:
            rays_o, Tensor, (N_rays, 3)
            rays_d, Tensor, (N_rays, 3)
        Return:
            pts_idx, Tensor, (N_rays, max_hit)
            t_near, t_far    (N_rays, max_hit)
        '''
        pts_idx_1d, t_near, t_far = _ext.aabb_intersect(
            rays_o.contiguous(), rays_d.contiguous(), 
            self.center_points.contiguous(), self.voxel_size, self.max_ray_hit)
        t_near.masked_fill_(pts_idx_1d.eq(-1), MAX_DEPTH)
        t_near, sort_idx = t_near.sort(dim=-1)
        t_far = t_far.gather(-1, sort_idx)
        pts_idx_1d = pts_idx_1d.gather(-1, sort_idx)
        hits = pts_idx_1d.ne(-1).any(-1)
        return pts_idx_1d, t_near, t_far, hits

    # def get_corner_points(self, center_idx):
    #     corner_idx = self.center2corner[center_idx] # [..., 8]
    #     return self.corner_points[corner_idx] # [..., 8, 3]

    def pruning(self, keep):
        n_vox_left = keep.sum()
        if n_vox_left > 0 and n_vox_left < keep.shape[0]:
            self.center_points = self.center_points[keep].contiguous()
            self.occupancy.masked_scatter_(self.occupancy, keep)
            self.n_voxels = n_vox_left
            self.max_ray_hit = self.get_max_ray_hit()
            if self.use_corner:
                c2corner_idx = self.center2corner[keep] # [..., 8]
                corner_idx, center2corner = c2corner_idx.unique(sorted=True, return_inverse=True) # [.] and [..., 8]
                self.center2corner = center2corner.contiguous()
                self.n_corners = self.n_corners * 0 + corner_idx.shape[0]
                return corner_idx

    def splitting(self, feats: Optional[Tensor]=None):
        offset = offset_points(device=self.center_points.device).long() # [8, 3] scale [0,1]
        n_subvox = offset.shape[0] # 8
        old_center_coords = discretize_points(self.center_points, self.voxel_size) # [N ,3]

        self.voxel_size *= 0.5
        half_voxel = self.voxel_size * 0.5
        self.center_points = (self.center_points[:,None,:] + (offset*2-1) * half_voxel).reshape(-1, 3)
        self.n_voxels = self.n_voxels * n_subvox
        self.grid_shape = self.grid_shape * 2
        self.occupancy = self.occupancy[...,None].repeat_interleave(n_subvox, -1).reshape(-1)
        self.max_ray_hit = self.get_max_ray_hit()
        if self.use_corner:
            center_coords = (2*old_center_coords[...,None,:] + offset).reshape(-1, 3) # [8N, 3] # x2
            # <==> discretize_points(self.center_points, self.voxel_size) # [8N ,3]
            corner_coords = (center_coords[...,None,:] + offset).reshape(-1, 3) # [64N, 3]
            unique_corners, center2corner = torch.unique(corner_coords, dim=0, sorted=True, return_inverse=True)
            self.n_corners = self.n_corners * 0 + unique_corners.shape[0]
            old_ct2cn = self.center2corner
            self.center2corner = center2corner.reshape(-1, n_subvox)
            if feats is not None:
                cn2oldct = center2corner.new_zeros(self.n_corners).scatter_(
                    0, center2corner, torch.arange(corner_coords.shape[0], device=feats.device) // n_subvox**2)
                feats_idx = old_ct2cn[cn2oldct] # [N_cn, 8]
                _feats = feats[feats_idx] # [N_cn, 8, D_f]
                new_feats = trilinear_interp(unique_corners-1, 2*old_center_coords[cn2oldct], _feats, 2., offset)
                return new_feats

    def get_max_ray_hit(self):
        # keep min max voxels, for ray_intersection
        min_voxel = self.center_points.min(0)[0]
        max_voxel = self.center_points.max(0)[0]
        aabb_box = ((max_voxel - min_voxel) / self.voxel_size).round().long() + 1
        max_ray_hit = min(aabb_box.sum(), self.n_voxels)
        return max_ray_hit
    
    def load_adjustment(self, n_voxels, grid_shape):
        self.center_points = self.center_points.new_empty(n_voxels, 3)
        self.center2corner = self.center2corner.new_empty(n_voxels, 8)
        self.occupancy = self.occupancy.new_empty(torch.tensor(grid_shape).prod())

    def get_edge(self):
        NotImplemented
        # TODO

class VoxelGridFusion(Explicit3D):
    '''
    Let's start with a simple voxel grid.
    '''
    def __init__(
        self,
        bbox: Tensor,
        voxel_size: float,
        occ: Tensor,
        use_corner: bool=True,
        ):
        '''
        bbox2voxel: https://github.com/facebookresearch/NSVF/fairnr/modules/encoder.py#L1053
        bbox: array [min_x,y,z, max_x,y,z]

        x represents center, O represents corner
            O O O O O
             x x x x 
            O O O O O
             x x x x 
            O O O O O
        Given a center x's coords [i,j,k]. its corners' coords are [i,j,k] + {0,1}^3
        '''
        super().__init__()
        self.use_corner = use_corner
        self.bbox = bbox
        v_min, v_max = bbox[:3], bbox[3:]
        steps = ((v_max - v_min) / voxel_size).round().long()
        # note the difference between torch.meshgrid and np.meshgrid.
        center_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
        center_points = (center_coords * voxel_size + v_min).reshape(-1, 3) # start from lower bound
        # self.register_buffer('center_coords', center_coords)
        n_voxels = center_points.shape[0]
        occupancy = torch.ones(n_voxels, dtype=torch.bool) # occupancy's length unchanges unless splitting

        # corner points
        if use_corner:
            corner_shape = steps+1
            n_corners = corner_shape.prod().item()
            offset = offset_points().long() # [8, 3]
            corner1d = torch.arange(n_corners).reshape(corner_shape.tolist()) 
            center2corner = (center_coords[...,None,:] + offset).reshape(-1, 8, 3) # [..., 8,3]
            center2corner = corner1d[center2corner[...,0], center2corner[...,1], center2corner[...,2]] # [..., 8]
            self.register_buffer('center2corner', center2corner)
            self.register_buffer('n_corners', torch.tensor(n_corners))
        
        # keep min max voxels, for ray_intersection
        max_ray_hit = min(steps.sum().item(), n_voxels)
        # register_buffer for saving and loading.
        self.register_buffer('occupancy', occupancy)
        self.register_buffer('grid_shape', steps) # self.grid_shape = steps
        self.register_buffer('center_points', center_points)
        self.register_buffer('n_voxels', torch.tensor(n_voxels))
        self.register_buffer('max_ray_hit', torch.tensor(max_ray_hit))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('unique_cor', torch.tensor(n_voxels))
        interp_offset = torch.stack(torch.meshgrid([torch.tensor([0.,1.])]*3),-1).reshape(-1,3)
        self.register_buffer('interp_offset', interp_offset)
        self.pruning(occ.flatten(), offset)


    def ray_intersect(self, rays_o: Tensor, rays_d: Tensor):
        '''
        Args:
            rays_o, Tensor, (N_rays, 3)
            rays_d, Tensor, (N_rays, 3)
        Return:
            pts_idx, Tensor, (N_rays, max_hit)
            t_near, t_far    (N_rays, max_hit)
        '''
        pts_idx_1d, t_near, t_far = _ext.aabb_intersect(
            rays_o.contiguous(), rays_d.contiguous(), 
            self.center_points.contiguous(), self.voxel_size, self.max_ray_hit)
        t_near.masked_fill_(pts_idx_1d.eq(-1), MAX_DEPTH)
        t_near, sort_idx = t_near.sort(dim=-1)
        t_far = t_far.gather(-1, sort_idx)
        pts_idx_1d = pts_idx_1d.gather(-1, sort_idx)
        hits = pts_idx_1d.ne(-1).any(-1)
        return pts_idx_1d, t_near, t_far, hits

    # def get_corner_points(self, center_idx):
    #     corner_idx = self.center2corner[center_idx] # [..., 8]
    #     return self.corner_points[corner_idx] # [..., 8, 3]

    def pruning(self, keep, offset):
        n_vox_left = keep.sum()
        if n_vox_left > 0 and n_vox_left < keep.shape[0]:
            self.center_points = self.center_points[keep].contiguous()
            self.occupancy.masked_scatter_(self.occupancy, keep)
            self.n_voxels = n_vox_left
            self.max_ray_hit = self.get_max_ray_hit()
            if self.use_corner:
                c2corner_idx = self.center2corner[keep] # [num of voxel, 8]
                corner_pts = self.center_points.unsqueeze(1) + ((offset.float()-0.5)*self.voxel_size).unsqueeze(0) # [num of voxel, 8, 3]
                corner_idx, center2corner = c2corner_idx.unique(sorted=True, return_inverse=True) # [num of corner] and [num of voxel, 8]
                remap = center2corner.new_zeros(corner_idx.shape[0]).scatter_(0, center2corner.reshape(-1), torch.arange(self.n_voxels*8))
                self.unique_cor = corner_pts.reshape(-1,3)[remap]
                self.center2corner = center2corner.contiguous()
                self.n_corners = self.n_corners * 0 + corner_idx.shape[0]


    def splitting(self, feats: Optional[Tensor]=None):
        offset = offset_points(device=self.center_points.device).long() # [8, 3] scale [0,1]
        n_subvox = offset.shape[0] # 8
        old_center_coords = discretize_points(self.center_points, self.voxel_size) # [N ,3]

        self.voxel_size *= 0.5
        half_voxel = self.voxel_size * 0.5
        self.center_points = (self.center_points[:,None,:] + (offset*2-1) * half_voxel).reshape(-1, 3)
        self.n_voxels = self.n_voxels * n_subvox
        self.grid_shape = self.grid_shape * 2
        self.occupancy = self.occupancy[...,None].repeat_interleave(n_subvox, -1).reshape(-1)
        self.max_ray_hit = self.get_max_ray_hit()
        if self.use_corner:
            center_coords = (2*old_center_coords[...,None,:] + offset).reshape(-1, 3) # [8N, 3] # x2
            # <==> discretize_points(self.center_points, self.voxel_size) # [8N ,3]
            corner_coords = (center_coords[...,None,:] + offset).reshape(-1, 3) # [64N, 3]
            unique_corners, center2corner = torch.unique(corner_coords, dim=0, sorted=True, return_inverse=True)
            self.n_corners = self.n_corners * 0 + unique_corners.shape[0]
            old_ct2cn = self.center2corner
            self.center2corner = center2corner.reshape(-1, n_subvox)
            if feats is not None:
                cn2oldct = center2corner.new_zeros(self.n_corners).scatter_(
                    0, center2corner, torch.arange(corner_coords.shape[0], device=feats.device) // n_subvox**2)
                feats_idx = old_ct2cn[cn2oldct] # [N_cn, 8]
                _feats = feats[feats_idx] # [N_cn, 8, D_f]
                new_feats = trilinear_interp(unique_corners-1, 2*old_center_coords[cn2oldct], _feats, 2., offset)
                return new_feats

    def get_max_ray_hit(self):
        # keep min max voxels, for ray_intersection
        min_voxel = self.center_points.min(0)[0]
        max_voxel = self.center_points.max(0)[0]
        aabb_box = ((max_voxel - min_voxel) / self.voxel_size).round().long() + 1
        max_ray_hit = min(aabb_box.max(), self.n_voxels)
        return max_ray_hit
    
    def load_adjustment(self, n_voxels, grid_shape):
        self.center_points = self.center_points.new_empty(n_voxels, 3)
        self.center2corner = self.center2corner.new_empty(n_voxels, 8)
        self.occupancy = self.occupancy.new_empty(torch.tensor(grid_shape).prod())

    def get_edge(self):
        NotImplemented
        # TODO

    def update_feat(self, feats, unq_index):
        self.feats_vol[unq_index] = feats
        self.feats_vol = self.feats_vol.reshape(1,self.grid_shape[0],self.grid_shape[1],self.grid_shape[2],-1).permute(0,4,3,2,1)

    def back_project(self, feats, KRcam):
        n_views, c, h, w = feats.shape

        rs_grid = self.unique_cor.unsqueeze(0).expand(n_views, -1, -1).permute(0,2,1) # Nv 3 N
        rs_grid = torch.cat([rs_grid, torch.ones((n_views,1,self.n_corners),device=rs_grid.device)], dim=1) #9 4 N
        im_grid, mask = self.project_grid(rs_grid, KRcam)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        pt_feats = F.grid_sample(feats, im_grid, padding_mode='zeros', align_corners=True)

        pt_feats = pt_feats.view(n_views, c , -1)
        mask = mask.view(n_views, -1)
        # remove nan
        pt_feats[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        # aggregate multi view
        features = pt_feats.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()
        self.voxel_embeder = features # N 64

    def get_feats(self, pts: Tensor, p2v_idx: Tensor, vox_rep: Explicit3D, per_voxel: bool=False):
        '''
        if per_voxel is False
            Args:
                pts: Tensor, [N_pts, 3]
                p2v_idx: Tensor, [N_pts] 
                    mapping pts to voxel idx, Note voxel idx are 1D index, and -1 idx should be masked out.
        if per_voxel is True
            Args:
                pts: Tensor, relative points in one voxel, [N_pts_per_vox, 3], scale [0,1]
                p2v_idx: Tensor, [N_vox]
            Return:
                embeds, [N_vox, N_pts_per_vox, embed_dim]
        '''
        # get corresponding voxel embeddings
        p2v_idx = p2v_idx.long()
        center_pts = self.center_points[p2v_idx] # (N, 3)
        corner_idx = self.center2corner[p2v_idx] # (N, 8)
        embeds = self.voxel_embeder[corner_idx] # (N, 8, embed_dim)
        
        # interpolation
        if not per_voxel:
            interp_embeds = trilinear_interp(pts, center_pts, embeds, 
                self.voxel_size, self.interp_offset)
        else:
            pts = pts[...,None,:] # [N_pts_per_vox, 1, 3]
            r = (pts*self.interp_offset + (1-pts)*(1-self.interp_offset))\
                    .prod(dim=-1, keepdim=True)[None,:] # [1, N_ppv, 8, 1]
            interp_embeds = (embeds[:,None,:] * r).sum(-2) # [N_v, N_ppv, embed_dim]
        return interp_embeds

    @staticmethod
    def project_grid(pts:Tensor, KRcam:Tensor, H:int=480, W:int=640):
        im_p = KRcam @ pts
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([im_x*2/W-1, im_y*2/H-1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)
        return im_grid, mask


class TSDF_Fusion(Explicit3D):
    def __init__(self, vol_bnds, voxel_size, margin=2, voxel_dim=None, origin=None,
        device=torch.device('cpu'), use_corner=True,use_metric_depth=True):
        super().__init__()
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        self.use_metric_depth = use_metric_depth
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._sdf_trunc = margin * self._voxel_size  # truncation on SDF
        self.device = device
        self._const = 256 * 256
        self._integrate_func = integrate

        if voxel_dim is not None:
            # Adjust volume bounds
            self._vol_origin = origin
            self._vol_dim = voxel_dim.long()
            self._num_voxels = torch.prod(self._vol_dim).item()
        else:
            # Adjust volume bounds and ensure C-order contiguous
            self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
                order='C').astype(int)
            self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
            self._vol_origin = torch.from_numpy(self._vol_bnds[:, 0].copy(order='C').astype(np.float32)).to(device)

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
        )
        self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)

        # Convert voxel coordinates to world coordinates
        center_points = self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords) # N, 3
        
        self._bds = torch.cat(((self._world_c[0]-self._voxel_size*0.5),(self._world_c[-1]+self._voxel_size*0.5)))
        self._world_c = torch.cat([
            self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1)
        self._pix_mask = []
        self.grid_shape = self._vol_dim
        self.reset()
        # print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
        # print("[*] num voxels: {:,}".format(self._num_voxels))

        # init voxel related
        if use_corner:
            center_coords = self._vox_coords
            self.center_points = center_points.reshape(-1,3)
            self.n_voxels = center_points.shape[0]
            corner_shape = self._vol_dim+1
            self.n_corners = corner_shape.prod().item()
            self.offset = offset_points().long().to(self.device) # [8, 3]
            corner1d = torch.arange(self.n_corners).reshape(corner_shape.tolist()).to(self.device)
            center2corner = (center_coords[...,None,:] + self.offset).reshape(-1, 8, 3) # [..., 8,3]
            self.center2corner = corner1d[center2corner[...,0], center2corner[...,1], center2corner[...,2]] # [..., 8]


    def reset(self):
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._color_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._pix_mask = []

    def _linearize_id(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        return xyz[:, 2] + self._vol_dim[-1] * xyz[:, 1] + (self._vol_dim[-1] * self._vol_dim[-2]) * xyz[:, 0]

    def _unlinearize_id(self, idx: torch.Tensor):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        return torch.stack([idx // (self._vol_dim[1] * self._vol_dim[2]),
                            (idx // self._vol_dim[2]) % self._vol_dim[1],
                            idx % self._vol_dim[2]], dim=-1)

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = cam_pose.float().to(self.device)
        cam_intr = cam_intr.float().to(self.device)
        depth_im = depth_im.float().to(self.device)
        im_h, im_w = depth_im.shape
        weight_vol, tsdf_vol, depth_mask, pixel_xyz_world,metric_dp = self._integrate_func(
            depth_im,
            cam_intr,
            cam_pose,
            obs_weight,
            self._world_c,
            self._vox_coords,
            self._weight_vol,
            self._tsdf_vol,
            self._sdf_trunc,
            im_h, im_w, use_metric_depth=self.use_metric_depth
        )
        vox_index = self._linearize_id(torch.round((pixel_xyz_world - self._vol_origin)/self._voxel_size))
        valid =torch.logical_and(vox_index>=0.,vox_index<self._vol_dim[0]) # only use pix in volume.
        valid_mask = torch.logical_and((valid.sum(dim=-1,keepdim=False)-2).bool(), depth_mask.squeeze()) # and the depth mask. 
        vox_index[~valid_mask]=-1
        # self._pix_mask.append(torch.cat((valid_mask, vox_index),dim=-1))
        self._pix_mask.append(vox_index)
        self._weight_vol = weight_vol
        self._tsdf_vol = tsdf_vol
        return metric_dp

    def get_pix_mask(self): 
        return self._pix_mask, self._bds

    def get_volume(self):
        return self._tsdf_vol, self._weight_vol

    @property # make it read only
    def sdf_trunc(self):
        return self._sdf_trunc

    @property 
    def voxel_size(self):
        return self._voxel_size

    def get_max_ray_hit(self):
        # keep min max voxels, for ray_intersection
        min_voxel = self.center_points.min(0)[0]
        max_voxel = self.center_points.max(0)[0]
        aabb_box = ((max_voxel - min_voxel) / self.voxel_size).round().long() + 1
        max_ray_hit = min(aabb_box.sum(), self.n_voxels)
        return max_ray_hit

    def cal_corner(self, use_corner=True):
        '''
            After fusion, cal the remaining corner
        '''
        self.occ_vol = torch.zeros_like(self._tsdf_vol).bool()
        self.occ_vol[(self._tsdf_vol < 0.999) & (self._tsdf_vol > -0.999) 
            & (self._weight_vol >= 1)] = True
        n_vox_left = self.occ_vol.sum()
        if n_vox_left > 0 and n_vox_left < self.n_corners:
            self.center_points = self.center_points[self.occ_vol.flatten()].contiguous()
            self.n_voxels = n_vox_left
            self.max_ray_hit = self.get_max_ray_hit()
            if use_corner:
                c2corner_idx = self.center2corner[self.occ_vol.flatten()] # [num of voxel, 8]
                corner_pts = self.center_points.unsqueeze(1) + ((self.offset.float()-0.5)*self.voxel_size).unsqueeze(0) # [num of voxel, 8, 3]
                corner_idx, center2corner = c2corner_idx.unique(sorted=True, return_inverse=True) # [num of corner] and [num of voxel, 8]
                remap = center2corner.new_zeros(corner_idx.shape[0]).scatter_(0, center2corner.reshape(-1), torch.arange(self.n_voxels*8,device=self.device))
                self.unique_cor = corner_pts.reshape(-1,3)[remap]
                self.center2corner = center2corner.contiguous()
                self.n_corners = self.n_corners * 0 + corner_idx.shape[0]

    def ray_intersect(self, rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''
        Args:
            rays_o, Tensor, (N_rays, 3)
            rays_d, Tensor, (N_rays, 3)
        Return:
            pts_idx, Tensor, (N_rays, max_hit)
            t_near, t_far    (N_rays, max_hit)
        '''
        pts_idx_1d, t_near, t_far = _ext.aabb_intersect(
            rays_o.contiguous(), rays_d.contiguous(), 
            self.center_points.contiguous(), self._voxel_size, self.max_ray_hit)
        t_near.masked_fill_(pts_idx_1d.eq(-1), MAX_DEPTH)
        t_near, sort_idx = t_near.sort(dim=-1)
        t_far = t_far.gather(-1, sort_idx)
        pts_idx_1d = pts_idx_1d.gather(-1, sort_idx)
        hits = pts_idx_1d.ne(-1).any(-1)
        return pts_idx_1d, t_near, t_far, hits

    def get_pts(self,pt_per_edge):
        sparse_pts = generate_pts_vox(self.center_points, self._voxel_size, pt_per_edge=pt_per_edge)
        voxel_index = torch.arange(0,self.n_voxels).to(sparse_pts.device)
        return sparse_pts, voxel_index, self.occ_vol

    def get_corner(self,pt_per_edge):
        '''
            get full pts from voxel-grid
        '''
        assert isinstance(pt_per_edge, int)

        sh = self._vol_dim*pt_per_edge # shape
        corner = self._vol_origin.cpu() + torch.tensor([-0.5,-0.5,-0.5])*self._voxel_size # vol corner

        if pt_per_edge <= 1:
            world_c = self._world_c[:,:3] 
            pt_mask = self.occ_vol.flatten()
            voxel_index = torch.arange(0,self.n_voxels).to(world_c.device)
            # pt_mask.masked_fill_(pt_mask, voxel_index)
            pt_idx = torch.ones_like(self.occ_vol.flatten())*-1
            pt_idx[pt_mask] = voxel_index
            
        else:
            xv, yv, zv = torch.meshgrid(
                torch.arange(0, self._vol_dim[0]*pt_per_edge), # start from -0.5,-0.5,-0.5
                torch.arange(0, self._vol_dim[1]*pt_per_edge),
                torch.arange(0, self._vol_dim[2]*pt_per_edge),
            )
            vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long()
            world_c = corner + (self._voxel_size * vox_coords)/pt_per_edge

            pt_mask = torch.ones_like(self.occ_vol.flatten(),device='cpu')*-1
            pt_mask[self.occ_vol.flatten()>0]=torch.arange(0,self.n_voxels)
            ids = self._linearize_id((vox_coords/pt_per_edge).long())
            pt_idx = pt_mask[ids]


        ### verify the range of lc
        # pt_mask = pt_idx.ne(-1)
        # index = torch.randint(0,90000,[10000])
        # lc = (world_c[pt_mask][index] - self.center_points[pt_idx[pt_mask][index]].cpu())/self._voxel_size
        # print(lc.max())
        # print(lc.min())
        return world_c, pt_idx, sh, corner

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol.cpu().numpy(), level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size+self._vol_origin.cpu().numpy()  # voxel grid coordinates to world coordinates

        # Get vertex colors
        # rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
        # colors_b = np.floor(rgb_vals/self._color_const)
        # colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
        # colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
        # colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        # colors = colors.astype(np.uint8)
        return verts, faces, norms

    def save_mesh(self,path=None):
        verts, faces, norms = self.get_mesh()
        if path is None:
            path = './data/mesh.ply'
        meshwrite(path, verts,faces,norms)
        print(f"mesh saved in {path}")