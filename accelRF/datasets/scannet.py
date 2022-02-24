# coding=utf-8
import copy
import statistics
import sys, os
import pickle

import cv2
import imageio
from matplotlib.pyplot import axis
import numpy as np
import torch
from PIL import Image,ImageOps

from .fusion import TSDFVolumeTorch
from .base import BaseDataset
# from accelRF.datasets import fusion

__all__ = ['Single_SCANNET','SCANNET']

def save_point_cloud(filename, xyz, rgb=None):
    from plyfile import PlyData,PlyElement
    if rgb is None:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2], rgb[k, 0], rgb[k, 1], rgb[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f6'), ('y', 'f6'), ('z', 'f6'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(filename)
    # from fairseq import pdb; pdb.set_trace()
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(open(filename, 'wb'))

def save_mesh(filename, xyz, faces, norms=None, colors=None):
    from plyfile import PlyData,PlyElement
    faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
    if colors is None:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2], colors[k, 0], colors[k, 1], colors[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f6'), ('y', 'f6'), ('z', 'f6'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    if norms is not None:
        norms = PlyElement.describe(norms, "normals")
        PlyData([PlyElement.describe(faces, 'face'), \
                PlyElement.describe(vertex, 'vertex'), \
                norms ]).write(open(filename, 'wb'))
    else:
        PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(faces, 'face')]).write(open(filename, 'wb'))

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def spherify_poses(poses, bds):
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, bds

def resize_images(images, H, W, interpolation=cv2.INTER_LINEAR):
    resized = np.zeros((images.shape[0], H, W, images.shape[3]), dtype=images.dtype)
    for i, img in enumerate(images):
        r = cv2.resize(img, (W, H), interpolation=interpolation)
        if images.shape[3] == 1:
            r = r[..., np.newaxis]
        resized[i] = r
    return resized

def pad_scannet(img, intrinsics):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2))
        intrinsics[1, 2] += 2
    return img, intrinsics


class ResizeImage(object):
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for i, im in enumerate(data['imgs']):
            im, intrinsics = pad_scannet(im, data['intrinsics'][i])
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)
            intrinsics[0, :] /= (w / self.size[0])
            intrinsics[1, :] /= (h / self.size[1])

            data['imgs'][i] = np.array(im, dtype=np.float32)
            data['intrinsics'][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    if isinstance(xyz, torch.Tensor):
        xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1))], dim=1)
        xyz_t_h = (transform @ xyz_h.T).T
    else:
        xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
        xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

def get_view_frustum(max_depth, size, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    view_frust_pts = torch.stack([
        (torch.tensor([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * torch.tensor(
            [0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (torch.tensor([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * torch.tensor(
            [0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        torch.tensor([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts

def get_view_frustum_np(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
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

def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

class Single_SCANNET(BaseDataset):
    def __init__(
        self, root: str, scene: str, factor: int=2, recenter: bool=True, use_ndc: bool=True,
        bd_factor: float=.75, spherify: bool=False, n_holdout: int=0, n_views: int=9, single_fragment: int=0,
        H: int=480,W: int=640, mode: str='train', voxel_size: float=.04, max_depth: float=5.0, voxel_dim: int=160
    ) -> None:
        super().__init__()
        self.spherify = spherify # not used.
        self.use_ndc = use_ndc
        self.H = H
        self.W = W
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.voxel_dim = [voxel_dim,voxel_dim,voxel_dim]
        vol_origin_coor = torch.tensor([0.,0.,0.]) # coord origin [000] or [voxel_size/2,...]
        # load meta
        self.n_views = n_views
        self.mode = mode
        self.tsdf_file = 'all_tsdf_{}'.format(n_views)
        self.datapath = os.path.join(root, scene) # root: data, scene: scannet
        self.metas = self.build_list()[single_fragment]

        # load fragments,specify H, W. 
        imgs,depth,c2w_list,intrinsics_list=[],[],[],[]
        
        tsdf = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), self.metas['scene'])
        for i, vid in enumerate(self.metas['image_ids']):
            
            # load intrinsics and c2w
            intrinsics, c2w = self.read_cam_file(os.path.join(self.datapath, self.tsdf_file, self.metas['scene']), vid)
            # load images
            img,intrinsics = self.read_img(
                    os.path.join(self.datapath, self.tsdf_file, self.metas['scene'], 'color', '{}.jpg'.format(vid)),H=H,W=W,intrinsics=intrinsics)
            imgs.append(img)
            intrinsics_list.append(intrinsics)
            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.tsdf_file, self.metas['scene'], 'depth', '{}.png'.format(vid)))
            )
            c2w_list.append(c2w)

        intrinsics = np.stack(intrinsics_list).astype(np.float32)
        poses = np.stack(c2w_list).astype(np.float32)
        imgs = np.stack(imgs).transpose([0,3,1,2])
        depths = np.stack(depth)
        

        print("loaded! images shape {}, pose shape {}, intrinsic shape {}".format(
            imgs.shape, poses.shape, intrinsics.shape
        ))
        # convert np array to torch tensor
        self.focal = (intrinsics[0][0][0], intrinsics[0][1][1]) # fx and fy
        
        self.data = {
            'imgs':torch.from_numpy(imgs),
            'depths':torch.from_numpy(depths),
            'intrinsics':torch.from_numpy(intrinsics),
            'poses':torch.from_numpy(poses), # [N, 4, 4]
            'tsdf_all':torch.from_numpy(tsdf),
            'old_vol_origin':torch.from_numpy(self.metas['vol_origin']),
            'vol_origin_coor': vol_origin_coor
        }
        self.K = self.data['intrinsics'][0]
        # fusion
        self.forward()

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        c2w = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, c2w

    def read_img(self, filepath,H,W,intrinsics):
        img = Image.open(filepath)
        if img.width != W: # need resize image
            im, intrinsics = pad_scannet(img, intrinsics)
            w,h = im.size
            im = im.resize((W,H), Image.BILINEAR)
            intrinsics[0, :] /= (w / W)
            intrinsics[1, :] /= (h / H)
        else: #image is right, K is not right
            intrinsics[0, :] /= (1296 / W)
            intrinsics[1, :] /= (968 / H)
        return np.array(im, dtype=np.float32)/255., intrinsics

    def read_depth(self, filepath, maxdepth=5.0):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > maxdepth] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer0.npz'),
                                    allow_pickle=True)

        return full_tsdf.f.arr_0

    def forward(self, align_corners=False):
        '''
        If want to use origin tsdf results, please refer to:
        https://github.com/zju3dv/NeuralRecon/blob/cd047e2356f68f60adfb923f33572d712f64b57a/datasets/transforms.py#L276
        '''
        bnds = torch.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf

        for i in range(self.data['imgs'].shape[0]):

            cam_intr = self.data['intrinsics'][i]
            cam_pose = self.data['poses'][i]
            view_frust_pts = get_view_frustum(self.max_depth, (self.H, self.W), cam_intr, cam_pose)
            bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
            bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])

        # -------adjust volume bounds------- TODO: this implement is necessary?
        num_layers = 3
        center = (torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2, -0.2)) - \
                                self.data['vol_origin_coor'])/ self.voxel_size
        center[:2] = torch.round(center[:2] / 2 ** num_layers) * 2 ** num_layers 
        center[2] = torch.floor(center[2] / 2 ** num_layers) * 2 ** num_layers
        origin = torch.zeros_like(center)
        origin[:2] = center[:2] - torch.tensor(self.voxel_dim[:2]) // 2
        origin[2] = center[2]
        vol_origin_partial = origin * self.voxel_size + self.data['vol_origin_coor']
        self.data['vol_origin_partial'] = vol_origin_partial

        # ------get partial tsdf and occupancy ground truth--------
        if self.data['tsdf_all'] is not None:
            # -------------grid coordinates------------------
            old_origin = self.data['old_vol_origin'].view(1, 3)
            # x, y, z = self.voxel_dim
            coords = coordinates(self.voxel_dim, device=old_origin.device)
            world = coords.type(torch.float) * self.voxel_size + vol_origin_partial.view(3, 1) 
            world = torch.cat((world, torch.ones_like(world[:1])), dim=0)
            # world = transform[:3, :] @ world
            coords = (world[:3] - old_origin.T) / self.voxel_size

            l = 0
            # ------get partial tsdf and occ-------
            vol_dim_s = torch.tensor(self.voxel_dim) // 2 ** l
            tsdf_vol = TSDFVolumeTorch(vol_dim_s, vol_origin_partial,
                                        voxel_size=self.voxel_size * 2 ** l, margin=3)
            for i in range(self.data['imgs'].shape[0]):
                depth_im = self.data['depths'][i]
                cam_intr = self.data['intrinsics'][i]
                cam_pose = self.data['poses'][i]

                tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)
            self.data['pix_mask'],self.data['bds'] = tsdf_vol.get_pix_mask() # list [307200, 1](true for valid)[307200,3](valid index)
            tsdf_vol, weight_vol = tsdf_vol.get_volume()
            occ_vol = torch.zeros_like(tsdf_vol).bool()
            occ_vol[(tsdf_vol < 0.999) & (tsdf_vol > -0.999) & (weight_vol >= 1)] = True # NOTE: this weight_vol in original setting is no bigger than 1.

            self.data['occ_vol'] = occ_vol
            self.data['tsdf_vol'] = tsdf_vol
            self.data.pop('tsdf_all')
            ## vis the tsdf
            # from mcubes import marching_cubes
            # vertices, faces = marching_cubes(tsdf_vol.numpy(),0.)
            # save_mesh('tsdf.ply',vertices,faces)

    def get_train_set(self):
        return copy.copy(self)

    def get_vox(self):
        return self.data['occ_vol'],self.data['bds']

    def get_HWK(self):
        return self.H, self.W, self.K

    def __len__(self):
        return self.data['poses'].shape[0]

    def get_test_set(self):
        self.get_test_part()
        return copy.copy(self)

    def get_test_part(self,H=480,W=640):
        '''
            get other poses for test.
        '''
        names = np.arange(self.metas['image_ids'][0], self.metas['image_ids'][-1]+1)
        imgs,depth,c2w_list,intrinsics_list=[],[],[],[]
        for vid in names:
            
            # load intrinsics and c2w
            intrinsics, c2w = self.read_cam_file(os.path.join(self.datapath, self.tsdf_file, self.metas['scene']), vid)
            # load images
            img,intrinsics = self.read_img(
                    os.path.join(self.datapath, self.tsdf_file, self.metas['scene'], 'color', '{}.jpg'.format(vid)),H=H,W=W,intrinsics=intrinsics)
            imgs.append(img)
            intrinsics_list.append(intrinsics)
            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.tsdf_file, self.metas['scene'], 'depth', '{}.png'.format(vid)))
            )
            c2w_list.append(c2w)

        intrinsics = np.stack(intrinsics_list).astype(np.float32)
        poses = np.stack(c2w_list).astype(np.float32)
        imgs = np.stack(imgs).transpose([0,3,1,2])
        depths = np.stack(depth)

        self.data = {
            'imgs':torch.from_numpy(imgs),
            'depths':torch.from_numpy(depths),
            'intrinsics':torch.from_numpy(intrinsics),
            'poses':torch.from_numpy(poses), # [N, 4, 4]
        }

class Room_SCANNET(BaseDataset):
    """
    use for test period
    """

    def __init__(
        self, root: str, scene: str, 
        half_res: bool = False, testskip: int = 1, white_bkgd: bool = False,
        with_bbox: bool = False, n_views: int = 9,H: int=480,W: int=640, mode: str='train', 
        voxel_size: float=.04, max_depth: float=5.0
    )->None:
        super().__init__()
        self.scene = scene
        self.datapath = os.path.join(root, scene)
        self.tsdf_file = 'all_tsdf_{}'.format(n_views)
        self.metas = self.load_specific_meta(self.datapath,self.tsdf_file,'scene0158_00')

        self.H = H
        self.W = W

        tmp = []
        for i in range(len(self.metas)):
            tmp.extend(self.metas[i]['image_ids'])
        imgs,depth,c2w_list,intrinsics_list=[],[],[],[]
        for index, fid in enumerate(tmp):
            # load intrinsics and c2w
            intrinsics, c2w = self.read_cam_file(os.path.join(self.datapath, self.tsdf_file, self.metas[0]['scene']), fid)
            # load images
            img,intrinsics = self.read_img(
                    os.path.join(self.datapath, self.tsdf_file, self.metas[0]['scene'], 'color', '{}.jpg'.format(fid)),H=H,W=W,intrinsics=intrinsics)
            imgs.append(img)
            intrinsics_list.append(intrinsics)
            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.tsdf_file, self.metas[0]['scene'], 'depth', '{}.png'.format(fid)),max_depth))
            c2w_list.append(c2w)

        intrinsics = np.stack(intrinsics_list).astype(np.float32)
        poses = np.stack(c2w_list).astype(np.float32)
        imgs = np.stack(imgs).transpose([0,3,1,2])
        depths = np.stack(depth)
        
        print("loaded! images shape {}, pose shape {}, intrinsic shape {}".format(
            imgs.shape, poses.shape, intrinsics.shape
        ))
        # cal the bds 
        bds = self.cal_bounds(depths, poses, intrinsics[0])
        # since the scannet z is up, limit the bds's min z to -0.2
        bds[2,0] = -0.2
        self.focal = (intrinsics[0][0][0], intrinsics[0][1][1]) # fx and fy
        # convert np array to torch tensor
        self.data = {
            'gt_img':torch.from_numpy(imgs), # NCHW
            'depths':torch.from_numpy(depths),
            'intrinsics':torch.from_numpy(intrinsics),
            'poses':torch.from_numpy(poses), # [N, 4, 4]
            'old_vol_origin':torch.from_numpy(self.metas[0]['vol_origin']),
            'bds': bds
        }
        self.K = self.data['intrinsics'][0]
        


    def build_list(self, datapath, tsdf_file, mode):
        with open(os.path.join(datapath, tsdf_file, 'fragments_{}.pkl'.format(mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def load_specific_meta(self,datapath, tsdf_file, scene_id):
        with open(os.path.join(datapath, tsdf_file, scene_id, 'fragments.pkl'), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        c2w = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, c2w

    def read_img(self, filepath,H,W,intrinsics):
        img = Image.open(filepath)
        if img.width != W: # need resize image
            im, intrinsics = pad_scannet(img, intrinsics)
            w,h = im.size
            im = im.resize((W,H), Image.BILINEAR)
            intrinsics[0, :] /= (w / W)
            intrinsics[1, :] /= (h / H)
        else: #image is right, K is not right
            intrinsics[0, :] /= (1296 / W)
            intrinsics[1, :] /= (968 / H)
        return np.array(im, dtype=np.float32)/255., intrinsics

    def read_depth(self, filepath, maxdepth=5.0):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > maxdepth] = 0
        return depth_im

    def get_render_set(self, n_frame: int=40, phi: float=30.0, radius: float=4.0):
        # render_poses = torch.stack([pose_spherical(angle, -phi, radius)
        #                         for angle in np.linspace(-180,180,n_frame+1)[:-1]], 0)
        render_set = copy.copy(self)
        # render_set.imgs = None
        # render_set.poses = render_poses
        return render_set

    @staticmethod
    def cal_bounds(depths, poses, intrinsic):
        # use get_view_frustum to get bounds
        vol_bnds = np.zeros((3, 2))
        for id in range(depths.shape[0]):
            depth_im = depths[id]
            cam_pose = poses[id]

            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum_np(depth_im, intrinsic, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        return vol_bnds

    def get_bds(self):
        return self.data['bds']
    
    def __len__(self):
        return self.data['poses'].shape[0]

    def get_HWK(self):
        return self.H, self.W, self.K

if __name__=="__main__":
    a = Room_SCANNET('demo_dir','scannet')


    # def get_render_set(self, n_frame: int=120, path_zflat: bool=False):
    #     poses = self.poses.numpy()
    #     if self.spherify:
    #         centroid = np.mean(poses[:,:3,3], 0)
    #         zh = centroid[2]
    #         radcircle = np.sqrt(1-zh**2)
    #         render_poses = []
            
    #         for th in np.linspace(0.,2.*np.pi, n_frame):
    #             camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
    #             up = np.array([0,0,-1.])

    #             vec2 = normalize(camorigin)
    #             vec0 = normalize(np.cross(vec2, up))
    #             vec1 = normalize(np.cross(vec2, vec0))
    #             pos = camorigin
    #             p = np.stack([vec0, vec1, vec2, pos], 1)
    #             render_poses.append(p)

    #         render_poses = np.stack(render_poses, 0).astype(np.float32) # [N, 3, 4]
    #     else:
    #         c2w = poses_avg(poses)
    #         print('recentered', c2w.shape, c2w[:3,:4])
    #         ## Get spiral
    #         # Get average pose
    #         up = normalize(poses[:, :3, 1].sum(0))
    #         # Find a reasonable "focus depth" for this dataset
    #         close_depth, inf_depth = self.bds.min()*.9, self.bds.max()*5.
    #         dt = .75
    #         mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    #         focal = mean_dz

    #         # Get radii for spiral path
    #         shrink_factor = .8
    #         zdelta = close_depth * .2
    #         tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    #         rads = np.percentile(np.abs(tt), 90, 0)
    #         c2w_path = c2w
    #         N_rots = 2
    #         if path_zflat:
    #             # zloc = np.percentile(tt, 10, 0)[2]
    #             zloc = -close_depth * .1
    #             c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
    #             rads[2] = 0.
    #             N_rots = 1
    #             n_frame/=2
    #         # Generate poses for spiral path
    #         render_poses = render_path_spiral(
    #             c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=n_frame)
    #         render_poses = np.array(render_poses).astype(np.float32)
    #     render_set = copy.copy(self)
    #     render_set.imgs = None
    #     render_set.poses = torch.from_numpy(render_poses)
    #     return render_set

