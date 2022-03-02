from turtle import forward
from typing import Callable, Dict, List, Optional, Tuple
from cv2 import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from accelRF.rep import Explicit3D
from accelRF.rep.voxel_grid import meshwrite
from accelRF.render.nerf_render import volumetric_rendering, volumetric_rendering_sdf
from accelRF.rep.utils import offset_points
# from mcubes import marching_cubes,export_obj
from skimage import measure

def masked_scatter(mask: torch.BoolTensor, x: Tensor):
    mask_shape = mask.shape
    if x.dim() == 1:
        return x.new_zeros(*mask_shape).masked_scatter(mask, x)
    return x.new_zeros(*mask_shape, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(*mask_shape, x.size(-1)), x)

def fill_in(shape, hits: Tensor, input: Tensor, initial=0.):
    if torch.Size(shape) == input.shape:
        return input   # shape is the same no need to fill

    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    if input is not None:
        if len(shape) == 1:
            output = output.masked_scatter(hits, input)
        else:
            output = output.masked_scatter(hits.unsqueeze(-1).expand(*shape), input)
    return output

class NerfusionRendererTest(nn.Module):
    def __init__(
        self,
        pts_embedder: Optional[nn.Module],
        view_embedder: Optional[nn.Module],
        voxel_embedder: nn.Module,
        model: nn.Module,
        point_sampler: nn.Module,
        vox_rep: Explicit3D,
        early_stop_thres: float=0,
        bg_color: Optional[nn.Module]=None,
        white_bkgd: bool=False,
        min_color: int=-1, # note gt_rbg is [0,1] by default. the final out rgb always [0,1] scale
        fwd_steps: int=4,
        chunk: int=1024*16
    ):
        '''
        Args:
            early_stop_thres: values in the form of $T$, where $T = \exp(-\sum_i\sigma_i\delta_i)$
        '''
        super().__init__()
        self.pts_embedder = pts_embedder if pts_embedder is not None else nn.Identity()
        self.view_embedder = view_embedder if view_embedder is not None else nn.Identity()
        self.voxel_embedder = voxel_embedder
        self.model = model
        self.point_sampler = point_sampler
        self.vox_rep = vox_rep
        self.early_stop = (early_stop_thres > 0)
        self.early_stop_thres = -np.log(early_stop_thres) if self.early_stop else 0
        self.chunk = chunk
        self.fwd_steps = fwd_steps
        self.bg_color = bg_color
        self.white_bkgd = (white_bkgd and bg_color is None)
        self.min_color = min_color
        self.ext_chunk = self.chunk//16


    def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, 
        t_near: Tensor, t_far: Tensor, ray_hits: Tensor):
        '''
        Args:
            rays_o, rays_d: [N_rays, 3]
            vox_idx, t_near, t_far: [N_rays, max_hits]
            hits: [N_rays]
        '''
        n_rays = ray_hits.shape[0]
        if n_rays > self.chunk:
            ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk], vox_idx[ci:ci+self.chunk], 
                                t_near[ci:ci+self.chunk], t_far[ci:ci+self.chunk], ray_hits[ci:ci+self.chunk]) 
                        for ci in range(0, n_rays, self.chunk)]
            ret = {
                k: torch.cat([out[k] for out in ret if out[k] is not None], 0)
                for k in ret[0]
            }
            return ret
        else:
            if ray_hits.sum() > 0:
                vox_idx, t_near, t_far = vox_idx[ray_hits], t_near[ray_hits], t_far[ray_hits] # [n_hits, max_hits]
                rays_o, rays_d = rays_o[ray_hits], rays_d[ray_hits] # [n_hits, 3]
                # point sampling
                sample_out = self.point_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
                pts, p2v_idx, t_vals, dists = sample_out # [n_hits, max_samples] + [3], [], [], []
                mask_pts = p2v_idx.ne(-1)
                max_samples= mask_pts.shape[1]
                n_pts_fwd, start_step = 0, 0
                accum_log_T = 0
                nn_out = []
                for i in range(0, max_samples, self.fwd_steps):
                    n_pts_step = mask_pts[:, i:i+self.fwd_steps].sum()
                    if n_pts_step + n_pts_fwd == 0:
                        break # or continue
                    if (i+self.fwd_steps >= max_samples) or (n_pts_fwd + n_pts_step > self.chunk):
                        l, r = start_step, i+self.fwd_steps
                        out, n_pts = self.step_forward(
                            pts[:,l:r], p2v_idx[:,l:r], rays_d, mask_pts[:,l:r], dists[:,l:r])
                        nn_out.append(out)
                        start_step, n_pts_fwd = r, 0
                        # oops, early_stop relies on free_energy (sigma * dist)
                        if self.early_stop:
                            with torch.no_grad():
                                accum_log_T = accum_log_T + out['free_energy'].sum(1)
                                mask_early_stop = accum_log_T > self.early_stop_thres
                                mask_pts[mask_early_stop] = 0
                    else:
                        n_pts_fwd += n_pts_step
                        
                nn_out = {
                    k: torch.cat([out[k] for out in nn_out], 1) 
                    for k in nn_out[0]
                }
                _max_samples = nn_out['rgb'].shape[1] # _max_sample might smaller than max_samples
                nn_out['rgb'] = nn_out['rgb'].sigmoid() if self.min_color == 0 else (nn_out['rgb'] + 1)/2
                
                r_out = volumetric_rendering(nn_out['rgb'], nn_out['free_energy'], t_vals[...,:_max_samples], 
                            white_bkgd=self.white_bkgd, with_dist=True)   
                ret = {'rgb': r_out['rgb'], "depth":r_out['depth'], }
                if 'feat_n2' in nn_out:
                    ret['regz_term'] = (nn_out['feat_n2'] * r_out['weights']).sum(-1)
            else:
                ret = {'rgb': None, 'disp': None, 'acc': None, "depth": None}
            # fill_in un-hitted holes
            ret['rgb'] = fill_in((n_rays, 3), ray_hits, ret['rgb'], 1.0 if self.white_bkgd else 0.0)
            ret['depth'] = fill_in((n_rays, ), ray_hits, ret['depth'], 0.0)
            if not self.training:
                ret['disp'] = fill_in((n_rays,), ray_hits, r_out['disp'], 0.0)
                ret['acc'] = fill_in((n_rays, ), ray_hits, r_out['acc'], 0.0)
            else:
                ret['weights']=fill_in((n_rays, _max_samples), ray_hits, r_out['weights'], 0.0)
                ret['dists']=fill_in((n_rays, _max_samples), ray_hits, dists, 0.0)
            ret['t_vals']=fill_in((n_rays, _max_samples), ray_hits, t_vals[...,:_max_samples], 0.0)
            if self.bg_color is not None:
                bg_rgb = self.bg_color(ret['rgb'])
                missed = (1 - ret['acc'])
                ret['rgb'] = ret['rgb'] + missed[...,None] * bg_rgb
            
            return ret

    def step_forward(self, pts: Tensor, p2v_idx: Tensor, rays_d: Tensor, 
        mask_pts: Tensor, dists: Optional[Tensor]=None):
        n_pts = mask_pts.sum()
        if n_pts == 0:
            return None, 0

        rays_d = rays_d[...,None,:].expand_as(pts)[mask_pts] # [n_pts, 3]
        pts, p2v_idx = pts[mask_pts], p2v_idx[mask_pts]
        
        vox_embeds = self.voxel_embedder(pts, p2v_idx, self.vox_rep)
        nn_out = self.model(
            self.pts_embedder(vox_embeds), self.view_embedder(rays_d)
        )

        # scatter back...
        out = {'rgb': masked_scatter(mask_pts, nn_out['rgb'])}
        if dists is not None:
            dists = dists[mask_pts]
            free_energy = F.softplus(nn_out['sigma']) * dists[...,None] # activation is here!
            out['free_energy'] = masked_scatter(mask_pts, free_energy)
        else:
            out['sigma'] = masked_scatter(mask_pts, nn_out['sigma'])
        if 'feat_n2' in nn_out:
            out['feat_n2'] = masked_scatter(mask_pts, nn_out['feat_n2'])
        return out, n_pts

    @torch.no_grad()
    def pruning(self, thres=0.5, bits=16):
        '''
        Based on NSVF's pruning function. fairnr/modules/encoder.py#L606
        '''
        center_pts = self.vox_rep.center_points # [n_vox, 3]
        device = center_pts.device
        n_vox, n_pts_vox = center_pts.shape[0], bits**3 # sample bits^3 points per voxel
        rel_pts = offset_points(bits, scale=0, device=device) # [bits**3, 3], scale [0,1]
        p2v_idx = torch.arange(n_vox, device=device) # [n_vox]
        # get pruning scores
        vox_chunk = (self.chunk - 1) // n_pts_vox + 1
        scores = []
        for i in range(0, n_vox, vox_chunk):
            vox_embeds = self.voxel_embedder(rel_pts, p2v_idx[i:i+vox_chunk], 
                                self.vox_rep, per_voxel=True) # [vc, bits**3, emb_dim]
            sigma = self.model(self.pts_embedder(vox_embeds))['sigma'][...,0] # [vc, bits**3]
            scores.append(torch.exp(-sigma.relu()).min(-1)[0]) # [vc]
        scores = torch.cat(scores, 0) 
        keep = (1 - scores) > thres
        # scores = torch.rand(n_vox, device=device) # only for test
        # keep = scores > 0.2
        emb_idx = self.vox_rep.pruning(keep)
        done = False
        if emb_idx is not None:
            new_embeddings = self.voxel_embedder.get_weight()[emb_idx]
            self.voxel_embedder.update_embeddings(new_embeddings)
            done = True
            print(f"pruning done. # of voxels before: {keep.size(0)}, after: {keep.sum()} voxels")
        # release gpu mem
        torch.cuda.empty_cache()
        return done

    @torch.no_grad()
    def splitting(self):
        n_voxel_old = self.vox_rep.n_voxels
        embeddings = self.voxel_embedder.get_weight()
        new_embeddings = self.vox_rep.splitting(embeddings)
        done = False
        if new_embeddings is not None:
            self.voxel_embedder.update_embeddings(new_embeddings)
            done = True
        print(f"splitting done. # of voxels before: {n_voxel_old}, after: {self.vox_rep.n_voxels} voxels")
        # release gpu mem
        torch.cuda.empty_cache()
        return done

    @torch.no_grad()
    def half_stepsize(self):
        old_stepsize = self.point_sampler.step_size
        self.point_sampler.half_stepsize()
        print(f'stepsize: {old_stepsize} -> {self.point_sampler.step_size}')
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_mesh(self,pt_per_edge=3,threshold=0,export_mesh=False, location=None):
        sparse_pts_all,p2v_idx_all,sh, corner = self.vox_rep.get_corner(pt_per_edge)

        valid_mask = p2v_idx_all.ne(-1)
        sparse_pts = sparse_pts_all[valid_mask]
        p2v_idx = p2v_idx_all[valid_mask]

        # sparse_pts,p2v_idx,occ= self.vox_rep.get_pts(pt_per_edge) # N_voxels, pt_per_vox, 3 / N_voxels
        # sh = list(occ.shape)
        
        if p2v_idx.shape[0] > self.ext_chunk:
            ret = [self.prd_sigma(sparse_pts[ci:ci+self.ext_chunk].to(self.vox_rep.device), \
            p2v_idx[ci:ci+self.ext_chunk].to(self.vox_rep.device)) \
            for ci in range(0, p2v_idx.shape[0], self.ext_chunk)]
            ret=torch.cat(ret,dim=0).cpu() # N_pt, 1
        else:
            ret = self.prd_sigma(sparse_pts, p2v_idx).cpu()

        # threshold
        threshold=ret.mean().item() if threshold < 0 else threshold
        print(f"Using threshold {threshold} to extract edge.")
        ### new method for mc
        # first create a cube same as all_pts
        sigma = torch.zeros_like(p2v_idx_all).float()
        sigma[valid_mask]=ret.squeeze()
        sigma = sigma.reshape(sh.tolist())
        vertices, triangles, normals, _ = measure.marching_cubes_lewiner(sigma.numpy(), threshold)
        # move the verts to location
        vertices = vertices*self.vox_rep.voxel_size/pt_per_edge+corner.unsqueeze(0).numpy()
        print('extract done with v: {}, f: {}'.format(vertices.shape, triangles.shape))
        if export_mesh:
            assert location is not None
            import os
            filename = f'thres_{int(threshold)}_'+location.split('/')[-1]
            root_path = location.split('/')[:-1]
            root_path.append(filename)
            filepath =os.path.join(*root_path)
            meshwrite(filepath, vertices,triangles,normals)
            # export_obj(vertices,triangles,filepath)

        ### old version
        # # first, create one new cube
        # sigma = torch.zeros([x*pt_per_edge for x in sh])
        # # naive implementation
        # x,y,z = torch.where(occ>0)
        # x,y,z=x.cpu().numpy(),y.cpu().numpy(),z.cpu().numpy()
        # for i in range(len(x)):
        #     sigma[x[i]*pt_per_edge:(x[i]+1)*pt_per_edge,
        #         y[i]*pt_per_edge:(y[i]+1)*pt_per_edge,
        #         z[i]*pt_per_edge:(z[i]+1)*pt_per_edge] = ret[i].reshape(pt_per_edge,pt_per_edge,pt_per_edge)
        
        # # nx,ny,nz,pt_per_vox -> nx*pt_per_edge,ny*pt_per_edge,nz*pt_per_edge
        # print('fraction occupied', np.mean(sigma.numpy() > threshold))
        # vertices, triangles, normals, _ = measure.marching_cubes_lewiner(sigma.numpy(), threshold)
        # # move the verts to location
        # vertices = vertices*self.vox_rep.voxel_size+self.vox_rep._bds[:3].unsqueeze(0).cpu().numpy()
        # print('extract done with v: {}, f: {}'.format(vertices.shape, triangles.shape))
        # if export_mesh:
        #     assert location is not None
        #     import os
        #     filename = f'thres_{int(threshold)}_'+location.split('/')[-1]
        #     root_path = location.split('/')[:-1]
        #     root_path.append(filename)
        #     filepath =os.path.join(*root_path)
        #     meshwrite(filepath, vertices,triangles,normals)
        #     # export_obj(vertices,triangles,filepath)

    @torch.no_grad()
    def prd_sigma(self,pts,p2v_idx):
        # p2v_idx = p2v_idx.unsqueeze(-1).expand(pts.shape[:2])
        # n_vox,pt_per_vox,_=pts.shape
        # pts=pts.reshape(-1,3) # n_pts, 3
        # p2v_idx=p2v_idx.flatten()
        vox_embeds = self.voxel_embedder(pts, p2v_idx, self.vox_rep)
        rays_d=torch.zeros_like(pts)
        nn_out = self.model(
            self.pts_embedder(vox_embeds), self.view_embedder(rays_d)
        )
        sigma = F.softplus(nn_out['sigma'])
        return sigma

class NerfusionRendererTestSDF(nn.Module):
    def __init__(
        self,
        pts_embedder: Optional[nn.Module],
        view_embedder: Optional[nn.Module],
        voxel_embedder: nn.Module,
        sdf_net: nn.Module,
        rgb_net: nn.Module,
        point_sampler: nn.Module,
        density_fn: nn.Module,
        vox_rep: Explicit3D,
        early_stop_thres: float=0,
        bg_color: Optional[nn.Module]=None,
        white_bkgd: bool=False,
        min_color: int=-1, # note gt_rbg is [0,1] by default. the final out rgb always [0,1] scale
        fwd_steps: int=4,
        chunk: int=1024*16,
        with_eikonal: bool=True,
    ):
        super().__init__()
        self.pts_embedder = pts_embedder if pts_embedder is not None else nn.Identity()
        self.view_embedder = view_embedder if view_embedder is not None else nn.Identity()
        self.voxel_embedder = voxel_embedder
        self.sdf_net = sdf_net
        self.rgb_net = rgb_net
        self.point_sampler = point_sampler
        self.vox_rep = vox_rep
        # self.early_stop = (early_stop_thres > 0) # not used
        # self.early_stop_thres = -np.log(early_stop_thres) if self.early_stop else 0
        self.chunk = chunk
        self.fwd_steps = fwd_steps
        self.bg_color = bg_color
        self.white_bkgd = (white_bkgd and bg_color is None)
        self.min_color = min_color
        self.ext_chunk = self.chunk//16
        self.with_eikonal = with_eikonal
        self.density_fn = density_fn

    def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, 
        t_near: Tensor, t_far: Tensor, ray_hits: Tensor):
        '''
        Args:
            rays_o, rays_d: [N_rays, 3]
            vox_idx, t_near, t_far: [N_rays, max_hits]
            hits: [N_rays]
        '''
        n_rays = ray_hits.shape[0]
        if n_rays > self.chunk:
            ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk], vox_idx[ci:ci+self.chunk], 
                                t_near[ci:ci+self.chunk], t_far[ci:ci+self.chunk], ray_hits[ci:ci+self.chunk]) 
                        for ci in range(0, n_rays, self.chunk)]
            ret = {
                k: torch.cat([out[k] for out in ret if out[k] is not None], 0)
                for k in ret[0]
            }
            return ret
        else:
            if ray_hits.sum() > 0:
                vox_idx, t_near, t_far = vox_idx[ray_hits], t_near[ray_hits], t_far[ray_hits] # [n_hits, max_hits]
                rays_o, rays_d = rays_o[ray_hits], rays_d[ray_hits] # [n_hits, 3]
                # point sampling
                sample_out = self.point_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
                pts, p2v_idx, t_vals, dists = sample_out # [n_hits, max_samples] + [3], [], [], []
                mask_pts = p2v_idx.ne(-1)
                max_samples= mask_pts.shape[1]
                n_pts_fwd, start_step = 0, 0
                nn_out = []
                for i in range(0, max_samples, self.fwd_steps):
                    n_pts_step = mask_pts[:, i:i+self.fwd_steps].sum()
                    if n_pts_step + n_pts_fwd == 0:
                        break # or continue
                    if (i+self.fwd_steps >= max_samples) or (n_pts_fwd + n_pts_step > self.chunk):
                        l, r = start_step, i+self.fwd_steps
                        out, n_pts = self.step_forward(
                            pts[:,l:r], p2v_idx[:,l:r], rays_d, mask_pts[:,l:r], dists[:,l:r])
                        nn_out.append(out)
                        start_step, n_pts_fwd = r, 0
                    else:
                        n_pts_fwd += n_pts_step
                        
                nn_out = {
                    k: torch.cat([out[k] for out in nn_out], 1) 
                    for k in nn_out[0]
                }
                _max_samples = nn_out['rgb'].shape[1] # _max_sample might smaller than max_samples
                # nn_out['rgb'] = nn_out['rgb'].sigmoid() if self.min_color == 0 else (nn_out['rgb'] + 1)/2
                # r_out = volumetric_rendering(nn_out['rgb'], nn_out['free_energy'], t_vals[...,:_max_samples], 
                #             white_bkgd=self.white_bkgd, with_dist=True)
                ret = volumetric_rendering_sdf(nn_out['rgb'], nn_out['free_energy'], t_vals, white_bkgd=False, 
                                    rgb_only=False, with_dist=True, with_T=False)
                ret['sdf'] = nn_out['sdf'][...,0] # B,N,1
            else:
                ret = {'rgb': None, 'disp': None, 'acc': None, "depth": None, "weights": None, "z_vals": None}
            if not self.training:
                ret['rgb'] = fill_in((n_rays, 3), ray_hits, ret['rgb'], 1.0 if self.white_bkgd else 0.0)
                ret['depth'] = fill_in((n_rays, ), ray_hits, ret['depth'], 0.0)
                if ret['disp'] is not None:
                    ret['disp'] = fill_in((n_rays,), ray_hits, ret['disp'], 1000.0)
                ret['acc'] = fill_in((n_rays, ), ray_hits, ret['acc'], 0.0)
                if 'weights' in ret:
                    ret['weights']=fill_in((n_rays, _max_samples), ray_hits, ret['weights'], 0.0)
                ret['sdf']=fill_in((n_rays, _max_samples), ray_hits, ret['sdf'], 1.0)
            return ret

    def step_forward(self, pts: Tensor, p2v_idx: Tensor, rays_d: Tensor, 
        mask_pts: Tensor, z_vals: Optional[Tensor]=None):
        n_pts = mask_pts.sum()
        if n_pts == 0:
            return None, 0

        rays_d = rays_d[...,None,:].expand_as(pts)[mask_pts] # [n_pts, 3]
        pts, p2v_idx = pts[mask_pts], p2v_idx[mask_pts]
        
        vox_embeds = self.voxel_embedder(pts, p2v_idx, self.vox_rep) # [n_pts, embed_dim+3]
        ### pred sdf
        sdf, feats = self.sdf_net(self.pts_embedder(vox_embeds))
        # gradients = torch.autograd.grad(sdf, vox_embeds, torch.ones_like(sdf), 
        #                     retain_graph=self.training, create_graph=self.training)[0]
        rgb = self.rgb_net(self.view_embedder(rays_d), feats)
        density = self.density_fn(sdf)

        # scatter back...
        out = {'rgb': masked_scatter(mask_pts, rgb)}
        dists = z_vals[mask_pts]
        free_energy = density * dists[...,None] # activation is here!
        out['free_energy'] = masked_scatter(mask_pts, free_energy)
        out['sdf'] = masked_scatter(mask_pts, sdf)
        return out, n_pts


# class NerfusionRenderer(nn.Module):
#     def __init__(
#         self,
#         pts_embedder: Optional[nn.Module],
#         view_embedder: Optional[nn.Module],
#         voxel_embedder: nn.Module,
#         model: nn.Module,
#         point_sampler: nn.Module,
#         vox_rep: Explicit3D,
#         early_stop_thres: float=0,
#         bg_color: Optional[nn.Module]=None,
#         white_bkgd: bool=False,
#         min_color: int=-1, # note gt_rbg is [0,1] by default. the final out rgb always [0,1] scale
#         fwd_steps: int=4,
#         chunk: int=1024*64
#     ):
#         '''
#         Args:
#             early_stop_thres: values in the form of $T$, where $T = \exp(-\sum_i\sigma_i\delta_i)$
#         '''
#         super().__init__()
#         self.pts_embedder = pts_embedder if pts_embedder is not None else nn.Identity()
#         self.view_embedder = view_embedder if view_embedder is not None else nn.Identity()
#         self.voxel_embedder = voxel_embedder
#         self.model = model
#         self.point_sampler = point_sampler
#         self.vox_rep = vox_rep
#         self.early_stop = (early_stop_thres > 0)
#         self.early_stop_thres = -np.log(early_stop_thres) if self.early_stop else 0
#         self.chunk = chunk
#         self.fwd_steps = fwd_steps
#         self.bg_color = bg_color
#         self.white_bkgd = (white_bkgd and bg_color is None)
#         self.min_color = min_color


#     def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, 
#         t_near: Tensor, t_far: Tensor, ray_hits: Tensor):
#         '''
#         Args:
#             rays_o, rays_d: [N_rays, 3]
#             vox_idx, t_near, t_far: [N_rays, max_hits]
#             hits: [N_rays]
#         '''
#         n_rays = ray_hits.shape[0]
#         if n_rays > self.chunk:
#             ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk], vox_idx[ci:ci+self.chunk], 
#                                 t_near[ci:ci+self.chunk], t_far[ci:ci+self.chunk], ray_hits[ci:ci+self.chunk]) 
#                         for ci in range(0, n_rays, self.chunk)]
#             ret = {
#                 k: torch.cat([out[k] for out in ret if out[k] is not None], 0)
#                 for k in ret[0]
#             }
#             return ret
#         else:
#             if ray_hits.sum() > 0:
#                 vox_idx, t_near, t_far = vox_idx[ray_hits], t_near[ray_hits], t_far[ray_hits] # [n_hits, max_hits]
#                 rays_o, rays_d = rays_o[ray_hits], rays_d[ray_hits] # [n_hits, 3]
#                 # point sampling
#                 sample_out = self.point_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
#                 pts, p2v_idx, t_vals, dists = sample_out # [n_hits, max_samples] + [3], [], [], []
#                 mask_pts = p2v_idx.ne(-1)
#                 max_samples= mask_pts.shape[1]
#                 n_pts_fwd, start_step = 0, 0
#                 accum_log_T = 0
#                 nn_out = []
#                 for i in range(0, max_samples, self.fwd_steps):
#                     n_pts_step = mask_pts[:, i:i+self.fwd_steps].sum()
#                     if n_pts_step + n_pts_fwd == 0:
#                         break # or continue
#                     if (i+self.fwd_steps >= max_samples) or (n_pts_fwd + n_pts_step > self.chunk):
#                         l, r = start_step, i+self.fwd_steps
#                         out, n_pts = self.step_forward(
#                             pts[:,l:r], p2v_idx[:,l:r], rays_d, mask_pts[:,l:r], dists[:,l:r])
#                         nn_out.append(out)
#                         start_step, n_pts_fwd = r, 0
#                         # oops, early_stop relies on free_energy (sigma * dist)
#                         if self.early_stop:
#                             with torch.no_grad():
#                                 accum_log_T = accum_log_T + out['free_energy'].sum(1)
#                                 mask_early_stop = accum_log_T > self.early_stop_thres
#                                 mask_pts[mask_early_stop] = 0
#                     else:
#                         n_pts_fwd += n_pts_step
                        
#                 nn_out = {
#                     k: torch.cat([out[k] for out in nn_out], 1) 
#                     for k in nn_out[0]
#                 }
#                 _max_samples = nn_out['rgb'].shape[1] # _max_sample might smaller than max_samples
#                 nn_out['rgb'] = nn_out['rgb'].sigmoid() if self.min_color == 0 else (nn_out['rgb'] + 1)/2
                
#                 r_out = volumetric_rendering(nn_out['rgb'], nn_out['free_energy'], t_vals[...,:_max_samples], 
#                             white_bkgd=self.white_bkgd, with_dist=True)   
#                 ret = {'rgb': r_out['rgb'], 'disp': r_out['disp'], 'acc': r_out['acc']}
#                 if 'feat_n2' in nn_out:
#                     ret['regz_term'] = (nn_out['feat_n2'] * r_out['weights']).sum(-1)
#             else:
#                 ret = {'rgb': None, 'disp': None, 'acc': None}
#             # fill_in un-hitted holes
#             ret['rgb'] = fill_in((n_rays, 3), ray_hits, ret['rgb'], 1.0 if self.white_bkgd else 0.0)
#             ret['disp'] = fill_in((n_rays,), ray_hits, ret['disp'], 0.0)
#             ret['acc'] = fill_in((n_rays, ), ray_hits, ret['acc'], 0.0)
#             if self.bg_color is not None:
#                 bg_rgb = self.bg_color(ret['rgb'])
#                 missed = (1 - ret['acc'])
#                 ret['rgb'] = ret['rgb'] + missed[...,None] * bg_rgb
            
#             return ret

#     def step_forward(self, pts: Tensor, p2v_idx: Tensor, rays_d: Tensor, 
#         mask_pts: Tensor, dists: Optional[Tensor]=None):
#         n_pts = mask_pts.sum()
#         if n_pts == 0:
#             return None, 0

#         rays_d = rays_d[...,None,:].expand_as(pts)[mask_pts] # [n_pts, 3]
#         pts, p2v_idx = pts[mask_pts], p2v_idx[mask_pts]
        
#         # TODO change this into one embedder. 
#         vox_embeds = self.voxel_embedder(pts, p2v_idx, self.vox_rep)
#         # TODO:
#         nn_out = self.model(
#             self.pts_embedder(vox_embeds), self.view_embedder(rays_d)
#         )

#         # scatter back...
#         out = {'rgb': masked_scatter(mask_pts, nn_out['rgb'])}
#         if dists is not None:
#             dists = dists[mask_pts]
#             free_energy = F.relu(nn_out['sigma']) * dists[...,None] # activation is here!
#             out['free_energy'] = masked_scatter(mask_pts, free_energy)
#         else:
#             out['sigma'] = masked_scatter(mask_pts, nn_out['sigma'])
#         if 'feat_n2' in nn_out:
#             out['feat_n2'] = masked_scatter(mask_pts, nn_out['feat_n2'])
#         return out, n_pts

#     @torch.no_grad()
#     def pruning(self, thres=0.5, bits=16):
#         '''
#         Based on NSVF's pruning function. fairnr/modules/encoder.py#L606
#         '''
#         center_pts = self.vox_rep.center_points # [n_vox, 3]
#         device = center_pts.device
#         n_vox, n_pts_vox = center_pts.shape[0], bits**3 # sample bits^3 points per voxel
#         rel_pts = offset_points(bits, scale=0, device=device) # [bits**3, 3], scale [0,1]
#         p2v_idx = torch.arange(n_vox, device=device) # [n_vox]
#         # get pruning scores
#         vox_chunk = (self.chunk - 1) // n_pts_vox + 1
#         scores = []
#         for i in range(0, n_vox, vox_chunk):
#             vox_embeds = self.voxel_embedder(rel_pts, p2v_idx[i:i+vox_chunk], 
#                                 self.vox_rep, per_voxel=True) # [vc, bits**3, emb_dim]
#             sigma = self.model(self.pts_embedder(vox_embeds))['sigma'][...,0] # [vc, bits**3]
#             scores.append(torch.exp(-sigma.relu()).min(-1)[0]) # [vc]
#         scores = torch.cat(scores, 0) 
#         keep = (1 - scores) > thres
#         # scores = torch.rand(n_vox, device=device) # only for test
#         # keep = scores > 0.2
#         emb_idx = self.vox_rep.pruning(keep)
#         done = False
#         if emb_idx is not None:
#             new_embeddings = self.voxel_embedder.get_weight()[emb_idx]
#             self.voxel_embedder.update_embeddings(new_embeddings)
#             done = True
#             print(f"pruning done. # of voxels before: {keep.size(0)}, after: {keep.sum()} voxels")
#         # release gpu mem
#         torch.cuda.empty_cache()
#         return done

#     @torch.no_grad()
#     def splitting(self):
#         n_voxel_old = self.vox_rep.n_voxels
#         embeddings = self.voxel_embedder.get_weight()
#         new_embeddings = self.vox_rep.splitting(embeddings)
#         done = False
#         if new_embeddings is not None:
#             self.voxel_embedder.update_embeddings(new_embeddings)
#             done = True
#         print(f"splitting done. # of voxels before: {n_voxel_old}, after: {self.vox_rep.n_voxels} voxels")
#         # release gpu mem
#         torch.cuda.empty_cache()
#         return done

#     @torch.no_grad()
#     def half_stepsize(self):
#         old_stepsize = self.point_sampler.step_size
#         self.point_sampler.half_stepsize()
#         print(f'stepsize: {old_stepsize} -> {self.point_sampler.step_size}')
#         torch.cuda.empty_cache()