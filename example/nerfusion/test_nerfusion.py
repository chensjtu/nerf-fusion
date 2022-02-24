import os
import sys
from unicodedata import name

from cv2 import threshold

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from accelRF.datasets import Room_SCANNET
from accelRF.metrics.nerfusion_loss import nerfusion_Loss
from accelRF.models import (NSVF_MLP, BackgroundField, PositionalEncoding,
                            VoxelEncoding)
from accelRF.pointsampler import NSVFPointSampler
from accelRF.raysampler import (TestFusionViewRaySampler, VisRaySampler,
                                VoxIntersectRaySampler)
from accelRF.render.nerfusion_test import NerfusionRendererTest
from accelRF.rep.voxel_grid import TSDF_Fusion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from opts import config_parser

parser = config_parser()
args = parser.parse_args()

n_gpus = torch.cuda.device_count()
n_replica = 1
device = 'cuda'
cudnn.benchmark = True
savedir = os.path.join(args.basedir, args.expname)

if args.local_rank >= 0:
    torch.cuda.set_device(args.local_rank)
    device = f'cuda:{args.local_rank}'
    dist.init_process_group(backend='nccl', init_method="env://")
    n_replica = n_gpus

if args.local_rank <= 0:
    tb_writer = SummaryWriter(os.path.join(savedir, 'summaries'))

# parameters
if args.pruning_every_steps>0:
    args.pruning_steps = list(range(0, args.N_iters, args.pruning_every_steps))[1:]
    
args.loss_weights = {
    'rgb': args.loss_w_rgb, 
    'alpha': args.loss_w_alpha, 
    'reg_term': args.loss_w_reg,
    'depth': args.loss_w_depth,
    'region':args.loss_w_region,
    'un_region': args.loss_w_unregion
    }
print(f"the weights of loss are:{args.loss_weights}")

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))
to01dp = lambda x: (x/x.max()).float()
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def main():
    # prepare data
    if args.dataset_type == 'test_scannet':
        dataset = Room_SCANNET(args.datadir, args.scene, args.half_res, 
                                args.testskip, args.white_bkgd, with_bbox=True)
    else:
        print(f'{args.dataset_type} has not been supported yet...')
        exit()
    
    # init vox
    vox_grid = TSDF_Fusion(dataset.get_bds(), args.voxel_size, device=device,
        use_metric_depth=not args.no_use_metric_depth,margin=3) # margin can control the voxel size
 
    # integrate tsdf
    scene_data = dataset.get_render_set()
    if not args.no_use_metric_depth:
        metric_depth=[]
    else:
        metric_depth = None

    for i in range(scene_data.data['gt_img'].shape[0]):
        depth_im = scene_data.data['depths'][i]
        cam_intr = scene_data.data['intrinsics'][i]
        cam_pose = scene_data.data['poses'][i]
        
        metric_dp = vox_grid.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)
        if metric_dp is not None:
            metric_depth.append(metric_dp)
    if metric_depth is not None:
        metric_depth = torch.stack(metric_depth)
        dataset.data['depths']=metric_depth.to(depth_im.device)
    vox_grid.cal_corner(use_corner=True)
    # poses = dataset.data['poses'].cpu().numpy()
    # for i in range(len(poses)):
    #     np.savetxt(f"./data/pose{i}.txt",poses[i])

    # create model
    input_ch, input_ch_views = (2*args.multires+1)*args.embed_dim, (2*args.multires_views)*3 # view not include x
    nsvf_render = NerfusionRendererTest(
        point_sampler=NSVFPointSampler(args.step_size),
        pts_embedder=PositionalEncoding(args.multires, pi_bands=True),
        view_embedder=PositionalEncoding(args.multires_views, angular_enc=True, include_input=False),
        voxel_embedder=VoxelEncoding(vox_grid.n_corners, args.embed_dim),
        model=NSVF_MLP(in_ch_pts=input_ch, in_ch_dir=input_ch_views, D_rgb=args.D_rgb, W_rgb=args.W_rgb, 
            D_feat=args.D_feat, W_feat=args.W_feat, D_sigma=args.D_sigma, W_sigma=args.W_sigma, 
            layernorm=(not args.no_layernorm), with_reg_term=(args.loss_w_reg!=0),skips_feat=[]), # NOTE: skip is not used! 
        vox_rep=vox_grid,
        bg_color=BackgroundField(bg_color=1. if args.white_bkgd else 0., trainable=False) if args.bg_field else None,
        white_bkgd=args.white_bkgd,
        min_color=args.min_color,
        early_stop_thres=args.early_stop_thres,
        chunk=args.chunk * n_gpus if args.local_rank < 0 else args.chunk,
    ).to(device)
        
    # create optimizer
    optimizer  = optim.Adam(nsvf_render.parameters(), args.lrate, betas=(0.9, 0.999))
    start = 0

    # init loss_fn
    loss_fn = nerfusion_Loss(args.loss_weights, args.region_epsilon, args.epsilon_steps, args.i_decay_epsilon)

    # load checkpoint
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(savedir, f) for f in sorted(os.listdir(savedir)) if f.endswith('.pt')]
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Load from: ', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        start = ckpt['global_step']
        loss_fn.load_epsilon(start, args.i_decay_epsilon)
        n_voxels, n_corners, grid_shape = ckpt['n_voxels'], ckpt['n_corners'], ckpt['grid_shape']
        if n_voxels != vox_grid.n_voxels:
            vox_grid.load_adjustment(n_voxels, grid_shape)
            nsvf_render.voxel_embedder.load_adjustment(n_corners)
            optimizer.param_groups.clear(); optimizer.state.clear()
            optimizer.add_param_group({'params':nsvf_render.parameters()})
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer.param_groups[0]['initial_lr'] = args.lrate
        except:
            pass
        nsvf_render.load_state_dict(ckpt['state_dict'])

    ### detach MLP
    if args.detach_mlp:
        nsvf_render.model.detach_model()
    # for name, weight in nsvf_render.named_parameters():
    #     # print("weight:", weight)
    #     if weight.requires_grad:
    #         print(name)
    # exit()

    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=start-1)
    
    # parallelization
    if args.local_rank >=0:
        nsvf_render_ = torch.nn.parallel.DistributedDataParallel(nsvf_render, device_ids=[args.local_rank])
    else:
        nsvf_render_ = nn.DataParallel(nsvf_render)
    # nsvf_render_ = nsvf_render

    # prepare dataloader
    train_base_raysampler = \
        TestFusionViewRaySampler(dataset.get_render_set(), args.N_rand, args.N_iters, args.N_views, 
            precrop=False, full_rays=args.full_rays, start_epoch=start,) # rank=args.local_rank, n_replica=n_replica)
    train_vox_raysampler = VoxIntersectRaySampler(args.N_rand, train_base_raysampler, vox_grid, device=device)
    # test_raysampler = VoxIntersectRaySampler(0, RenderingRaySampler(dataset.get_sub_set('test')), 
    #                         vox_grid, mask_sample=False, device=device, num_workers=1)
    val_raysampler = VoxIntersectRaySampler(0, VisRaySampler(dataset.get_render_set()), 
                            vox_grid, mask_sample=False, device=device, num_workers=0)

    train_rayloader = DataLoader(train_vox_raysampler, num_workers=0) # vox_rays  ampler's N_workers==0, pin_mem==False
    # test_rayloader = DataLoader(test_raysampler, num_workers=0)
    train_ray_iter = iter(train_rayloader)

    # start training
    train_iter = start # ref train_iter to aviod not defined.
    for train_iter in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb, gt_dp = ray_batch['hits'][0], ray_batch['gt_rgb'][0],ray_batch['gt_dp'][0].squeeze()

        render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits)
        loss, sub_losses = loss_fn.get_loss(render_out, gt_rgb, gt_depth=gt_dp)
        # print("vis the weight distribution of weight.")
        optimizer.zero_grad()
        if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()

        # for name, weight in nsvf_render_.named_parameters():
        #     # print("weight:", weight) # print weight
        #     if weight.requires_grad:
        #         if weight.grad is not None:
        #             # print("weight:", weight.grad)
        #             print(f"{name} weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())

        lr_sched.step()
        loss_fn.update_epsilon(train_iter+1, args.i_decay_epsilon)

        if train_iter%args.i_print==0:
            if "rgb" in sub_losses.keys():
                psnr = mse2psnr(sub_losses['rgb'])
            else:
                psnr = torch.tensor(0.0)
            tqdm.write(f"[TRAIN] train_iter: {train_iter} Loss: {loss.item()} PSNR: {psnr.item()} " + \
                    f"Hit ratio: {hits.sum().item()/hits.shape[0]} " + \
                    f"Bg ratio: {(gt_rgb.sum(-1).eq(0)).sum().item()/hits.shape[0]} " + \
                    f"cur_epsilon: {loss_fn.get_cur_epsilon()}")
            if args.local_rank <= 0:
                tb_writer.add_scalar('total_loss', loss, train_iter)
                if "rgb" in sub_losses.keys():
                    tb_writer.add_scalar('losses/rgb_loss', sub_losses['rgb'], train_iter)
                    tb_writer.add_scalar('psnr', psnr, train_iter)
                if 'alpha' in sub_losses.keys():
                    tb_writer.add_scalar('losses/alpha_loss', sub_losses['alpha'], train_iter)
                if 'depth' in sub_losses.keys():
                    tb_writer.add_scalar('losses/depth_loss', sub_losses['depth'], train_iter)
                if 'region' in sub_losses.keys():
                    tb_writer.add_scalar('losses/region_loss', sub_losses['region'], train_iter)
                if 'unregion' in sub_losses.keys():
                    tb_writer.add_scalar('losses/unregion_loss', sub_losses['unregion'], train_iter)
                if 'reg_term' in sub_losses.keys():
                    tb_writer.add_scalar('losses/reg_term_loss', sub_losses['reg_term'], train_iter)


        # if i%args.i_testset == 0 and i > 0 and args.local_rank <= 0:
        #     eval(nsvf_render_, test_rayloader, dataset.get_hwf()[:2], os.path.join(savedir, f'testset_{i:06d}'))
        
        if (train_iter+1)%args.i_weights==0 and args.local_rank <= 0:
            path = os.path.join(savedir, f'{train_iter+1:06d}.pt')
            torch.save({
                'global_step': train_iter+1,
                'state_dict': nsvf_render.state_dict(), # keys start with 'module'
                'optimizer_state_dict': optimizer.state_dict(),
                'n_voxels': vox_grid.n_voxels, 'n_corners': vox_grid.n_corners,
                'grid_shape': vox_grid.grid_shape.tolist()
            }, path)

        ### used for vis the pred depth and gt_dpeth
        if train_iter%100==0 and args.local_rank <= 0:
            vis_depth(nsvf_render_, val_raysampler,tb_writer, train_iter)
    
    ### extract mesh.
    if args.export_mesh:
        nsvf_render_.module.extract_mesh(pt_per_edge=args.pt_per_edge,
            export_mesh=args.export_mesh, threshold=args.threshold,
            location=os.path.join(savedir, f'pt_{args.pt_per_edge}_{train_iter+1:06d}.obj'))

    ### vis
    for img_i in trange(len(val_raysampler)):
        ray_batch = val_raysampler[img_i]
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
        vox_idx, t_near, t_far = ray_batch['vox_idx'], ray_batch['t_near'], ray_batch['t_far']
        hits, gt_rgb = ray_batch['hits'], ray_batch['gt_rgb']
        with torch.no_grad():
            nsvf_render_.eval()
            render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits)
            nsvf_render_.train()
        if "rgb" in render_out.keys():
            psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
            tb_writer.add_scalar('psnr_eval', psnr, img_i)
            H, W, _ = dataset.get_hwf()
            tb_writer.add_image('color/gt_rgb', gt_rgb.reshape(H,W,-1), img_i, dataformats="HWC")
            tb_writer.add_image('color/rgb', to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1), img_i, dataformats='HWC')
        tb_writer.add_image('geo/acc', render_out['acc'].reshape(H,W), img_i, dataformats="HW")
        tb_writer.add_image('geo/depth', to01dp(render_out['depth']).reshape(H,W), img_i, dataformats="HW")
        tb_writer.add_image('geo/disp', to01dp(render_out['disp']).reshape(H,W), img_i, dataformats="HW")
        tb_writer.add_image('geo/gt_depth', to01dp(ray_batch['gt_dp']).reshape(H,W), img_i, dataformats="HW")


def eval(nsvf_render, rayloader, img_hw, testsavedir):
    os.makedirs(testsavedir, exist_ok=True) 
    nsvf_render.eval()
    for i, ray_batch in enumerate(tqdm(rayloader)):
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]
        with torch.no_grad():
            render_out = nsvf_render(rays_o, rays_d, vox_idx, t_near, t_far, hits)
        psnr = mse2psnr(img2mse(gt_rgb, render_out['rgb']))
        imageio.imwrite(os.path.join(testsavedir, f'{i:03d}.png'), 
                    to8b(render_out['rgb'].cpu().numpy()).reshape(*img_hw,-1))
        tqdm.write(f"[Test] #: {i} PSNR: {psnr.item()}")
    nsvf_render.train()

def vis_depth(nsvf_render, rayloader,summary_writer, train_iter):
    '''
        use valid pts to check depth prediction.
    '''
    nsvf_render.eval()
    ray_batch = rayloader.get_rays_for_test(0)
    rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
    vox_idx, t_near, t_far = ray_batch['vox_idx'], ray_batch['t_near'], ray_batch['t_far']
    hits, _, gt_dp = ray_batch['hits'], ray_batch['gt_rgb'], ray_batch['gt_dp'].squeeze()
    with torch.no_grad():
        render_out = nsvf_render(rays_o, rays_d, vox_idx, t_near, t_far, hits)
        gt_dp_mask = gt_dp.ne(0.0)
        gt_dp_masked = gt_dp[gt_dp_mask]
        pred_dp_masked = render_out['depth'][gt_dp_mask]
        pred_dp = render_out['depth'].masked_fill(~gt_dp_mask,0.0)
        ### vis the heatmap instead of the 
        summary_writer.add_scalars('depth_vis0', {
            "gt_dp000000":gt_dp_masked[0],
            "gt_dp050000":gt_dp_masked[50000],
            "pred_dp000000":pred_dp_masked[0],
            "pred_dp050000":pred_dp_masked[50000],}, train_iter)
        summary_writer.add_scalars('depth_vis1', {
            "gt_dp100000":gt_dp_masked[100000],
            "gt_dp150000":gt_dp_masked[150000],
            "pred_dp100000":pred_dp_masked[100000],
            "pred_dp150000":pred_dp_masked[150000],}, train_iter)
        summary_writer.add_scalars('depth_vis2', {
            "gt_dp200000":gt_dp_masked[200000],
            "gt_dp250000":gt_dp_masked[250000],
            "pred_dp200000":pred_dp_masked[200000],
            "pred_dp250000":pred_dp_masked[250000]}, train_iter)
        ### here we use 1.27 meter to judge depth error
        compared_map = ((pred_dp-gt_dp).abs()*200.).reshape(480,640).cpu()
        vis_map = cv2.cvtColor(cv2.applyColorMap(compared_map.numpy().astype(np.uint8),
            cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB) # be careful of cv2 special channel rules.
        summary_writer.add_image('eval_depth',vis_map,train_iter,dataformats="HWC")
    nsvf_render.train()




if __name__ == '__main__':
    # print("Args: \n", args, "\n", "-"*40)
    os.makedirs(savedir, exist_ok=True)
    f = os.path.join(savedir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(savedir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    # start main program
    main()
