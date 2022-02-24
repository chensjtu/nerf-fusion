import os
import sys
from distutils.log import debug

from matplotlib.pyplot import axes, axis

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from accelRF.datasets import Blender, Single_SCANNET, fusion
from accelRF.metrics.nsvf_loss import nsvf_loss
from accelRF.models import (NSVF_MLP, BackgroundField, PositionalEncoding,
                            SpatialEncoder)
from accelRF.pointsampler import NSVFPointSampler
from accelRF.raysampler import (FusionRaySampler, PerViewRaySampler,
                                RenderingRaySampler, VoxIntersectRaySampler)
from accelRF.render.nerfusion_render import NerfusionRenderer
from accelRF.rep.voxel_grid import VoxelGridFusion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from opts import config_parser

parser = config_parser()
args = parser.parse_args()
# args.expname = 'debug_nerfusion'
n_gpus = torch.cuda.device_count()
n_replica = 1
device = 'cuda'
cudnn.benchmark = True
savedir = os.path.join(args.basedir, args.expname)

if args.local_rank >= 0:
    dist.init_process_group(backend='nccl', init_method="env://")
    device = f'cuda:{args.local_rank}'
    n_replica = n_gpus

if args.local_rank <= 0:
    tb_writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

# parameters
if args.pruning_every_steps>0:
    args.pruning_steps = list(range(0, args.N_iters, args.pruning_every_steps))[1:]
args.loss_weights = {
    'rgb': args.loss_w_rgb, 'alpha': args.loss_w_alpha, 'reg_term': args.loss_w_reg, "depth": args.loss_w_depth
    }

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to01dp = lambda x: (x/x.max()).float()

def main():
    # prepare data
    if args.dataset_type == 'blender':
        dataset = Blender(args.datadir, args.scene, args.half_res, 
                                args.testskip, args.white_bkgd, with_bbox=True)
    elif args.dataset_type == 'scannet':
        dataset = Single_SCANNET(args.datadir, args.scene, single_fragment=24)
    else:
        print(f'{args.dataset_type} has not been supported yet...')
        exit()

    # init vox_rep
    occ, bds = dataset.get_vox()
    vox_grid = VoxelGridFusion(bds, args.voxel_size, occ=occ, use_corner=True).to(device)

    # prepare train set
    start = 0
    train_base_raysampler=FusionRaySampler(dataset.get_train_set(), args.N_rand, args.N_iters, args.N_views, 
            precrop=False, full_rays=args.full_rays, start_epoch=start, rank=args.local_rank, n_replica=n_replica)
    train_vox_raysampler=VoxIntersectRaySampler(args.N_rand,train_base_raysampler,vox_grid,device=device,num_workers=4)

    # prepare test set
    test_base_raysampler = FusionRaySampler(dataset.get_test_set(), args.N_rand, args.N_iters, args.N_views, 
            precrop=False, full_rays=args.full_rays, start_epoch=start, rank=args.local_rank, n_replica=n_replica)
    test_raysampler = VoxIntersectRaySampler(0, test_base_raysampler, 
                            vox_grid, mask_sample=False, device=device, num_workers=4)
    # val_raysampler = VoxIntersectRaySampler(0, RenderingRaySampler(dataset.get_sub_set('val')), 
    #                         vox_grid, mask_sample=False, device=device, num_workers=0)

    train_rayloader = DataLoader(train_vox_raysampler, num_workers=0) # vox_raysampler's N_workers==0, pin_mem==False
    # test_rayloader = DataLoader(test_raysampler, num_workers=0)
    train_ray_iter = iter(train_rayloader)

    # create model
    input_ch, input_ch_views = (2*args.multires+1)*args.embed_dim, (2*args.multires_views)*3 # view not include x
    nsvf_render = NerfusionRenderer(
        point_sampler=NSVFPointSampler(args.step_size),
        pts_embedder=PositionalEncoding(args.multires, pi_bands=True),
        view_embedder=PositionalEncoding(args.multires_views, angular_enc=True, include_input=False),
        voxel_embedder=SpatialEncoder(backbone="resnet34"),
        model=NSVF_MLP(in_ch_pts=input_ch, in_ch_dir=input_ch_views, D_rgb=args.D_rgb, W_rgb=args.W_rgb, 
            D_feat=args.D_feat, W_feat=args.W_feat, D_sigma=args.D_sigma, W_sigma=args.W_sigma, 
            layernorm=(not args.no_layernorm), with_reg_term=(args.loss_w_reg!=0)),
        vox_rep=vox_grid,
        bg_color=BackgroundField(bg_color=1. if args.white_bkgd else args.min_color, trainable=False) \
                if args.bg_field else None,
        white_bkgd=args.white_bkgd, 
        min_color=args.min_color,
        early_stop_thres=args.early_stop_thres,
        chunk=args.chunk * n_gpus if args.local_rank < 0 else args.chunk,
    ).to(device)
        
    if args.local_rank >=0:
        nsvf_render_ = torch.nn.parallel.DistributedDataParallel(nsvf_render, device_ids=[args.local_rank])
    else:
        # nsvf_render_ = nn.DataParallel(nsvf_render)
        nsvf_render_ = nsvf_render
    # nsvf_render_ = nsvf_render
    # create optimizer
    optimizer  = optim.Adam(nsvf_render_.parameters(), args.lrate, betas=(0.9, 0.999))
    
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
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except:
            pass
        nsvf_render.load_state_dict(ckpt['state_dict'])

    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=start-1)

    # start training
    start = 0
    for i in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)

        imgs = ray_batch['imgs'].squeeze()
        KRcam = ray_batch['KRcam'].squeeze() # V 4 4
        # pix_mask = ray_batch['pix_mask'].squeeze()
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]

        render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits, imgs, KRcam)

        loss, sub_losses = nsvf_loss(render_out, gt_rgb, args.loss_weights,ray_batch['gt_dp'][0].squeeze())
        optimizer.zero_grad()
        if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()

        lr_sched.step()
        if i%args.i_print==0:
            psnr = mse2psnr(sub_losses['rgb'])
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {sub_losses}  PSNR: {psnr.item()} " + \
                    # f"Alpha: {sub_losses['alpha'].item()} " + \
                    f"Hit ratio: {hits.sum().item()/hits.shape[0]} " + \
                    f"Bg ratio: {(gt_rgb.sum(-1).eq(0)).sum().item()/hits.shape[0]}")
            if args.local_rank <= 0:
                tb_writer.add_scalar('loss/rgb', sub_losses['rgb'], i)
                tb_writer.add_scalar('loss/psnr', psnr, i)
                if 'depth' in sub_losses:
                    tb_writer.add_scalar('loss/depth', sub_losses['depth'], i)
                if 'alpha' in sub_losses:
                    tb_writer.add_scalar('loss/alpha', sub_losses['alpha'], i)

                if i%args.i_img==0:
                    # Log a rendered validation view to Tensorboard
                    index = torch.randint(args.N_views, ())
                    ray_batch = test_raysampler[index]
                    rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
                    vox_idx, t_near, t_far = ray_batch['vox_idx'], ray_batch['t_near'], ray_batch['t_far']
                    hits, gt_rgb = ray_batch['hits'], ray_batch['gt_rgb']
                    imgs = ray_batch['imgs'].squeeze() 
                    KRcam = ray_batch['KRcam'].squeeze() # V 4 4
                    with torch.no_grad():
                        nsvf_render_.eval()
                        render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits, imgs, KRcam)
                        nsvf_render_.train()
                        psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
                    tb_writer.add_scalar('psnr_eval', psnr, i)
                    H, W, _ = dataset.get_hwf()
                    tb_writer.add_image('gt_rgb', gt_rgb.reshape(H,W,-1), i, dataformats="HWC")
                    tb_writer.add_image('rgb', to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1), i, dataformats='HWC')
                    tb_writer.add_image('disp', to01dp(render_out['disp']).reshape(H,W), i, dataformats="HW")
                    tb_writer.add_image('acc', render_out['acc'].reshape(H,W), i, dataformats="HW")
                    tb_writer.add_image('depth', to01dp(render_out['depth']).reshape(H,W), i, dataformats="HW")
        # if i%args.i_testset == 0 and i > 0 and args.local_rank <= 0:
        #     eval(nsvf_render_, test_rayloader, dataset.get_hwf()[:2], os.path.join(savedir, f'testset_{i:06d}'))

    # --------for NVS---------
    Psnr = []
    for i in range(len(dataset)):
        ray_batch_train = next(train_ray_iter)
        ray_batch = test_raysampler[i]
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
        vox_idx, t_near, t_far = ray_batch['vox_idx'], ray_batch['t_near'], ray_batch['t_far']
        hits, gt_rgb = ray_batch['hits'], ray_batch['gt_rgb']
        imgs = ray_batch_train['imgs'].squeeze() 
        KRcam = ray_batch_train['KRcam'].squeeze() # V 4 4
        with torch.no_grad():
            nsvf_render_.eval()
            render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits, imgs, KRcam)
            # nsvf_render_.train()
            H, W, _ = dataset.get_hwf()
            rgb = to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1)
            gt = to8b(gt_rgb.cpu().numpy()).reshape(H,W,-1)
            imageio.imsave(f"/home/yangchen/projects/Accel-RF/logs/depth_nerf/images/{i}.jpg", 
                np.concatenate((rgb,gt),axis=1))
            psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
            Psnr.append(psnr)
    print(torch.stack(Psnr).mean())
    
        # if (i+1)%args.i_weights==0 and args.local_rank <= 0:
        #     path = os.path.join(savedir, f'{i+1:06d}.pt')
        #     torch.save({
        #         'global_step': i+1,
        #         'state_dict': nsvf_render.state_dict(), # keys start with 'module'
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, path)


def eval(nsvf_render, rayloader, img_hw, testsavedir):
    os.makedirs(testsavedir, exist_ok=True) 
    nsvf_render.eval()
    for i, ray_batch in enumerate(tqdm(rayloader)):
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]
        with torch.no_grad():
            render_out = nsvf_render(rays_o, rays_d)
        psnr = mse2psnr(img2mse(gt_rgb, render_out['rgb']))
        imageio.imwrite(os.path.join(testsavedir, f'{i:03d}.png'), 
                    to8b(render_out['rgb'].cpu().numpy()).reshape(*img_hw,-1))
        tqdm.write(f"[Test] #: {i} PSNR: {psnr.item()}")
    nsvf_render.train()

if __name__ == '__main__':
    print("Args: \n", args, "\n", "-"*40)
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

# # model refinement
# reset_module = False
# if i in args.pruning_steps:
#     done = nsvf_render.pruning(thres=args.pruning_thres)
#     reset_module = done or reset_module
# if i in args.splitting_steps:
#     done = nsvf_render.splitting()
#     reset_module = done or reset_module
# if i in args.half_stepsize_steps:
#     nsvf_render.half_stepsize()
# if reset_module:
#     if args.local_rank >= 0:
#         del nsvf_render_
#         nsvf_render_ = nn.parallel.DistributedDataParallel(...)
#     optimizer.zero_grad()
#     # https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814/8
#     optimizer.param_groups.clear() # optimizer.param_group = []
#     optimizer.state.clear() # optimizer.state = defaultdict(dict)
#     optimizer.add_param_group({'params':nsvf_render_.parameters()}) # necessary!
