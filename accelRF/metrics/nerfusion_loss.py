from typing import Dict, Tuple, Optional
from matplotlib.pyplot import step
import torch
from torch import Tensor
import torch.nn.functional as F

INF = 1e3

# @torch.jit.script
class nerfusion_Loss(object):
    def __init__(self, loss_weights, args, epsilon=0.02, steps=100, i_decay_epsilon=0) -> None:
        self.loss_w = {}
        if 'rgb' in loss_weights and loss_weights['rgb'] > 0:
            self.loss_w['rgb'] = loss_weights['rgb']
        
        if 'alpha' in loss_weights and loss_weights['alpha'] > 0:
            self.loss_w['alpha'] = loss_weights['alpha']
        
        if 'reg_term' in loss_weights and loss_weights['reg_term'] > 0:
            self.loss_w['reg_term'] = loss_weights['reg_term']

        if 'depth' in loss_weights and loss_weights['depth'] > 0:
            self.loss_w['depth'] = loss_weights['depth']
            
        if 'region' in loss_weights and loss_weights['region'] > 0:
            self.loss_w['region'] = loss_weights['region']
            self.init_epsilon = 2
            self.final_epsilon = epsilon
            self.steps = steps
            if steps > 0:
                self.interval = (self.init_epsilon - self.final_epsilon)/steps
            else:
                print("the epsilon will not change during the training period!")
                self.interval = 0.0
            self.cur_epsilon = self.final_epsilon+steps*self.interval
            self.gaussian = torch.distributions.normal.Normal(torch.tensor(0.), torch.tensor(self.cur_epsilon/3))
            print(f"enable region loss! the region loss will fall to {epsilon} in {i_decay_epsilon*steps} epoches")
        else:
            self.cur_epsilon = -1
            self.steps = 1
            self.final_epsilon = -1
            
        if 'un_region' in loss_weights and loss_weights['un_region'] > 0:
            self.loss_w['unregion'] = loss_weights['un_region']

        if 'sdf' in loss_weights and loss_weights['sdf']>0:
            self.loss_w['sdf'] = loss_weights['sdf']
            self.truncation = args.truncation

        if 'empty' in loss_weights and loss_weights['empty']>0:
            self.loss_w['empty'] = loss_weights['empty']
    
        if 'eikonal' in loss_weights and loss_weights['eikonal']>0:
            self.loss_w['eikonal'] = loss_weights['eikonal']

    @staticmethod
    def compute_loss(prediction, target, type="l2"):
        if type == 'l2':
            return F.mse_loss(prediction, target)
        elif type == 'l1':
            return F.l1_loss(prediction, target)

    def get_loss(self, r_out, gt_rgb, gt_depth=None, loss_type="l2"):
        losses = {}
        if 'rgb' in self.loss_w.keys():
            pred = r_out['rgb'] 
            tgt = gt_rgb
            losses['rgb'] = self.compute_loss(pred, tgt) # 102
        
        if 'alpha' in self.loss_w.keys():
            _alpha = r_out['weights'].sum(-1).reshape(-1)
            pred = _alpha
            tgt = torch.ones_like(_alpha)
            losses['alpha'] = self.compute_loss(pred, tgt)

        if 'reg_term' in self.loss_w.keys():
            losses['reg_term'] = r_out['regz_term'].mean()

        if 'depth' in self.loss_w.keys() and gt_depth is not None: 
            dp_mask = gt_depth.ne(0)
            pred = r_out['depth'][dp_mask]
            tgt = gt_depth[dp_mask]
            losses['depth'] = self.compute_loss(pred, tgt)

        if 'region' in self.loss_w.keys() and gt_depth is not None:
            dp_mask = gt_depth.ne(0) 
            t_vals = r_out['t_vals'][dp_mask]
            weights = r_out['weights'][dp_mask]
            dists = r_out['dists'][dp_mask]
            gt_depth = gt_depth[dp_mask]
            ### verify the the data
            # near = t_vals[:,0]
            # far = t_vals.max(dim=-1)[0]
            # tmp_dp = gt_depth[dp_mask]
            # print("inside ratio is:",((tmp_dp>near)&(far>tmp_dp)).sum()/dp_mask.sum())
            t_vals_empty = t_vals.ne(0.0) # n_rays, n_pts
            region_mask = (t_vals>(gt_depth-self.cur_epsilon).unsqueeze(-1).expand_as(t_vals)) & \
                        (t_vals<(gt_depth+self.cur_epsilon).unsqueeze(-1).expand_as(t_vals)) & \
                        t_vals_empty
            # region_mask_num = region_mask.sum(-1)
            expected_w = self.gaussian.log_prob(t_vals-gt_depth.unsqueeze(-1).expand_as(t_vals)).exp()
            dists.masked_fill_(~region_mask, 0.0)
            expected_w = expected_w*dists

            losses['region']=F.mse_loss(weights[region_mask],expected_w[region_mask])
            losses['unregion']=F.mse_loss(weights[~region_mask],torch.zeros_like(weights[~region_mask]))

        if 'sdf' in self.loss_w.keys()and gt_depth is not None:
            front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(r_out['z_vals'], gt_depth, self.truncation)
            losses['empty']=self.compute_loss(r_out['sdf'] * front_mask, torch.ones_like(r_out['sdf']) * front_mask, loss_type) * fs_weight
            losses['sdf']=self.compute_loss((r_out['z_vals'] + r_out['sdf'] * self.truncation)[sdf_mask], gt_depth[:,None].expand_as(sdf_mask)[sdf_mask], loss_type) * sdf_weight

        if 'eikonal' in self.loss_w.keys():
            pass

        loss = sum(losses[key] * self.loss_w[key] for key in losses)
        return loss, losses

    @staticmethod
    def get_masks(z_vals, target_d, truncation):

        front_mask = z_vals < (target_d[:,None] - truncation)
        back_mask = z_vals > (target_d[:,None] + truncation)
        depth_mask = target_d.ne(0.0)
        sdf_mask = (~front_mask) & (~back_mask) & depth_mask[:,None].expand_as(front_mask)

        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_sdf_samples + num_fs_samples # 164013
        fs_weight = 1.0 - num_fs_samples / num_samples # 0.036
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, sdf_mask, fs_weight, sdf_weight

    def update_epsilon(self, iteration, i_decay_epsilon):
        '''
            use linear method to improve performance
        '''
        if self.final_epsilon > 0:
            if iteration % i_decay_epsilon==0:
                cur_step = iteration//i_decay_epsilon
                if cur_step <= self.steps:
                    self.cur_epsilon = self.final_epsilon + (self.steps-cur_step)*self.interval
                    self.gaussian = torch.distributions.normal.Normal(torch.tensor(0.), torch.tensor(self.cur_epsilon/3))

    def get_cur_epsilon(self):
        return self.cur_epsilon

    def load_epsilon(self, iteration, i_decay_epsilon):
        if self.final_epsilon > 0:
            cur_step = iteration//i_decay_epsilon
            if cur_step <= self.steps:
                self.cur_epsilon = self.final_epsilon + (self.steps-cur_step)*self.interval
            else:
                self.cur_epsilon = self.final_epsilon
            self.gaussian = torch.distributions.normal.Normal(torch.tensor(0.), torch.tensor(self.cur_epsilon/3))


# def compute_loss(prediction, target, loss_type='l2'):
#     if loss_type == 'l2':
#         return F.mse_loss(prediction, target)
#     elif loss_type == 'l1':
#         return F.l1_loss(prediction, target)

#     raise Exception('Unsupported loss type')


# def get_masks(z_vals, target_d, truncation):

#     front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)) # 1024， 336
#     back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals)) # 1024， 336
#     depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d)) # 1024， 1
#     sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask # 1024， 336

#     num_fs_samples = torch.math.count_nonzero(front_mask, dtype=torch.float32) # 158070
#     num_sdf_samples = torch.math.count_nonzero(sdf_mask, dtype=torch.float32) # 5943
#     num_samples = num_sdf_samples + num_fs_samples # 164013
#     fs_weight = 1.0 - num_fs_samples / num_samples # 0.036
#     sdf_weight = 1.0 - num_sdf_samples / num_samples

#     return front_mask, sdf_mask, fs_weight, sdf_weight


# def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type):

#     front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

#     fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
#     sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

#     return fs_loss, sdf_loss


# def get_depth_loss(predicted_depth, target_d, loss_type='l2'):
#     depth_mask = torch.where(target_d > 0, torch.ones_like(target_d), torch.zeros_like(target_d))
#     eps = 1e-4
#     num_pixel = torch.size(depth_mask, out_type=torch.float32)
#     num_valid = torch.math.count_nonzero(depth_mask, dtype=torch.float32) + eps
#     depth_valid_weight = num_pixel / num_valid

#     return compute_loss(predicted_depth[..., torch.newaxis] * depth_mask, target_d * depth_mask, loss_type) * depth_valid_weight