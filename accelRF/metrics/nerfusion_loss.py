from typing import Dict, Tuple, Optional
from matplotlib.pyplot import step
import torch
from torch import Tensor
import torch.nn.functional as F

INF = 1e3

# @torch.jit.script
class nerfusion_Loss(object):
    def __init__(self, loss_weights, epsilon=0.02, steps=100, i_decay_epsilon=0) -> None:
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


    def get_loss(self, r_out, gt_rgb, gt_depth=None):
        losses = {}
        if 'rgb' in self.loss_w.keys():
            losses['rgb'] = ((r_out['rgb'] - gt_rgb)**2).mean() # remove .sum(-1), weights x3
        
        if 'alpha' in self.loss_w.keys():
            _alpha = r_out['acc'].reshape(-1)
            losses['alpha'] = F.mse_loss(_alpha, torch.ones_like(_alpha))

        if 'reg_term' in self.loss_w.keys():
            losses['reg_term'] = r_out['regz_term'].mean()

        if 'depth' in self.loss_w.keys() and gt_depth is not None: 
            dp_mask = gt_depth.ne(0)
            losses['depth'] = F.mse_loss(r_out['depth'][dp_mask],gt_depth[dp_mask])

        if 'region' in self.loss_w.keys() and gt_depth is not None:
            dp_mask = gt_depth.ne(0)
            t_vals = r_out['t_vals'][dp_mask]
            weights = r_out['weights'][dp_mask]
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
            expected_w.masked_fill_(~region_mask,0.0)
            expected_w = expected_w/expected_w.sum(-1,keepdim=True).expand_as(t_vals) # may cause nan CAUTION!!!

            losses['region']=F.mse_loss(weights[region_mask],expected_w[region_mask])
            losses['unregion']=F.mse_loss(weights[~region_mask],torch.zeros_like(weights[~region_mask]))

        loss = sum(losses[key] * self.loss_w[key] for key in losses)
        return loss, losses

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