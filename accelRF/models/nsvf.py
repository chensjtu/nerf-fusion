from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['NSVF_MLP', 'BackgroundField', 'SDFNet', "RGBNet", "LaplaceDensity"]

# start with two help nn.Modules

class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/master/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        self.net += [nn.ReLU()]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 

class ImplicitField(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False, with_ln=True, skips=[], spec_init=True):
        super().__init__()
        self.skips = skips
        self.net = []

        prev_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == (num_layers - 1) else hidden_dim
            if (i == (num_layers - 1)) and outmost_linear:
                self.net.append(nn.Linear(prev_dim, next_dim))
            else:
                self.net.append(FCLayer(prev_dim, next_dim, with_ln=with_ln))
            prev_dim = next_dim
            if (i in self.skips) and (i != (num_layers - 1)):
                prev_dim += in_dim
        
        if num_layers > 0:
            self.net = nn.ModuleList(self.net)
            if spec_init:
                self.net.apply(self.init_weights)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.net):
            if i-1 in self.skips:
                y = torch.cat((x, y), dim=-1)
            y = layer(y)
        return y

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

### nsvf MLP
class NSVF_MLP(nn.Module):
    def __init__(self, 
        in_ch_pts: int=416, in_ch_dir: int=24,
        D_feat: int=3, W_feat: int=256, skips_feat: List[int]=[],
        D_sigma: int=2, W_sigma: int=128, skips_sigma: List[int]=[],
        D_rgb: int=5, W_rgb: int=256, skips_rgb: List[int]=[],
        layernorm: bool=True, with_activation: bool=False, 
        with_reg_term: bool=True 
        ):
        '''
        Re-organize NSVF's MLP definition into one simple nn.Module, easier for users to see 
        the whole NN architecture and to make their own modifications.
        NSVF's core MLP is actually very similar to NeRF's MLP
        '''
        super().__init__()
        self.with_reg_term = with_reg_term
        self.feat_layers = \
            ImplicitField(in_ch_pts, W_feat, W_feat, D_feat, with_ln=layernorm, skips=skips_feat)

        self.sigma_layers = nn.Sequential(
            ImplicitField(W_feat, 1, W_sigma, D_sigma, 
                with_ln=layernorm, skips=skips_sigma, outmost_linear=True),
            nn.ReLU(True) if with_activation else nn.Identity() # maybe softplus
        )

        out_ch_rgb = 3 # or 4 if with_alpha
        self.rgb_layers = nn.Sequential(
            ImplicitField(W_feat+in_ch_dir, out_ch_rgb, W_rgb, D_rgb,
                with_ln=layernorm, skips=skips_rgb, outmost_linear=True),
            nn.Sigmoid() if with_activation else nn.Identity()
        )
    
    def detach_model(self,detach_list=None):
        '''
            detach specific layers of the MLP module
        '''
        if detach_list is None:
            detach_list = ['feat_layers', 'rgb_layers','sigma_layers']
        if 'feat_layers' in detach_list:
            for name, module in self.feat_layers._modules.items():
                for p in module.parameters():
                    p.requires_grad = False
        if 'rgb_layers' in detach_list:
            for name, module in self.rgb_layers._modules.items():
                for p in module.parameters():
                    p.requires_grad = False
        if 'sigma_layers' in detach_list:
            for name, module in self.sigma_layers._modules.items():
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, pts: Tensor, dir: Optional[Tensor]=None):
        feat = self.feat_layers(pts)
        sigma = self.sigma_layers(feat)
        ret = {'sigma': sigma}
        if self.with_reg_term and self.training:
            ret['feat_n2'] = (feat ** 2).sum(-1)
        if dir is None:
            return ret
        rgb_input = torch.cat([feat, dir], -1)
        rgb = self.rgb_layers(rgb_input)
        ret['rgb'] = rgb
        return ret

class BackgroundField(nn.Module):
    """
    Background (we assume a uniform color)
    """
    def __init__(self, out_dim=3, bg_color=1.0, min_color=0, 
        trainable=True, background_depth=5.0):
        '''
        Args:
            bg_color: List or Scalar
            min_color: int
        '''
        super().__init__()

        if out_dim == 3:  # directly model RGB
            if isinstance(bg_color, List) or isinstance(bg_color, Tuple):
                assert len(bg_color) in [3,1], "bg_color should have size 1 or 3"
            else:
                bg_color = [bg_color]
            if min_color == -1:
                bg_color = [b * 2 - 1 for b in bg_color]
            if len(bg_color) == 1:
                bg_color = bg_color + bg_color + bg_color
            bg_color = torch.tensor(bg_color).float()
        else:    
            bg_color = torch.ones(out_dim).uniform_()
            if min_color == -1:
                bg_color = bg_color * 2 - 1
        self.out_dim = out_dim
        self.bg_color = nn.Parameter(bg_color, requires_grad=trainable)
        self.depth = background_depth

    def forward(self, x):
        return self.bg_color.unsqueeze(0).expand(
            *x.size()[:-1], self.out_dim)

### volsdf MLP
class SDFNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=False,
            bias=0.0,
            skip_in=(),
            weight_norm=False,
            sphere_scale=1.0,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims # + [d_out + feature_vector_size] 
        self.weight_norm = weight_norm
        self.num_layers = len(dims) - 2
        self.skip_in = skip_in

        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            # if l + 1 in self.skip_in:
            #     out_dim = dims[l + 1] - dims[0] # modified to add extra data
            # else:
            #     out_dim = dims[l + 1]

            if l + 1 in self.skip_in:
                input_dim = dims[l] + d_in
            else:
                input_dim = dims[l]

            out_dim = dims[l + 1]
            lin = nn.Linear(input_dim, out_dim)

            if geometric_init:
                if d_in > 3 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif d_in > 3 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            self.lins.append(lin)

        # last layer.
        self.sdf_layer = nn.Linear(dims[-1], d_out)
        self.feat_layer = nn.Linear(dims[-1], feature_vector_size)

        for lin in [self.sdf_layer, self.feat_layer]:
            if geometric_init:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.softplus = nn.Softplus(beta=100) # softplus!

    def forward(self, input: Tensor, xyz: Optional[Tensor]=None, sdf_only: bool=False):
        x = input
        for i, lin in enumerate(self.lins):
            if i + 1 in self.skip_in:
                x = torch.cat([x, input], -1) / (2**0.5)
            x = lin(x)
            x = self.softplus(x)
        sdf = self.sdf_layer(x)

        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0 and xyz is not None:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - xyz.norm(2, -1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)

        if sdf_only:
            return sdf, None
        else:
            return sdf, self.feat_layer(x)

class RGBNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=False,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.weight_norm = weight_norm
        self.num_layers = len(dims) - 1

        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            self.lins.append(lin)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, view_dirs: Tensor, feature_vectors: Tensor, 
        points: Optional[Tensor]=None, normals: Optional[Tensor]=None
    ):
        if points is not None and normals is not None:    # idr
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        else:                     # nerf
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < self.num_layers - 1:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


######### Density #########

class Density(nn.Module):
    def __init__(self, beta=None):
        super().__init__()
        if beta is not None:
            self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

def laplace_fn(sdf: Tensor, beta: Tensor):
    alpha = 1 / beta
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta, beta_min=0.0001):
        super().__init__(beta)
        self.register_buffer('beta_min', torch.tensor(beta_min))

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)

class SimpleDensity(Density):  # like NeRF
    def __init__(self, beta, noise_std=1.0):
        super().__init__(beta)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)

