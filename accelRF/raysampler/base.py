from typing import Optional
import torch
import numpy as np
import torch.utils.data as data
from ..datasets import BaseDataset
from .utils import get_rays,get_rays_openCV

class BaseRaySampler(data.Dataset):
    def __init__(
        self,
        dataset: Optional[BaseDataset],
        N_rand: int,
        length: Optional[int]=None,
        device: torch.device='cpu',
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
        ) -> None:
        super().__init__()
        self.dataset = dataset
        self.N_rand = N_rand
        self.length = length if length is not None else len(dataset)
        self.device = device

        # for distributed settings
        self.rng = torch.Generator(device=device)
        self.n_replica = n_replica
        self.rank = 0
        if rank >= 0:
            self.rank = rank
            self.rng.manual_seed(0)
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        output = {}
        img_dict = self.dataset[index]
        pose = img_dict['pose'][:3,:4]
        # cam_viewdir = img_dict['pose'][:3,2]
        rays_o, rays_d = get_rays(*self.dataset.get_hwf(), pose)
        output['rays_o'] = rays_o.reshape(-1,3)  # (N, 3)
        output['rays_d'] = rays_d.reshape(-1,3)  # (N, 3)
        if 'gt_img' in img_dict:
            output['gt_rgb'] = img_dict['gt_img'].reshape(-1,3) # (N, 3)

        return output


class RenderingRaySampler(BaseRaySampler):
    '''
    Just a alias of BaseRaySampler.
    '''
    def __init__(
        self, dataset: BaseDataset, N_rand: int=0, device: torch.device = 'cpu', 
        rank: int = -1, n_replica: int = 1, seed: Optional[int] = None) -> None:

        super().__init__(dataset, N_rand, length=None, device=device, rank=rank, n_replica=n_replica, seed=seed)

class VisRaySampler(BaseRaySampler):
    '''
        Alias for vis full image.
    '''
    def __init__(
        self, dataset: BaseDataset, N_rand: int=0, device: torch.device = 'cpu', 
        rank: int = -1, n_replica: int = 1, seed: Optional[int] = None) -> None:

        super().__init__(dataset, N_rand, length=None, device=device, rank=rank, n_replica=n_replica, seed=seed)
        H,W,_ = dataset.get_hwf()
        self.N_rand=H*W
        self.is_GL = dataset.is_GL
    
    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = {'rays_o': [], 'rays_d': [], 'gt_rgb': [], 'gt_dp': []}
        # img_i = torch.randint(len(self.dataset), (), generator=self.rng, device=self.device)
        img_dict = self.dataset.data
        pose = img_dict['poses'][index][:3,:4]
        # cam_viewdir = img_dict['pose'][:3,2]
        target_d = img_dict['depths'][index] # 480, 640
        target = img_dict['gt_img'][index].permute(1,2,0) # NOTE!!! rgb is CHW, transfer to HWC
        if self.is_GL:
            rays_o, rays_d = get_rays(*self.dataset.get_hwf(), pose, normalize_dir=True)
        else:
            rays_o, rays_d = get_rays_openCV(*self.dataset.get_HWK(), pose, normalize_dir=True)
        output['rays_o']=rays_o.reshape(-1, 3)  # (N_rand, 3)
        output['rays_d']=rays_d.reshape(-1, 3)  # (N_rand, 3)
        output['gt_rgb']=target.reshape(-1, 3)  # (N_rand, 3)
        output['gt_dp']=target_d.reshape(-1, 1) 

        return output