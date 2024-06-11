import torch
from math import sqrt, exp
import numpy as np

from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho

def proj_window_partition(x, window_size):
    h,w = x.shape       # x.shape = [256, 256], window_size = (32, 32)
    x = x.view(h // window_size[0], window_size[0], w // window_size[1], window_size[1]) # [256, 256] -> [8, 32, 8, 32]
    windows = x.permute(0, 2, 1, 3).contiguous().view(-1, window_size[0], window_size[1]) # [8, 32, 8, 32] -> [8, 8, 32, 32] -> [64, 32, 32]
    return windows

def ray_window_partition(x, window_size):
    h,w,c = x.shape       # x.shape = [256, 256, 8], window_size = (32, 32)
    x = x.view(h // window_size[0], window_size[0], w // window_size[1], window_size[1], c) # [256, 256, 8] -> [8, 32, 8, 32, 8]
    windows = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size[0], window_size[1], c) # x: [8, 32, 8, 32, 8] -> [8, 8, 32, 32, 8] -> [64, 32, 32, 8]
    return windows

class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs

class ImgToPatch_MLG(object):
    def __init__(self, ray_sampler, hwf, window_size=(32,32), window_num=4, n_rays=512):
        self.ray_sampler = ray_sampler
        self.hwf = hwf  # camera intrinsics
        self.window_size = window_size
        self.window_num = window_num
        self.n_rays = n_rays

    def __call__(self, img):
        rgbs = []
        for img_i in img: 

            rays = img_i.view(self.hwf[0], self.hwf[1], -1)  # Assuming img_i shape is [H, W, C]
            projs = torch.ones(self.hwf[0], self.hwf[1])

            rays_window = ray_window_partition(rays, self.window_size)
            projs_window = proj_window_partition(projs, self.window_size)
            
            # 筛选有效窗口
            projs_window_valid_indx = ((projs_window > 0).sum(dim=-1).sum(dim=-1) == self.window_size[0] * self.window_size[1])
            select_inds_window = np.random.choice(projs_window_valid_indx.shape[0], size=[self.window_num], replace=False)
            
            projs_window_select = projs_window[select_inds_window]  # [window_num, 32, 32]
            rays_window_select = rays_window[select_inds_window]    # [window_num, 32, 32, 8]
            
            selected_rays_window = rays_window_select.reshape(-1, 8)
            selected_projs_window = projs_window_select.flatten()
            
            total_inds = [i for i in range(projs_window.shape[0])]
            else_inds = [x for x in total_inds if x not in select_inds_window]
            projs_window_else = projs_window[else_inds]
            rays_window_else = rays_window[else_inds]
            
            else_inds_pixel_valid = projs_window_else > 0
            
            rays_else_valid = rays_window_else[else_inds_pixel_valid]
            projs_else_valid = projs_window_else[else_inds_pixel_valid]
            
            else_valid_select_index = np.random.choice(projs_else_valid.shape[0], size=[self.n_rays], replace=False)
            
            selected_rays_else = rays_else_valid[else_valid_select_index]
            selected_projs_else = projs_else_valid[else_valid_select_index]
            
          
            rgbs.append(selected_rays_else)
    
        rgbs = torch.cat(rgbs, dim=0)  # (B*N)x3

        return rgbs

class RaySampler(object):
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, focal, pose):
        if self.orthographic:
            size_h, size_w = focal      # Hacky
            rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        else:
            rays_o, rays_d = get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W)

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W)


class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        if self.scale_anneal>0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
            h = self.h * scale 
            w = self.w * scale 

        if self.random_shift:
            max_offset = 1-scale.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

            h += h_offset
            w += w_offset

        self.scale = scale

        return torch.cat([h, w], dim=2)
