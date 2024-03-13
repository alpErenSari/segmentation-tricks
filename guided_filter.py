import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_data
from boxfilter import boxfilter2d

class GuidedFilter2d(nn.Module):
    def __init__(self, radius: int, eps: float):
        super().__init__()
        self.r = radius
        self.eps = eps

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps)
        else:
            raise NotImplementedError

class myGuidedFilter2d(nn.Module):
    def __init__(self, radius: int, eps: float):
        super().__init__()
        self.r = radius
        self.eps = eps
        self.unnorm = utils_data.UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x, guide):
        guide = self.unnorm(guide).mean(1, keepdims=True)
        # guide = self.unnorm(guide)
        x_mean = x.mean((-2, -1), keepdims=True)
        x_std = x.std((-2, -1), keepdims=True)
        x = (x - x_mean) / (x_std + 1e-6)
        x_gf = guidedfilter2d_gray(guide, x, self.r, self.eps)
        # x_gf = guidedfilter2d_general(guide, x, self.r, self.eps)
        x_gf = (x_gf * x_std) + x_mean
        return x_gf

class FastGuidedFilter2d(GuidedFilter2d):
    """Fast guided filter"""
    def __init__(self, radius: int, eps: float, s: int):
        super().__init__(radius, eps)
        self.s = s

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps, self.s)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps, self.s)
        else:
            raise NotImplementedError

def guidedfilter2d_color(guide, src, radius, eps, scale=None):
    """guided filter for a color guide image
    
    Parameters
    -----
    guide: (B, 3, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    assert guide.shape[1] == 3
    if src.ndim == 3:
        src = src[:, None]
    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1) # b x 1 x H x W
    ones = torch.ones_like(guide_r)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N # b x 3 x H x W
    mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1) # b x 1 x H x W

    mean_p = boxfilter2d(src, radius) / N # b x C x H x W

    mean_Ip_r = boxfilter2d(guide_r * src, radius) / N # b x C x H x W
    mean_Ip_g = boxfilter2d(guide_g * src, radius) / N # b x C x H x W
    mean_Ip_b = boxfilter2d(guide_b * src, radius) / N # b x C x H x W

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p # b x C x H x W
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p # b x C x H x W
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p # b x C x H x W

    var_I_rr = boxfilter2d(guide_r * guide_r, radius) / N - mean_I_r * mean_I_r + eps # b x 1 x H x W
    var_I_rg = boxfilter2d(guide_r * guide_g, radius) / N - mean_I_r * mean_I_g # b x 1 x H x W
    var_I_rb = boxfilter2d(guide_r * guide_b, radius) / N - mean_I_r * mean_I_b # b x 1 x H x W
    var_I_gg = boxfilter2d(guide_g * guide_g, radius) / N - mean_I_g * mean_I_g + eps # b x 1 x H x W
    var_I_gb = boxfilter2d(guide_g * guide_b, radius) / N - mean_I_g * mean_I_b # b x 1 x H x W
    var_I_bb = boxfilter2d(guide_b * guide_b, radius) / N - mean_I_b * mean_I_b + eps # b x 1 x H x W

    # determinant
    cov_det = var_I_rr * var_I_gg * var_I_bb \
        + var_I_rg * var_I_gb * var_I_rb \
            + var_I_rb * var_I_rg * var_I_gb \
                - var_I_rb * var_I_gg * var_I_rb \
                    - var_I_rg * var_I_rg * var_I_bb \
                        - var_I_rr * var_I_gb * var_I_gb # b x 1 x H x W

    # inverse
    inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rg = - (var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det # b x 1 x H x W
    inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det # b x 1 x H x W
    inv_var_I_gb = - (var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det # b x 1 x H x W
    inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det # b x 1 x H x W

    inv_sigma = torch.stack([
        torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
        torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
        torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
    ], 1).squeeze(-3) # b x 3 x 3 x H x W

    cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1) # b x 3 x C x H x W

    a = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
    b = mean_p - a[:, 0] * mean_I_r - a[:, 1] * mean_I_g - a[:, 2] * mean_I_b # b x C x H x W

    mean_a = torch.stack([boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = torch.stack([F.interpolate(mean_a[:, i], guide.shape[-2:], mode='bilinear') for i in range(3)], 1)
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = torch.einsum("bichw,bihw->bchw", (mean_a, guide)) + mean_b

    return q

def guidedfilter2d_gray(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image
    
    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N  # (B, 1, H, W)
    mean_p = boxfilter2d(src, radius) / N # (B, C, H, W)
    mean_Ip = boxfilter2d(guide*src, radius) / N  # (B, C, H, W)
    cov_Ip = mean_Ip - mean_I * mean_p  # (B, C, H, W)

    mean_II = boxfilter2d(guide*guide, radius) / N # (B, 1, H, W)
    var_I = mean_II - mean_I * mean_I # (B, 1, H, W)

    a = cov_Ip / (var_I + eps)  # (B, C, H, W)
    b = mean_p - a * mean_I  # (B, C, H, W)

    mean_a = boxfilter2d(a, radius) / N  # (B, C, H, W)
    mean_b = boxfilter2d(b, radius) / N  # (B, C, H, W)

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = mean_a * guide + mean_b
    return q

def my_tensor_product(tensor1, tensor2):
    return torch.einsum('bckhw, bdkhw->bcdhw', tensor1, tensor2)


def guidedfilter2d_general(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image

    Parameters
    -----
    guide: (B, C, H, W)-dim torch.Tensor
        guide image
    src: (B, D, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1. / scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1. / scale, mode="nearest")
        radius = radius // scale

    B, C, H, W = guide.shape
    guide = guide.unsqueeze(2)
    src = src.unsqueeze(2)

    ones = torch.ones(B, 1, 1, H, W, device=src.device)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N  # (B, C, 1, H, W)
    mean_p = boxfilter2d(src, radius) / N  # (B, D, 1, H, W)
    # mean_Ip = boxfilter2d(guide * src, radius) / N
    mean_Ip = boxfilter2d(my_tensor_product(src, guide), radius) / N # (B, D, C, H, W)
    cov_Ip = mean_Ip - my_tensor_product(mean_p, mean_I)  # (B, D, C, H, W)

    mean_II = boxfilter2d(guide * guide, radius) / N  # (B, C, 1, H, W)
    var_I = mean_II - mean_I * mean_I  # (B, 1, H, W)
    var_I_eps = var_I + eps

    a = my_tensor_product(cov_Ip, 1/var_I_eps) # (B, C, H, W)
    b = mean_p - my_tensor_product(a, mean_I.transpose(1,2))  # (B, C, H, W)

    mean_a = boxfilter2d(a, radius) / N  # (B, C, H, W)
    mean_b = boxfilter2d(b, radius) / N  # (B, C, H, W)

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = my_tensor_product(mean_a, guide.transpose(1,2)) + mean_b
    return q.squeeze(2)
