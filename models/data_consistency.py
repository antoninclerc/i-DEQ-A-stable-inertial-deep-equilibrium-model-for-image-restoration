import torch
import matplotlib.pyplot as plt
from utils import ifft2c, fft2c, grad_data_consistency_gaussian, grad_data_consistency_rician, fast_irl1

# Data consistency for MRI reconstruction

def DC_prox_MRI(image, k0, mask, lambda_dc=1.):
    lambda_dc = 1 / lambda_dc
    Ah_k0 = ifft2c(k0)
    lambda_dc = lambda_dc.view(-1, 1, 1, 1) if isinstance(lambda_dc, torch.Tensor) else lambda_dc
    rhs = Ah_k0 + lambda_dc * image

    RHS_k = fft2c(rhs)
    denom = mask + lambda_dc
    denom_2ch = torch.cat(
        [denom, denom], dim=1)

    x_dc_k_2ch = RHS_k / denom_2ch

    x_dc = ifft2c(x_dc_k_2ch)
    return x_dc

def DC_grad_MRI(image, k0, mask, lambda_dc=1.):
    gradf = grad_data_consistency_gaussian(image, k0, mask, forward_op=Forward_MRI, adjoint_op=Adjoint_MRI)
    return image - lambda_dc * gradf

def Forward_MRI(image, mask):
    kspace = fft2c(image)
    mask_2ch = torch.cat([mask, mask], dim=1)
    return kspace * mask_2ch

def Adjoint_MRI(k0, mask):
    mask_2ch = torch.cat([mask, mask], dim=1)
    Ah_k0 = ifft2c(k0 * mask_2ch)
    return Ah_k0

# Data consistency for random inpainting

def DC_prox_inpainting(image, image_obs, mask, lambda_dc=1.):

    n_ch = image.shape[1]
    lambda_dc = 1 / lambda_dc

    if mask.shape[1] != n_ch:
        mask_nch = torch.stack([mask] * n_ch, dim=1)
    else:
        mask_nch = mask

    if isinstance(lambda_dc, torch.Tensor):
        lambda_dc = lambda_dc.view(-1,1,1,1)

    rhs = mask_nch * image_obs + lambda_dc * image
    denom = mask_nch + lambda_dc

    x_dc_nch = rhs / denom

    return x_dc_nch

def DC_grad_inpainting(image, image_obs, mask, lambda_dc=1.):
    gradf = grad_data_consistency_gaussian(image, image_obs, mask, forward_op=Forward_inpainting, adjoint_op=Adjoint_inpainting)
    return image - lambda_dc * gradf

def Forward_inpainting(image, mask):
    return image * mask

def Adjoint_inpainting(image_obs, mask):
    return image_obs * mask

def Forward_Rician(image, *kwargs):
    return image

def Adjoint_Rician(image_obs, *kwargs):
    return image_obs

def DC_grad_Rician(image, image_obs, sigma, lambda_dc=1.):
    grad = grad_data_consistency_rician(image, image_obs, sigma)
    return image - lambda_dc * grad

def DC_prox_Rician(image, image_obs, sigma, lambda_dc=1., max_iter=10):
    lambda_dc = 1 / lambda_dc
    return fast_irl1(image, image_obs, sigma, lambda_dc, max_iter)

