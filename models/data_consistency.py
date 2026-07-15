import torch
import torch.nn.functional as F
from utils import (
    ifft2c, 
    fft2c, 
    grad_data_consistency_gaussian, 
    grad_data_consistency_rician, 
    fast_irl1)
import utils_deblur as deblur
# ----------------------------
# Data consistency for MRI
# ----------------------------

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
    gradf = grad_data_consistency_gaussian(image, k0, mask, 
                                           forward_op=Forward_MRI, 
                                           adjoint_op=Adjoint_MRI)
    return image - lambda_dc * gradf

def Forward_MRI(image, mask):
    kspace = fft2c(image)
    mask_2ch = torch.cat([mask, mask], dim=1)
    return kspace * mask_2ch

def Adjoint_MRI(k0, mask):
    mask_2ch = torch.cat([mask, mask], dim=1)
    Ah_k0 = ifft2c(mask_2ch * k0)
    return Ah_k0

# ----------------------------
# Data consistency for inpainting
# ----------------------------

def DC_prox_inpainting(image, image_obs, mask, lambda_dc=1.):

    n_ch = image.shape[1]
    lambda_dc = 1/lambda_dc

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
    gradf = grad_data_consistency_gaussian(image, image_obs, mask, 
                                           forward_op=Forward_inpainting, 
                                           adjoint_op=Adjoint_inpainting)
    return image - lambda_dc * gradf

def Forward_inpainting(image, mask):
    return image * mask

def Adjoint_inpainting(image_obs, mask):
    return image_obs * mask

# ----------------------------
# Data consistency for Rician noise
# ----------------------------

def Forward_Rician(image, *kwargs):
    return image

def Adjoint_Rician(image_obs, *kwargs):
    return image_obs

def DC_grad_Rician(image, image_obs, sigma, lambda_dc=1.):
    grad = grad_data_consistency_rician(image, image_obs, sigma)
    return image - lambda_dc * grad

def DC_prox_Rician(image, image_obs, sigma, 
                   lambda_dc=1., max_iter=10):
    return fast_irl1(image, image_obs, sigma, lambda_dc, max_iter)

# ----------------------------
# Data consistency for Debluring
# ----------------------------

def Forward_deblurring(x, kernel):
    # x: [B, 3, H, W]
    # kernel: [B, 1, kH, kW]
    fft_k = deblur.p2o(kernel, x.shape[-2:])
    return torch.real(deblur.ifftn(fft_k * deblur.fftn(x)))

def Adjoint_deblurring(x, kernel):
    # x: [B, 3, H, W]
    # kernel: [B, 1, kH, kW]
    
    fft_k = deblur.p2o(kernel, x.shape[-2:])
    fft_kH = torch.conj(fft_k)
    return torch.real(deblur.ifftn(fft_kH * deblur.fftn(x)))

def DC_grad_deblurring(x, y, kernel, lambda_dc=1.0):
    # x: [B, 3, H, W]
    # y: [B, 3, H, W]
    # kernel: [B, 1, kH, kW]
    # lambda_dc: scalar

    # Compute the forward operation
    fft_k = deblur.p2o(kernel, x.shape[-2:])
    fft_kH = torch.conj(fft_k)
    abs_k = fft_kH * fft_k

    grad = abs_k * deblur.fftn(x) - fft_kH * deblur.fftn(y)
    grad = torch.real(deblur.ifftn(grad))
    return x - lambda_dc * grad

def DC_prox_deblurring(image, image_obs, kernel, lambda_dc=1.0):
    H_img, W_img = image.shape[-2:]

    # Pad kernel to image size
    kernel_pad = torch.zeros(
        (H_img, W_img),
        dtype=image.dtype,
        device=image.device,
    )

    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel

    # Center kernel
    kernel_pad = torch.fft.ifftshift(kernel_pad)

    H_fft = torch.fft.fft2(kernel_pad)
    V_fft = torch.fft.fft2(image)
    Y_fft = torch.fft.fft2(image_obs)

    X_fft = (
        V_fft + lambda_dc * torch.conj(H_fft) * Y_fft
    ) / (
        1.0 + lambda_dc * torch.abs(H_fft) ** 2
    )

    return torch.fft.ifft2(X_fft).real