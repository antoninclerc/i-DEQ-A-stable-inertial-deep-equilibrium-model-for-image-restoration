import torch
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity

# ============================================================
# GENERIC UTILITIES
# ============================================================

def add_zero_channel(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.zeros_like(x)], dim=1)

def process_image(tensor):
    return normalize_image(image_2ch_to_magnitude(tensor)).detach().cpu()

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean expected")

def normalize_problem(p):
    p = p.lower()
    if p == "mri":
        return "MRI"
    if p == "inpainting":
        return "inpainting"
    if p == "rician":
        return "rician"
    raise ValueError(p)

# ============================================================
# DATASET UTILITIES
# ============================================================

def load_pretrained(model, checkpoint_path, device=None, optimizer=None, load_optimizer=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    device = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
        state_dict = checkpoint
    else:
        state_dict = checkpoint["model_state_dict"]

    model_state = model.state_dict()

    # Find mismatches
    missing_keys = []
    unexpected_keys = []

    for key in model_state.keys():
        if key not in state_dict:
            missing_keys.append(key)

    for key in state_dict.keys():
        if key not in model_state:
            unexpected_keys.append(key)

    # Report unexpected keys (ignored)
    if unexpected_keys:
        print("Unexpected keys in checkpoint (ignored):")
        for k in unexpected_keys:
            print(f"  {k}")

    # Initialize missing keys randomly
    if missing_keys:
        print("Missing keys (randomly initialized):")
        for k in missing_keys:
            print(f"  {k}")
            param = model_state[k]
            if param.dtype.is_floating_point:
                model_state[k] = torch.randn_like(param)
            else:
                model_state[k] = param  # leave as is if not float

    # Load with strict=False
    model.load_state_dict(state_dict, strict=True)

    print(f"Loaded model weights from {checkpoint_path}")

    # Load optimizer if requested
    if load_optimizer and optimizer is not None:
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state")

# ============================================================
# COMPLEX / FFT OPERATIONS
# ============================================================

def image_2ch_to_magnitude(x):
    # x: B C=2 H W
    if x.shape[1] == 2:  # channel dimension
        x = torch.complex(x[:, 0, :, :], x[:, 1, :, :]).abs()
    return x

def normalize_image(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def tensor_2ch_to_complex(x):
    # x: B C=2 H W
    if x.shape[1] == 2:
        x = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
    return x

def complexe_to_tensor_2ch(x):
    # x: B H W complex
    x = torch.stack([x.real, x.imag], dim=1)  # channel dim = 1
    return x

def ifft2c(x):
    dim = (-2, -1)  # H, W dimensions
    x = tensor_2ch_to_complex(x)
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm='ortho')
    x = torch.fft.fftshift(x, dim=dim)
    x = complexe_to_tensor_2ch(x)
    return x

def fft2c(x):
    dim = (-2, -1)
    x = tensor_2ch_to_complex(x)
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm='ortho')
    x = torch.fft.fftshift(x, dim=dim)
    x = complexe_to_tensor_2ch(x)
    return x

# ============================================================
# DATA CONSISTENCY OPERATORS
# ============================================================

def data_consistency_func_gaussian(z, y, mask, forward_op):
    k_diff = forward_op(z, mask) - y
    value = 0.5 * torch.sum(k_diff**2, dim=(1,2,3))  # (B,)
    return value

def grad_data_consistency_gaussian(z, y, mask, forward_op, adjoint_op):
    k_diff = forward_op(z, mask) - y
    grad = adjoint_op(k_diff, mask)
    return grad

def data_consistency_func_rician(z, y, sigma, lambda_rician=5e-3):
    sigma2 = sigma ** 2
    logI0 = torch.log(torch.special.i0e((z * y) / sigma2) + 1e-12)
    value = z**2/(2 * sigma2) - logI0
    return lambda_rician * torch.sum(value, dim=(1,2,3))  # (B,)

def Bspecial(x):
    return torch.special.i1e(x) / (torch.special.i0e(x) + 1e-12)

def grad_data_consistency_rician(z, y, sigma, lambda_rician=5e-3):
    sigma2 = sigma ** 2
    B = Bspecial((z * y) / sigma2)
    grad = z / sigma2 - (y / sigma2) * B
    return lambda_rician * grad

def fast_irl1(z, y, sigma, lambda_dc, max_iter=10):
    f = y
    v = z
    irl1_input = z
    lambda_dc = lambda_dc.view(-1,1,1,1)  # Ensure lambda_dc is broadcastable
    f_sigma2 = f / (sigma ** 2)
    lamb_f_sigma2 = lambda_dc * f_sigma2
    lamb_sigma2_beta = lambda_dc / (sigma ** 2) + 1
    for _ in range(max_iter):
        Iz = Bspecial(f_sigma2 * v)
        Iz = torch.clamp(Iz, min=0)
        u = f_sigma2 * (1 - Iz)
        v = (lamb_f_sigma2 + irl1_input - lambda_dc * u) / lamb_sigma2_beta
    return v

# ============================================================
# OPTIMIZATION / ALGORITHMIC STEPS
# ============================================================

def one_step_PGD_back(image, obs, mask, 
                      DC, R, nabla_R, lambda_dc, lambda_Rtheta, 
                      gamma, eta, 
                      forward_op, noise_type, sigma):
    
    def f(z, y):
        if noise_type == 'gaussian':
            return data_consistency_func_gaussian(z, y, mask, forward_op)
        else:
            return data_consistency_func_rician(z, y, sigma, lambda_rician=1.0)
        
    def F(z, y):
        return lambda_Rtheta * R(z) + f(z, y)

    def prox(z, y, lambda_dc):
        if noise_type == 'gaussian':
            return DC(z, y, mask, lambda_dc)
        else:
            return DC(z, y, sigma, lambda_dc)
    
    image, tau = prox_backtracking(x=image, y=obs, prox_f=prox, nabla_R=nabla_R, F=F, 
                                   tau0=lambda_dc, lambda_Rtheta=lambda_Rtheta, 
                                   gamma=gamma, eta=eta)

    return image, tau

def one_step_GD_back(image, obs, mask, 
                     R, nabla_R, lambda_dc, lambda_Rtheta, 
                     gamma, eta, 
                     forward_op, adjoint_op, noise_type, sigma):
    
    def f(z, y):
        if noise_type == 'gaussian':
            return data_consistency_func_gaussian(z, y, mask, forward_op)
        else:
            return data_consistency_func_rician(z, y, sigma)

    def gradf(z, y):
        if noise_type == 'gaussian':
            return grad_data_consistency_gaussian(z, y, mask, forward_op, adjoint_op)
        else:
            return grad_data_consistency_rician(z, y, sigma)

    image, tau = backtracking(
            image, obs,
            f, gradf,
            R,
            nabla_R,
            lambda_dc,
            lambda_Rtheta,
            gamma,
            eta)
    return image, tau

def one_step(image, obs, mask, 
             DC, 
             nabla_R, lambda_dc, lambda_Rtheta, 
             DC_type, noise_type, sigma, forward_op, adjoint_op):
    
    # RED gradient step
    gradR = lambda_Rtheta * nabla_R(image)

    if DC_type == 'prox':
        image = image - lambda_dc * gradR
        if noise_type == 'gaussian':
            image_new = DC(image, obs, mask, lambda_dc)
        else:
            image_new = DC(image, obs, sigma, lambda_dc)

    elif DC_type == 'grad':
        if noise_type == 'gaussian':
            gradf = grad_data_consistency_gaussian(image, obs, mask, forward_op, adjoint_op)
            image_new = image - lambda_dc * (gradf + gradR)

        else:
            gradf = grad_data_consistency_rician(image, obs, sigma)
            image_new = image - lambda_dc * (gradf + gradR)
                
    return image_new, image

# ============================================================
# LINE SEARCH / BACKTRACKING
# ============================================================

def backtracking(x, y, f, gradf, R, gradR, tau0, lambda_Rtheta, gamma, eta):
    B = x.shape[0]

    if tau0.ndim == 0:
        tau = torch.full((B,), tau0, device=x.device, dtype=x.dtype)
    else:
        tau = tau0

    def Phi(x_, y_):
        return f(x_, y_) + lambda_Rtheta * R(x_)

    def grad_phi(x_):
        return gradf(x_, y) + lambda_Rtheta * gradR(x_)

    Phix = Phi(x, y)
    grad_phi_x = grad_phi(x)
    grad_norm_sq = torch.sum(grad_phi_x**2, dim=(1,2,3))

    active = torch.ones(B, dtype=torch.bool, device=x.device)

    with torch.no_grad():
        n_iter = 0
        while active.any():

            active_idx = torch.nonzero(active, as_tuple=True)[0]

            Tx_try = x - tau.view(-1,1,1,1) * grad_phi_x
            Phi_try = Phi(Tx_try, y)

            armijo_rhs = Phix - gamma * tau * grad_norm_sq

            condition = Phi_try <= armijo_rhs

            for i, idx in enumerate(active_idx):
                tau[idx] = tau[idx] if condition[i] else tau[idx] * eta
                if tau[idx] < 1e-10:
                    tau[idx] = 0.0  # Avoid excessively small step sizes

            for i, idx in enumerate(active_idx):
                if condition[i]:
                    active[idx] = False

            n_iter += 1
            if n_iter > 100:
                print("Warning: backtracking exceeded 100 iterations")
                raise RuntimeError("Backtracking line search did not converge after 100 iterations")

    Tx = x - tau.view(-1,1,1,1) * grad_phi_x

    return Tx, tau

def prox_backtracking(
    x, y,
    prox_f,
    nabla_R,
    F,
    tau0,
    lambda_Rtheta,
    gamma,
    eta
):
    B = x.shape[0]
    
    if tau0.ndim == 0:
        tau = torch.full((B,), tau0, device=x.device, dtype=x.dtype)
    else:
        tau = tau0

    xk = x.clone()
    Fxk = F(x, y)
    nabla_Rxk = lambda_Rtheta * nabla_R(xk)

    with torch.no_grad():

        active = torch.ones(B, dtype=torch.bool, device=x.device)

        while active.any():

            active_idx = torch.nonzero(active, as_tuple=True)[0]
            zk = xk - tau.view(-1,1,1,1) * nabla_Rxk

            x_next = prox_f(zk, y, tau)
            Fx_next = F(x_next, y)

            lhs = Fxk - Fx_next
            rhs = gamma / tau * torch.sum((xk - x_next)**2, dim=(1,2,3))

            condition = lhs >= rhs

            for i, idx in enumerate(active_idx):
                tau[idx] = tau[idx] if condition[i] else tau[idx] * eta
                if tau[idx] < 1e-10:
                    tau[idx] = 1e-10  # Avoid excessively small step sizes
                    condition[i] = True  # Stop backtracking if step size is too small

            for i, idx in enumerate(active_idx):
                if condition[i]:
                    active[idx] = False
                if lhs[i] == 0.0:
                    active[idx] = False  # If no improvement, stop backtracking

    zk = xk - tau.view(-1,1,1,1) * nabla_Rxk
    x_next = prox_f(zk, y, tau)

    return x_next, tau

def restart_condition(intermediates, B_restart):

    if len(intermediates) < 2:
        return False

    with torch.no_grad():

        sum_iter = 0.0
        k = len(intermediates) - 1

        for t in range(k):
            diff = intermediates[t+1] - intermediates[t]
            norm = torch.sum(diff * diff)
            sum_iter += norm.item()

        return k * sum_iter >= B_restart ** 2

# ============================================================
# METRICS
# ============================================================

def MSE(output, target):
    return mean_squared_error(
        output.detach().cpu().numpy(),
        target.detach().cpu().numpy()
    )
    
def MSE_batch(output, target):
    if output.ndim == 4:  # Batch
        mse_list = [
            mean_squared_error(
                output[b].detach().cpu().numpy(),
                target[b].detach().cpu().numpy()
            )
            for b in range(output.shape[0])
        ]
    else:  # Single image
        mse_list = [
            mean_squared_error(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy()
            )
        ]

    return torch.tensor(mse_list, device=output.device)

def PSNR(output, target):
    if output.ndim >= 3:  # Batch
        psnr_list = [
            peak_signal_noise_ratio(
                output[b].detach().cpu().numpy(),
                target[b].detach().cpu().numpy(),
                data_range=1.0
            )
            for b in range(output.shape[0])
        ]
    else:  # Single image
        psnr_list = [
            peak_signal_noise_ratio(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                data_range=1.0
            )
        ]

    return torch.tensor(psnr_list, device=output.device)

def SSIM(output, target):
    if output.ndim >= 3:  # Batch
        ssim_list = [
            structural_similarity(
                target[b].detach().cpu().numpy(),
                output[b].detach().cpu().numpy(),
                data_range=target[b].max().item(),
                channel_axis=0 if output.shape[1] == 3 else None
                
            )
            for b in range(output.shape[0])
        ]
    else:  # Single image
        ssim_list = [
            structural_similarity(
                target.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
                data_range=target.max().item(),
                channel_axis=0 if output.shape[0] == 3 else None
            )
        ]

    return torch.tensor(ssim_list, device=output.device)

def compute_batch_metrics(output, target, input_image=None):
    # Convert to magnitude if complex
    if output.shape[1] == 2:
        output = image_2ch_to_magnitude(output)
        target = image_2ch_to_magnitude(target)
    if output.ndim == 4:
        if output.shape[1] == 1:
            output = output[:, 0, :, :]
            target = target[:, 0, :, :]

    if input_image is not None and input_image.shape[1] == 2:
        input_image = image_2ch_to_magnitude(input_image)
    if input_image is not None and input_image.ndim == 4: 
        if input_image.shape[1] == 1:
            input_image = input_image[:, 0, :, :]

    # MSE per sample (B,)
    mse = MSE_batch(output, target)

    psnr = PSNR(output, target)

    ssim = SSIM(output, target)

    mse_list = mse.detach().cpu().tolist()
    psnr_list = psnr.detach().cpu().tolist()
    ssim_list = ssim.detach().cpu().tolist()

    if input_image is not None:
        psnr_input = PSNR(input_image, target)
        input_psnr_list = psnr_input.detach().cpu().tolist()
    else:
        input_psnr_list = None

    return mse_list, psnr_list, ssim_list, input_psnr_list


# ============================================================
# TRAINING SETUP
# ============================================================

def combined_loss(output, target, eta_k=None, eta_TV=None, eta_l1=None, model=None, sigma=None):
    # MSE across all channels (including complex channels)
    if sigma is None:
        total_loss = torch.mean((output - target) ** 2)
    else:
        total_loss = torch.mean(((output - target) / sigma) ** 2)
    
    # Frequency-domain consistency
    if eta_k is not None:
        k_out = fft2c(output)    # output already has real+imag
        k_tgt = fft2c(target)
        loss_k = torch.mean(torch.abs(k_out - k_tgt) ** 2)
        total_loss += eta_k * loss_k

    # Total Variation (anisotropic) per channel
    if eta_TV is not None:
        dx = output[:, :, 1:, :] - output[:, :, :-1, :]
        dy = output[:, :, :, 1:] - output[:, :, :, :-1]
        tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        total_loss += eta_TV * tv_loss

    # L1 regularization on model parameters
    if eta_l1 is not None and model is not None:
        l1_loss = sum(p.abs().mean() for p in model.parameters())
        total_loss += eta_l1 * l1_loss

    return total_loss

def setup_training(
    model,
    device,
    network=None,
    lr=1e-3,
    eta_k=None,
    eta_TV=None,
    eta_l1=None,
    optimizer=torch.optim.Adam,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    optimizer_kwargs=None,
    scheduler_kwargs=None,
):
    model.to(device)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    def criterion(output, target, sigma):
        return combined_loss(
            output,
            target,
            eta_k=eta_k,
            eta_TV=eta_TV,
            eta_l1=eta_l1,
            model=network,
            sigma=sigma
        )

    optimizer_instance = optimizer(model.parameters(), lr=lr, **optimizer_kwargs)
    if scheduler is not None:
        scheduler_instance = scheduler(optimizer_instance, **scheduler_kwargs)
    else:
        scheduler_instance = None

    return criterion, optimizer_instance, scheduler_instance

# ============================================================
# VISUALIZATION / PLOTTING
# ============================================================

def plot_training_state(
    epoch,
    train_losses,
    val_losses,
    train_PSNRs,
    val_PSNRs,
    outputs,
    target,
    save_path
):
    plt.figure(figsize=(10, 10))

    # --- Loss ---
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()

    # --- PSNR ---
    plt.subplot(2, 2, 2)
    plt.plot(train_PSNRs, label="Train PSNR")
    plt.plot(val_PSNRs, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    # --- Reconstruction ---
    plt.subplot(2, 2, 3)
    if outputs.shape[1] == 2:  # complex
        image = image_2ch_to_magnitude(outputs)[-1]
        type = "greyscale"
    elif outputs.shape[1] == 3:  # RGB
        image = outputs[-1].permute(1, 2, 0)  # C H W -> H W C
        type = "color"
    else:
        image = outputs[-1, 0, :, :]
        type = "greyscale"
    plt.title("Reconstructed Image")
    if type == "color":
        plt.imshow(image.detach().cpu())
    else:
        plt.imshow(image.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    # --- Ground truth ---
    plt.subplot(2, 2, 4)
    if target.shape[1] == 2:  # complex
        image = image_2ch_to_magnitude(target)[-1]
        type = "greyscale"
    elif target.shape[1] == 3:  # RGB
        image = target[-1].permute(1, 2, 0)  # C H W -> H W C
        type = "color"
    else:
        type = "greyscale"

        image = target[-1, 0, :, :]
    plt.title("Ground Truth Image")
    if type == "color":
        plt.imshow(image.detach().cpu())
    else:
        plt.imshow(image.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Val_{epoch}.pdf"))
    plt.close()

def show_image(ax, img, title, is_error=False):
    ax.set_title(title)

    # Conversion tensor → numpy
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    # Convert (C,H,W) → (H,W,C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # Gestion grayscale
    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
        img = img.squeeze()
        if is_error:
            ax.imshow(img, cmap="hot")
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)

    ax.axis("off")

def validation_and_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    epoch_val_mse,
    best_val_loss,
    patience,
    max_epochs,
    save_path,
    min_iter=50,
):
    # --- Early stopping ---
    if epoch_val_mse < best_val_loss:
        best_val_loss = epoch_val_mse
        patience = 0
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "best_model.pth"),
        )
    elif epoch > min_iter:
            patience += 1
        
    # --- Scheduler ---
    if scheduler is not None:
        if isinstance(
            scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            scheduler.step(epoch_val_mse)
        else:
            scheduler.step()

    # --- Periodic checkpoint ---
    save_interval = max_epochs // 10
    if save_interval > 0 and epoch % save_interval == 0:
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth"),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(save_path, f"optimizer_checkpoint_epoch_{epoch}.pth"),
        )

    return best_val_loss, patience

def jacobian_free_backpropagation(
    z_fixed,
    z_fixed_before,
    mask,
    loss_fn,
    target,
    y,
    params,
    DC,
    DC_type,
    forward_op,
    adjoint_op,
    noise_type,
    sigma,
    random_noise,
    Rtheta,
    nabla_x_network,
    lambda_dc,
    lambda_Rtheta,
    gamma,
    eta,
    theta,
    accelerated,
    backtracking,
    K_JFB,
):
    # 1) Differentiable leaf
    z = z_fixed.detach().requires_grad_(True)
    z_before = z_fixed_before.detach()  # No grad needed for previous iterate
    
    # 2) Compute loss
    if random_noise:
        noise_loss = sigma
    else:
        noise_loss = None

    loss = loss_fn(z, target, sigma=noise_loss)

    # 3) Gradient g = dL/dz
    g = torch.autograd.grad(loss, z, allow_unused=False)[0]

    if accelerated:
        z_iter = z + (1 - theta) * (z- z_before)
    else:
        z_iter = z
    
    # 4) Single-step evaluation f_theta(z)
    tau0 = lambda_dc if isinstance(lambda_dc, float) else lambda_dc
    if backtracking:
        if DC_type == "prox":
            fz, _ = one_step_PGD_back(
                            image=z_iter, obs=y, mask=mask, DC=DC, R=Rtheta,
                            nabla_R=nabla_x_network,
                            lambda_dc=tau0, lambda_Rtheta=lambda_Rtheta,
                            gamma=gamma, eta=eta,
                            forward_op=forward_op, noise_type=noise_type, sigma=sigma
                        )
            
        elif DC_type == "grad":
            fz, _ = one_step_GD_back(
                image=z_iter, obs=y, mask=mask, R=Rtheta,
                nabla_R=nabla_x_network,
                lambda_dc=tau0, lambda_Rtheta=lambda_Rtheta,
                gamma=gamma, eta=eta,
                forward_op=forward_op, adjoint_op=adjoint_op,
                noise_type=noise_type, sigma=sigma)
        else:
            raise ValueError("Unsupported DC for backtracking")
    else:
        fz, _ = one_step(
                image=z_iter, obs=y, mask=mask, DC=DC, nabla_R=nabla_x_network, 
                lambda_dc=tau0, lambda_Rtheta=lambda_Rtheta, 
                DC_type=DC_type, noise_type=noise_type, sigma=sigma,
                forward_op=forward_op, adjoint_op=adjoint_op)

    # 5) Neumann approximation of (I - J_f^T)^{-1} g
    v = g
    acc = g

    if K_JFB > 0:
        for _ in range(K_JFB):
            v = torch.autograd.grad(
                fz, z,
                grad_outputs=v,
                retain_graph=True,
                allow_unused=False
            )[0]

        acc = acc + v

    # 6) Gradients w.r.t parameters
    grads = torch.autograd.grad(
        fz,
        params,
        grad_outputs=acc,
        retain_graph=False,
        allow_unused=False
    )

    # 7) Assign gradients
    for p, gparam in zip(params, grads):
        if gparam is None:
            print('No grad !')
            p.grad = torch.zeros_like(p, device=p.device)
        else:
            p.grad = gparam

    return loss