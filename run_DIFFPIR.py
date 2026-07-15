import os
import sys
import numpy as np
import torch
import argparse
import random
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from networks.DRUnet import GSDRUNet

from utils import add_zero_channel, PSNR, ifft2c, image_2ch_to_magnitude, SSIM, show_image
from models.data_consistency import DC_prox_MRI, DC_prox_inpainting, DC_prox_Rician
from gen_data import get_dataloaders


# =========================
# Configuration object
# =========================
class Config:
    def __init__(
        self,
        lambda_=0.13,
        zeta=0.8,
        diffusion_steps=20,
        t_start=200,
        noise_level_img=1.,
        degradation="mri",
        device="cuda:0",
    ):
        self.lambda_ = lambda_
        self.zeta = zeta
        self.diffusion_steps = diffusion_steps
        self.t_start = t_start
        self.device = device
        self.noise_level_img = noise_level_img
        self.degradation = degradation
        if degradation == "mri":
            self.prox = DC_prox_MRI
        elif degradation == "inpainting":
            self.prox = DC_prox_inpainting
        else:
            self.prox = DC_prox_Rician
# =========================
# Diffusion utilities
# =========================
def get_alphas(T, device):
    betas = np.linspace(0.1 / 1000, 20 / 1000, T, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    return torch.tensor(np.cumprod(alphas.cpu(), axis=0)).to(device)


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


# =========================
# Core restoration
# =========================
def restore(cfg: Config, dataloader):

    if cfg.degradation == "mri":
        model = GSDRUNet(
            in_channels=1,
            out_channels=1,
            pretrained="networks/GSDRUNet_grayscale_torch.ckpt",
        ).to(cfg.device)
    else:
        model = GSDRUNet(
            in_channels=3,
            out_channels=3,
            pretrained='./networks/GS_DRUNet_SPlus.ckpt', 
            act_mode='s'
        ).to(cfg.device)


    def denoiser(x, sigma):
        if cfg.degradation == "mri":
            x_in = x[:, 0, :, :].unsqueeze(1)
        else:
            x_in = x
        x_out = model(x_in, sigma)
        if cfg.degradation == "mri":
            x_out = add_zero_channel(x_out)
        return x_out

    psnr_list = []
    ssim_list = []
    for x_true, y, params in dataloader:

        x_true = x_true.to(cfg.device)
        y = y.to(cfg.device)
        mask = params['mask'].to(cfg.device)
        if cfg.degradation != "rician":
            y = y + cfg.noise_level_img / 255.0 * torch.randn_like(y)

        if cfg.degradation == "mri":
            y = y * mask
        
        T = 1000
        alphas = get_alphas(T, cfg.device)
        sigmas = torch.sqrt(1.0 - alphas) / torch.sqrt(alphas)

        rhos = cfg.lambda_ * (cfg.noise_level_img / 255.0) ** 2 / (sigmas**2)

        seq = np.sqrt(np.linspace(0, cfg.t_start**2, cfg.diffusion_steps))
        seq = [int(s) for s in seq]
        seq[-1] -= 1

        if cfg.degradation == "mri":
            x0 = ifft2c(y)
            x0 = torch.clamp(x0, 0, 1)
        elif cfg.degradation == "inpainting":
            x0 = mask * y + (1 - mask) * 0.5
        else:
            x0 = y

        time_start = time.time()
        x = 2 * x0 - 1

        x = torch.clip(
            torch.sqrt(alphas[cfg.t_start]) * x
            + torch.sqrt(1 - alphas[cfg.t_start]) * torch.randn_like(x),
            0,
            1,
        )

        with torch.no_grad():
            for i in tqdm(range(len(seq))):
                curr_sigma = sigmas[cfg.t_start - 1 - seq[i]].item()
                x0 = denoiser(x, curr_sigma)

                if i != len(seq) - 1:
                    t_i = find_nearest(sigmas.cpu(), curr_sigma)

                    x0 = cfg.prox(
                        x0, y, mask, 1 / (2 * rhos[t_i])
                    )

                    x0 = 2 * x0 - 1
                    x = 2 * x - 1

                    next_sigma = sigmas[cfg.t_start - 1 - seq[i + 1]].item()
                    t_im1 = find_nearest(sigmas.cpu(), next_sigma)

                    eps = (x - torch.sqrt(alphas[t_i]) * x0) / torch.sqrt(
                        1.0 - alphas[t_i]
                    )

                    x = torch.sqrt(alphas[t_im1]) * x0 + torch.sqrt(
                        1.0 - alphas[t_im1]
                    ) * (
                        np.sqrt(1 - cfg.zeta) * eps
                        + np.sqrt(cfg.zeta) * torch.randn_like(x)
                    )

                    x = (x + 1) / 2
            time_end = time.time()
            print(f"Restoration took {time_end - time_start:.2f} seconds.")

            x_mag = torch.clamp(x, 0, 1)
            if cfg.degradation == "mri":
                x_mag = image_2ch_to_magnitude(x_mag)
                x_true_mag = image_2ch_to_magnitude(x_true)
            else:
                x_true_mag = x_true

            output_psnr = PSNR(x_true_mag, x_mag).cpu()
            output_ssim = SSIM(x_true_mag, x_mag).cpu()
            psnr_list.append(output_psnr)
            ssim_list.extend(output_ssim.tolist())

    psnrs = []
    for batch in range(len(psnr_list)):
        for image_psnr in psnr_list[batch]:
            psnrs.append(image_psnr.item())

    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssim_list)

    path_folder = 'DIFFPIR/lambda_{}_zeta_{}'.format(cfg.lambda_, cfg.zeta)
    os.makedirs(path_folder, exist_ok=True)

    # Save results
    with open(os.path.join(path_folder, 'psnr_results.txt'), 'w') as f:
        # Parameters
        f.write("Parameters:\n")
        f.write(f"Lambda: {cfg.lambda_}\n")
        f.write(f"Zeta: {cfg.zeta}\n")
        f.write(f"Diffusion Steps: {cfg.diffusion_steps}\n")
        f.write(f"t_start: {cfg.t_start}\n")
        f.write(f"Noise Level (Image): {cfg.noise_level_img}\n")
        f.write(f"Degradation: {cfg.degradation}\n\n")
        # Mean PSNR
        f.write(f"Mean PSNR: {mean_psnr:.4f} dB\n\n")
        # Mean SSIM
        f.write(f"Mean SSIM: {mean_ssim:.4f}\n\n")

    # Plot Reconstruction only
    for index in range(min(5, len(psnrs))):  # Plot up to 5 images
        plt.figure(figsize=(5,5))
        ax = plt.subplot(1,1,1)
        show_image(
            ax,
            x_mag[index],
            f"PSNR {psnrs[index]:.2f}, SSIM {ssim_list[index]:.4f}"
            )
        plt.tight_layout()
        plt.savefig(os.path.join(path_folder, f"test_reconstruction_only_{index}.pdf"), dpi=300)
        plt.close()



# =========================================================
# GRID SEARCH
# =========================================================
def run_grid_search(args, dataloader):
    lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambda).tolist()
    zetas = np.linspace(args.zeta_min, args.zeta_max, args.n_zeta).tolist()

    for lambda_, zeta in product(lambdas, zetas):
        cfg = Config(
            lambda_=lambda_,
            zeta=zeta,
            device=args.device,
            noise_level_img=args.noise_level,
            degradation=args.problem,
        )
        restore(cfg, dataloader)


# =========================================================
# SINGLE TEST
# =========================================================
def run_test(args, dataloader):
    cfg = Config(
        lambda_=args.lambda_,
        zeta=args.zeta,
        device=args.device,
        noise_level_img=args.noise_level,
        degradation=args.problem,
    )
    restore(cfg, dataloader)

def find_best_model(args, dataloader):
    best_psnr = -float('inf')
    best_lambda = None
    best_zeta = None

    lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambda).tolist()
    zetas = np.linspace(args.zeta_min, args.zeta_max, args.n_zeta).tolist()

    for lambda_, zeta in product(lambdas, zetas):
        path_folder = 'DIFFPIR/lambda_{}_zeta_{}'.format(lambda_, zeta)
        psnr_file = os.path.join(path_folder, 'psnr_results.txt')

        if os.path.exists(psnr_file):
            with open(psnr_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Mean PSNR:"):
                        current_psnr = float(line.split(":")[1].strip().split()[0])
                        if current_psnr > best_psnr:
                            best_psnr = current_psnr
                            best_lambda = lambda_
                            best_zeta = zeta
                        break
    
    print(f"Best PSNR: {best_psnr:.4f} dB with Lambda: {best_lambda} and Zeta: {best_zeta}")
    with open(os.path.join('DIFFPIR', 'best_model_results.txt'), 'w') as f:
        f.write(f"Best PSNR: {best_psnr:.4f} dB\n")
        f.write(f"Best Lambda: {best_lambda}\n")
        f.write(f"Best Zeta: {best_zeta}\n")


# =========================================================
# MAIN
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    # -------------------------
    # global setup
    # -------------------------
    parser.add_argument("--problem", choices=["mri", "inpainting", "rician"], required=True)
    parser.add_argument("--mode", choices=["grid", "test"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # -------------------------
    # noise
    # -------------------------
    parser.add_argument("--noise_level", type=float, default=12.75)

    # =====================================================
    # TEST MODE PARAMS
    # =====================================================
    parser.add_argument("--lambda_", type=float, default=10.0)
    parser.add_argument("--zeta", type=float, default=0.5)

    # =====================================================
    # GRID SEARCH PARAMS
    # =====================================================
    parser.add_argument("--lambda_min", type=float, default=0.1)
    parser.add_argument("--lambda_max", type=float, default=25.)
    parser.add_argument("--n_lambda", type=int, default=10)

    parser.add_argument("--zeta_min", type=float, default=0.0)
    parser.add_argument("--zeta_max", type=float, default=1.0)
    parser.add_argument("--n_zeta", type=int, default=10)

    # -------------------------
    args = parser.parse_args()

    # =====================================================
    # reproducibility
    # =====================================================
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # =====================================================
    # dataset selection (simple)
    # =====================================================
    if args.problem == "mri":
        config = {
            "acceleration": 8,
            "img_size": (320, 320),
            "train_path": "DATA/MRI/singlecoil_train",
            "val_path": "DATA/MRI/singlecoil_val",
            "test_path": "DATA/MRI/singlecoil_test",
        }
    else:
        config = {
            "split_ratio": 0.5,
            "img_size": (3, 320, 320),
            "train_path": "DATA/BSDS500/train",
            "val_path": "DATA/BSDS500/val",
            "test_path": "DATA/BSDS500/test",
            "sigma": args.noise_level / 255.0,
        }

    train_loader, val_loader, test_loader, physics = get_dataloaders(
        args.problem, config
    )

    # =====================================================
    # RUN
    # =====================================================
    if args.mode == "grid":
        run_grid_search(args, val_loader)
        find_best_model(args, val_loader)
    else:
        run_test(args, test_loader)


if __name__ == "__main__":
    main()