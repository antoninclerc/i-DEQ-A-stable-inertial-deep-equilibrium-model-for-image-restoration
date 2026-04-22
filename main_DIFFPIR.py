import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import add_zero_channel, PSNR, ifft2c, image_2ch_to_magnitude, SSIM
from models.data_consistency import DC_prox_MRI

import deepinv as dinv

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
        degradation="MRI",
        device="cuda:0",
    ):
        self.lambda_ = lambda_
        self.zeta = zeta
        self.diffusion_steps = diffusion_steps
        self.t_start = t_start
        self.device = device
        self.noise_level_img = noise_level_img
        self.degradation = degradation
        if degradation == "MRI":
            self.prox = DC_prox_MRI

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

    model = dinv.models.GSDRUNet(
        in_channels=1,
        out_channels=1,
        pretrained="networks/GSDRUNet_grayscale_torch.ckpt",
    ).to(cfg.device)


    def denoiser(x, sigma):
        x_in = x[:, 0, :, :].unsqueeze(1)
        x_out = model(x_in, sigma)
        x_out = add_zero_channel(x_out)
        return x_out

    psnr_list = []
    ssim_list = []
    for x_true, y, params in dataloader:

        x_true = x_true.to(cfg.device)
        y = y.to(cfg.device)
        mask = params['mask'].to(cfg.device)

        y = y + cfg.noise_level_img / 255.0 * torch.randn_like(y)

        if cfg.degradation == "MRI":
            y = y * mask
        
        T = 1000
        alphas = get_alphas(T, cfg.device)
        sigmas = torch.sqrt(1.0 - alphas) / torch.sqrt(alphas)

        rhos = cfg.lambda_ * (cfg.noise_level_img / 255.0) ** 2 / (sigmas**2)

        seq = np.sqrt(np.linspace(0, cfg.t_start**2, cfg.diffusion_steps))
        seq = [int(s) for s in seq]
        seq[-1] -= 1

        if cfg.degradation == "MRI":
            x0 = ifft2c(y)
            x0 = torch.clamp(x0, 0, 1)
        else:
            x0 = y

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

                x0 = 2 * denoiser((x + 1) / 2, curr_sigma) - 1

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
            x_mag = torch.clamp(x, 0, 1)
            x_mag = image_2ch_to_magnitude(x)
            x_true_mag = image_2ch_to_magnitude(x_true)
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


    # Plot Images
    for index in range(min(5, len(psnrs))):  # Plot up to 5 images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(x_true_mag[index].cpu(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(x_mag[index].cpu(), cmap='gray')
        output_psnr = psnrs[index]
        output_ssim = ssim_list[index]
        plt.title(f'Restored Image\nPSNR: {output_psnr:.2f} dB, SSIM: {output_ssim:.4f}')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        input = ifft2c(y)
        input = torch.clamp(input, 0, 1)
        input = image_2ch_to_magnitude(input)[index]
        plt.imshow(input.cpu(), cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path_folder, f'restored_image_{index}.png'))
        plt.close()



# =========================
# Entry point
# =========================
if __name__ == "__main__":
    import os
    import random
    import numpy as np

    # Forcer PyTorch à n'utiliser qu'un seul GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # seul le GPU 0 sera visible
    device = "cuda:1"

    import deepinv as dinv
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from deepinv.datasets import FastMRISliceDataset
    from itertools import product
    from models.deep_equilibrium import DeepEquilibrium
    from networks.DRUnet import GSDRUNet

    # Générateur pour reproductibilité
    seed = 42
    torch.manual_seed(seed)              # CPU
    torch.cuda.manual_seed(seed)         # GPU courant
    torch.cuda.manual_seed_all(seed)     # Toutes les GPUs visibles
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)                    # Python random
    np.random.seed(seed)                 # NumPy

    # Générateur PyTorch pour usage explicite si besoin
    rng = torch.Generator(device=device).manual_seed(seed)

    # ------------------------------------------------------------
    # Construction des DataLoaders avec Data_Transform et filtrage
    # ------------------------------------------------------------
    acceleration = 8
    img_size = (320, 320)
    root = "DATA/singlecoil_train"
    dataset = FastMRISliceDataset(root=root, slice_index="middle")
    train_subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt", pad_to_size=img_size)

    root = "DATA/singlecoil_val"
    dataset = FastMRISliceDataset(root=root, slice_index="middle")
    val_subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt", pad_to_size=img_size)

    root = "DATA/singlecoil_test"
    dataset = FastMRISliceDataset(root=root, slice_index="middle")
    test_subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt", pad_to_size=img_size)

    physics_generator = dinv.physics.generator.GaussianMaskGenerator(
        img_size=img_size, acceleration=acceleration, rng=rng, device=device
    )
    mask = physics_generator.step()["mask"]

    physics = dinv.physics.MRI(mask=mask, img_size=img_size, device=device)

    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_subset,
        test_dataset=test_subset,
        val_dataset=val_subset,
        physics=physics,
        physics_generator=physics_generator,
        save_physics_generator_params=True,
        overwrite_existing=True,
        device=device,
        save_dir='datasets/acceleration_{}'.format(acceleration),
        batch_size=4,
    )

    train_dataset = dinv.datasets.HDF5Dataset(
        dataset_path, split="train", load_physics_generator_params=True
    )
    test_dataset = dinv.datasets.HDF5Dataset(
        dataset_path, split="test", load_physics_generator_params=True
    )
    val_dataset = dinv.datasets.HDF5Dataset(
        dataset_path, split="val", load_physics_generator_params=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # zetas = (np.linspace(0, 1, 10)).tolist()
    # lambdas = (np.linspace(3., 25., 10)).tolist()
    # for lambda_, zeta in product(lambdas, zetas):
    #     cfg = Config(lambda_=lambda_, zeta=zeta, device=device)
    #     restore(cfg, val_dataloader)
    
    # # Recherche du meilleur couple (lambda, zeta) sur la validation
    # # avec les résultats sauvegardés dans 'DIFFPIR/lambda_{}/zeta_{}/psnr_results.txt'. Le meilleur couple est celui qui maximise le PSNR moyen sur la validation. Ensuite, on peut utiliser ce couple pour restaurer les images de test et évaluer les performances finales.
    # max_psnr = -float('inf')
    # best_lambda = None
    # best_zeta = None
    # best_path = None
    # for lambda_, zeta in product(lambdas, zetas):
    #     path_folder = 'DIFFPIR/lambda_{}_zeta_{}'.format(lambda_, zeta)
    #     with open(os.path.join(path_folder, 'psnr_results.txt'), 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             if line.startswith("Mean PSNR:"):
    #                 mean_psnr = float(line.split(":")[1].strip().split()[0])
    #                 print(f"Lambda: {lambda_}, Zeta: {zeta}, Mean PSNR: {mean_psnr:.2f} dB")
    #                 if mean_psnr > max_psnr:
    #                     max_psnr = mean_psnr
    #                     best_lambda = lambda_
    #                     best_zeta = zeta
    #                     best_path = path_folder
    # print(f"Best Lambda: {best_lambda}, Best Zeta: {best_zeta}, Best Mean PSNR: {max_psnr:.2f} dB, Path: {best_path}")
    # # On écrit les résultats finaux sur le test avec le meilleur couple (lambda, zeta)
    # with open(os.path.join('DIFFPIR/', 'best_results.txt'), 'w') as f:
    #     f.write(f"Best Lambda: {best_lambda}\n")
    #     f.write(f"Best Zeta: {best_zeta}\n")
    #     f.write(f"Best Mean PSNR on Validation: {max_psnr:.2f} dB\n")
    #     f.write("Test results with best parameters:\n")

    zeta = 0.55
    lambda_ = 3.0
    cfg = Config(lambda_=lambda_, zeta=zeta, device=device)
    restore(cfg, test_dataloader)

