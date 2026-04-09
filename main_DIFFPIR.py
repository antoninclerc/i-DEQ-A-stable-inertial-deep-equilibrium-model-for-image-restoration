import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Force PyTorch to use a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import deepinv as dinv

from models.deep_equilibrium import DeepEquilibrium
from networks.DRUnet import GSDRUNet
from itertools import product

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

# ---------------------------
# Dataset parameters
# ---------------------------
img_size = (3, 320, 320)
split_ratio = 0.5

transform = transforms.ToTensor()

# ---------------------------
# Load images from folder
# ---------------------------
def load_folder_as_tensor(root):
    """
    Load all images from a directory and return a stacked tensor.
    """
    root = Path(root)
    images = []

    for path in sorted(root.glob("*")):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)

    return torch.stack(images)


# ---------------------------
# Load datasets
# ---------------------------
train_tensor = load_folder_as_tensor("DATA/BSDS500/train")
val_tensor = load_folder_as_tensor("DATA/BSDS500/val")
test_tensor = load_folder_as_tensor("DATA/BSDS500/test")

# Dataset for deepinv (x,y)
train_subset = TensorDataset(train_tensor, train_tensor)
val_subset = TensorDataset(val_tensor, val_tensor)
test_subset = TensorDataset(test_tensor, test_tensor)

# ---------------------------
# Physics (inpainting)
# ---------------------------
physics_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    img_size=img_size,
    split_ratio=split_ratio,
    device=device,
    rng=rng,
)

mask = physics_generator.step()["mask"]

physics = dinv.physics.Inpainting(
    img_size=img_size,
    mask=mask,
    device=device,
)

# ---------------------------
# Generate dataset
# ---------------------------
dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_subset,
    test_dataset=test_subset,
    val_dataset=val_subset,
    physics=physics,
    physics_generator=physics_generator,
    save_physics_generator_params=True,
    overwrite_existing=False,
    device=device,
    save_dir=f"datasets/{split_ratio}",
    batch_size=4,
)

# ---------------------------
# Load HDF5 datasets
# ---------------------------
train_dataset = dinv.datasets.HDF5Dataset(
    dataset_path,
    split="train",
    load_physics_generator_params=True,
)

val_dataset = dinv.datasets.HDF5Dataset(
    dataset_path,
    split="val",
    load_physics_generator_params=True,
)

test_dataset = dinv.datasets.HDF5Dataset(
    dataset_path,
    split="test",
    load_physics_generator_params=True,
)

# ---------------------------
# DataLoaders
# ---------------------------
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

sigmas = np.linspace(0.01, 0.1, num=5).tolist()
max_iter = 200
zeta = np.linspace(0.1, 0.5, num=5).tolist() # 10 valeurs de zeta entre 0.1 et 1.0
lambdas = np.linspace(3., 25., num=5).tolist() # 10 valeurs de lambda entre 3.0 et 25.0

for sigma, zeta, lambda_ in product(sigmas, zeta, lambdas):
    path_folder = 'Unrolling_comparison/inpainting/DiffPIR/' + f'sigma_{sigma}_zeta_{zeta}_lambda_{lambda_}'

    model = dinv.sampling.DiffPIR(
        dinv.models.DiffUNet(),
        data_fidelity=dinv.optim.data_fidelity.L2(),
        device=device,
        sigma=sigma,
        zeta=zeta,
        lambda_=lambda_,
        max_iter=max_iter,
    ).to(device)

    for i, (xtrue, _, _) in enumerate(val_dataloader):

        xtrue = xtrue.to(device)

        # Forward MRI operator
        y = physics(xtrue)

        # Reconstruction
        x_rec = model(y, physics)

        # Zero-filled reconstruction (for visualization of measurements)
        x_zf = physics.A_adjoint(y)

        xtrue = torch.clamp(xtrue, 0, 1)
        x_rec = torch.clamp(x_rec, 0, 1)
        x_zf = torch.clamp(x_zf, 0, 1)

        MSE_batch = torch.mean((xtrue - x_rec) ** 2, dim=[1, 2, 3])
        PSNR_batch = 10 * torch.log10(1 / MSE_batch)
        print(f"Batch {i}: PSNR = {PSNR_batch.mean():.2f} dB")
        
        for idx in range(min(xtrue.shape[0], 7)):
            xtrue_np = xtrue[idx].detach().cpu().permute(1, 2, 0).numpy()
            xrec_np  = x_rec[idx].detach().cpu().permute(1, 2, 0).numpy()
            xzf_np   = x_zf[idx].detach().cpu().permute(1, 2, 0).numpy()
            err_np   = np.abs(xtrue_np - xrec_np)

            mse = np.mean((xtrue_np - xrec_np) ** 2)
            mse0 = np.mean((xtrue_np - xzf_np) ** 2)
            psnr = 10 * np.log10(1 / mse)
            psnr0 = 10 * np.log10(1 / mse0)
            print(f"Image {i}: PSNR = {psnr:.2f} dB (ZF: {psnr0:.2f} dB)")

            plt.figure(figsize=(16,4))

            plt.subplot(1,4,1)
            plt.imshow(xtrue_np)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1,4,2)
            plt.imshow(xzf_np)
            plt.title("Zero-filled,\nPSNR: {:.2f} dB".format(psnr0))
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.imshow(xrec_np)
            plt.title("Reconstruction,\nPSNR: {:.2f} dB".format(psnr))
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.imshow(err_np, cmap="hot")
            plt.title("Error")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"diffpir_reconstruction_{i}.png")
            plt.close()

        break

    # On sauvegarde les paramètres du modèle et les résultats
    with open(os.path.join(path_folder, "results.txt"), "w") as f:
        f.write(f"Sigma: {sigma}\n")
        f.write(f"Zeta: {zeta}\n")
        f.write(f"Lambda: {lambda_}\n")
        f.write(f"Max Iterations: {max_iter}\n")
        f.write(f"PSNR: {PSNR_batch.mean():.2f} dB\n")
    
    break

# Recherche du meilleur modèle (en fonction du PSNR)
best_psnr = -float("inf")
best_params = None
best_model_path = None
for sigma, zeta, lambda_ in product(sigmas, zeta, lambdas):
    path_folder = 'Unrolling_comparison/inpainting/DiffPIR/' + f'sigma_{sigma}_zeta_{zeta}_lambda_{lambda_}'
    results_path = os.path.join(path_folder, "results.txt")
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            lines = f.readlines()
            psnr_line = [line for line in lines if line.startswith("PSNR:")][0]
            psnr_value = float(psnr_line.split(":")[1].strip().split()[0])
            
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_params = (sigma, zeta, lambda_)
                best_model_path = path_folder

# On écrit le meilleur modèle dans un fichier texte
with open("best_model.txt", "w") as f:
    f.write(f"Best PSNR: {best_psnr:.2f} dB\n")
    f.write(f"Best Parameters:\n")
    f.write(f"Sigma: {best_params[0]}\n")
    f.write(f"Zeta: {best_params[1]}\n")
    f.write(f"Lambda: {best_params[2]}\n")
    f.write(f"Model Path: {best_model_path}\n")
