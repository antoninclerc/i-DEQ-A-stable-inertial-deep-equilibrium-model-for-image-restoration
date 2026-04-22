import os
import random
import numpy as np

# Forcer PyTorch à n'utiliser qu'un seul GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # seul le GPU 0 sera visible
device = "cuda:0"

import deepinv as dinv
import torch
import torchvision
from torch.utils.data import DataLoader
from deepinv.datasets import FastMRISliceDataset
from itertools import product
from models.Unrolling import Unrolling
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

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

DC = 'grad'
mode = 'Varnet'
max_iter = 5
noise = 1./255

lambda_dc = 0.5

# -------------------------------------------------------------------------------
# Dossier pour sauvegarde et paramètres du modèle
# -------------------------------------------------------------------------------

path_folder = 'Unrolling_comparison/Unrolling/{}/{}/DC_{}_lambdadc_{}_noise_{:.2f}'.format(
    mode, acceleration, DC, lambda_dc, noise
)
os.makedirs(path_folder, exist_ok=True)

CNNBlock_model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None)

model = Unrolling(
    Network=CNNBlock_model, 
    problem="MRI",
    DC_type=DC,
    lambda_dc=lambda_dc,
    learn_lambda_dc=True,
    max_iter=max_iter, 
    device=device,
    path_folder=path_folder, 
    sigma_noise=noise,
    use_noise=True,
    sigma_denoiser=noise
)

# -------------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------------

pretrained_path = None
train = False

if train:
    model.train_model(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        mode=mode,
        lr=1e-4,
        eta_k=None,
        eta_TV=None,
        eta_l1=None,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.9, 0.999)},
        scheduler=None,
        scheduler_kwargs=None,
        max_patience=25,
        max_epochs=500,
        plot_interval=1,
        pretrained_path=pretrained_path
    )

test = True
pretrained_path = path_folder + "/best_model.pth"
if test:
    dict_VarNet = model.evaluate(
        test_loader=test_dataloader,
        mode=mode,
        n_display=7,
        pretrained_path=pretrained_path
    )

test_mse = dict_VarNet['test_mse']
test_psnr = dict_VarNet['test_PSNR']
input_mse = dict_VarNet['input_mse']
input_psnr = dict_VarNet['input_PSNR']
input_ssim = dict_VarNet['input_SSIM']
test_ssim = dict_VarNet['test_SSIM']

# Sauvegarde des résultats dans un fichier texte
results_path = os.path.join(path_folder, "results.txt")
with open(results_path, "w") as f:
    f.write(f"Test MSE: {test_mse:.6f}\n")
    f.write(f"Test PSNR: {test_psnr:.2f} dB\n")
    f.write(f"Input MSE: {input_mse:.6f}\n")
    f.write(f"Input PSNR: {input_psnr:.2f} dB\n")
    f.write(f"Input SSIM: {input_ssim:.4f}\n")
    f.write(f"Test SSIM: {test_ssim:.4f}\n")

DC = 'prox'
mode = 'MoDL'
max_iter = 5
noise = 1./255
lambda_dc = 1.

# -------------------------------------------------------------------------------
# Dossier pour sauvegarde et paramètres du modèle
# -------------------------------------------------------------------------------

path_folder = 'Unrolling_comparison/Unrolling/{}/{}/DC_{}_lambdadc_{}_noise_{:.2f}'.format(
    mode, acceleration, DC, lambda_dc, noise
)
os.makedirs(path_folder, exist_ok=True)

CNNBlock_model = dinv.models.DRUNet(in_channels=1, out_channels=1)

model = Unrolling(
    Network=CNNBlock_model, 
    problem="MRI",
    DC_type=DC,
    lambda_dc=lambda_dc,
    learn_lambda_dc=True,
    max_iter=max_iter, 
    device=device,
    path_folder=path_folder, 
    sigma_noise=noise,
    use_noise=True,
    sigma_denoiser=noise
)

# -------------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------------

pretrained_path = None
train = False

if train:
    model.train_model(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        mode=mode,
        lr=1e-4,
        eta_k=None,
        eta_TV=None,
        eta_l1=None,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.9, 0.999)},
        scheduler=None,
        scheduler_kwargs=None,
        max_patience=25,
        max_epochs=500,
        plot_interval=1,
        pretrained_path=pretrained_path
    )

test = True
pretrained_path = path_folder + "/best_model.pth"
if test:
    dict_MoDL = model.evaluate(
        test_loader=test_dataloader,
        mode=mode,
        n_display=7,
        pretrained_path=pretrained_path
    )

test_mse = dict_MoDL['test_mse']
test_psnr = dict_MoDL['test_PSNR']
input_mse = dict_MoDL['input_mse']
input_psnr = dict_MoDL['input_PSNR']
input_ssim = dict_MoDL['input_SSIM']
test_ssim = dict_MoDL['test_SSIM']

# Sauvegarde des résultats dans un fichier texte
results_path = os.path.join(path_folder, "results.txt")
with open(results_path, "w") as f:
    f.write(f"Test MSE: {test_mse:.6f}\n")
    f.write(f"Test PSNR: {test_psnr:.2f} dB\n")
    f.write(f"Input MSE: {input_mse:.6f}\n")
    f.write(f"Input PSNR: {input_psnr:.2f} dB\n")
    f.write(f"Input SSIM: {input_ssim:.4f}\n")
    f.write(f"Test SSIM: {test_ssim:.4f}\n")