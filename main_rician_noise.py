import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

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

# ---------------------------
# Reproducibility
# ---------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

rng = torch.Generator(device=device).manual_seed(seed)

# ---------------------------
# Dataset parameters
# ---------------------------
img_size = (3, 320, 320)
sigma_noise = 0.1

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
            img = transform(img)  # [C,H,W], values 0-1
            images.append(img)

    return torch.stack(images)

# ---------------------------
# Add Rician noise
# ---------------------------
def add_rician_noise(x, sigma):
    """
    Add multiplicative Rician noise to a batch of images (tensor [B,C,H,W]).
    """
    noise_real = torch.randn_like(x) * sigma
    noise_imag = torch.randn_like(x) * sigma
    noisy = torch.sqrt((x + noise_real) ** 2 + noise_imag ** 2)
    return noisy

# ---------------------------
# Rician Dataset
# ---------------------------
class RicianDataset(TensorDataset):
    def __init__(self, clean_tensor, sigma):
        super().__init__(clean_tensor, clean_tensor)
        self.sigma = sigma

    def __getitem__(self, idx):
        target = self.tensors[0][idx]
        input_ = add_rician_noise(target, self.sigma)
        mask = {'mask': torch.ones_like(target[0:1])}  # Example mask, single channel
        return target, input_, mask

# ---------------------------
# Load datasets
# ---------------------------
train_tensor = load_folder_as_tensor("DATA/BSDS500/train")
val_tensor = load_folder_as_tensor("DATA/BSDS500/val")
test_tensor = load_folder_as_tensor("DATA/BSDS500/test")

train_dataset = RicianDataset(train_tensor, sigma=sigma_noise)
val_dataset = RicianDataset(val_tensor, sigma=sigma_noise)
test_dataset = RicianDataset(test_tensor, sigma=sigma_noise)

# ---------------------------
# DataLoaders
# ---------------------------
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# ---------------------------
# Test print
# ---------------------------
for target, input_, mask in train_dataloader:
    print("Batch target:", target.shape)
    print("Batch input:", input_.shape)
    print("Batch mask:", mask['mask'].shape)
    break

# ------------------------------------------------------------------------------
# Paramètres (facilement modifiables)
# ------------------------------------------------------------------------------
DC_type = 'grad'
lambda_Rthetas = np.logspace(-1.0, 1.0, num=10).tolist()
Network = 'DRUNet'
accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 200
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, num=10).tolist()
n_iter_init = [20]
lambda_dcs = [0.03]
init_trains = [True, False]

for lambda_Rtheta, sigma_denoiser, n_iter, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, n_iter_init, lambda_dcs, init_trains):

    # -------------------------------------------------------------------------------
    # Folder for saving model and parameters
    # -------------------------------------------------------------------------------
    path_folder = 'Unrolling_comparison/rician/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}_lambda_dc_{lambda_dc}_init_train_{init_train}'
    os.makedirs(path_folder, exist_ok=True)

    # ---------------------------
    # Model definition
    # ---------------------------
    CNNBlock_model = dinv.models.GSDRUNet(in_channels=3, out_channels=3, pretrained='download')

    model = DeepEquilibrium(
        Network=CNNBlock_model, 
        problem="rician",
        DC_type=DC_type,
        backtracking=backtracking, 
        lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
        learn_lambda_dc=False, learn_lambda_Rtheta=False,
        gamma=0., eta=0.5,
        thresh=1e-10, max_iter=max_iter, 
        device=device, path_folder=path_folder, 
        sigma_noise=0.1,
        sigma_denoiser=sigma_denoiser,
        theta_interpol=0.01,
        restart=True,
        B_restart=100,
        learn_theta_interpol=False,
        learn_B_restart=False)
    
    if init_train:
        init_train_params = {"epoch_pretraining" : n_iter, "sigma_pretraining" : 0.2}
    else:
        init_train_params = None
    # -------------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------------
    pretrained_path = None
    train = False
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=accelerated,
            init_train=init_train_params,
            JFB=True,
            K_JFB=3,
            lr=1e-5,
            eta_k=None,
            eta_TV=None,
            eta_l1=None,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={"betas": (0.9, 0.999)},
            scheduler=None,
            scheduler_kwargs=None,
            max_patience=25,
            max_epochs=1,
            plot_interval=1,
            pretrained_path=pretrained_path)

    # -------------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------------
    test = True
    if test:
        dict = model.evaluate(test_loader=val_dataloader, n_display=2, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

    test_mse = dict["test_mse"]
    test_PSNR = dict["test_PSNR"]
    input_mse = dict["input_mse"]
    input_PSNR = dict["input_PSNR"]

    # On sauvegarde les paramètres et les résultats dans un fichier texte
    with open(os.path.join(path_folder, "results.txt"), "w") as f:
        f.write(f"DC_type: {DC_type}\n")
        f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
        f.write(f"sigma_denoiser: {sigma_denoiser}\n")
        f.write(f"n_iter_init: {n_iter}\n")
        f.write(f"test_mse: {test_mse}\n")
        f.write(f"test_PSNR: {test_PSNR}\n")
        f.write(f"input_mse: {input_mse}\n")
        f.write(f"input_PSNR: {input_PSNR}\n")

# recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
DC_type = 'grad'
best_PSNR = -float('inf')
best_model_path = None
for lambda_Rtheta, sigma_denoiser, n_iter, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, n_iter_init, lambda_dcs, init_trains):
    path_folder = 'Unrolling_comparison/rician/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}_lambda_dc_{lambda_dc}_init_train_{init_train}'
    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test_PSNR:"):
                PSNR = float(line.split(":")[1].strip())
                if PSNR > best_PSNR:
                    best_PSNR = PSNR
                    best_model_path = path_folder

# On écrit le meilleur modèle dans un fichier texte
with open("Unrolling_comparison/rician/best_model_grad.txt", "w") as f:
    f.write(f"Best model path: {best_model_path}\n")
    f.write(f"Best PSNR: {best_PSNR}\n")

# ------------------------------------------------------------------------------
# Paramètres (facilement modifiables)
# ------------------------------------------------------------------------------
DC_type = 'prox'
lambda_Rthetas = np.linspace(1., 10., num=10).tolist()
Network = 'DRUNet'
accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 200
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, num=10).tolist()
n_iter_init = [20]
lambda_dcs = [5e-4]
init_trains = [True, False]

for lambda_Rtheta, sigma_denoiser, n_iter, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, n_iter_init, lambda_dcs, init_trains):
    
    # -------------------------------------------------------------------------------
    # Folder for saving model and parameters
    # -------------------------------------------------------------------------------
    path_folder = 'Unrolling_comparison/rician/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}_lambda_dc_{lambda_dc}_init_train_{init_train}'
    os.makedirs(path_folder, exist_ok=True)

    # ---------------------------
    # Model definition
    # ---------------------------
    CNNBlock_model = dinv.models.GSDRUNet(in_channels=3, out_channels=3, pretrained='download')

    model = DeepEquilibrium(
        Network=CNNBlock_model, 
        problem="rician",
        DC_type=DC_type,
        backtracking=backtracking, 
        lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
        learn_lambda_dc=False, learn_lambda_Rtheta=False,
        gamma=0., eta=0.5,
        thresh=1e-10, max_iter=max_iter, 
        device=device, path_folder=path_folder, 
        sigma_noise=sigma_noise,
        theta_interpol=0.01,
        restart=True,
        B_restart=100,
        learn_theta_interpol=False,
        learn_B_restart=False)
    if init_train:
        init_train_params = {"epoch_pretraining" : n_iter, "sigma_pretraining" : 0.2}
    else:
        init_train_params = None
    # -------------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------------
    pretrained_path = None
    train = False
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=accelerated,
            init_train=init_train_params,
            JFB=True,
            K_JFB=3,
            lr=1e-5,
            eta_k=None,
            eta_TV=None,
            eta_l1=None,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={"betas": (0.9, 0.999)},
            scheduler=None,
            scheduler_kwargs=None,
            max_patience=25,
            max_epochs=1,
            plot_interval=1,
            pretrained_path=pretrained_path)

    # -------------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------------
    test = True
    if test:
        dict = model.evaluate(test_loader=val_dataloader, n_display=2, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

    test_mse = dict["test_mse"]
    test_PSNR = dict["test_PSNR"]
    input_mse = dict["input_mse"]
    input_PSNR = dict["input_PSNR"]

    # On sauvegarde les paramètres et les résultats dans un fichier texte
    with open(os.path.join(path_folder, "results.txt"), "w") as f:
        f.write(f"DC_type: {DC_type}\n")
        f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
        f.write(f"sigma_denoiser: {sigma_denoiser}\n")
        f.write(f"n_iter_init: {n_iter}\n")
        f.write(f"test_mse: {test_mse}\n")
        f.write(f"test_PSNR: {test_PSNR}\n")
        f.write(f"input_mse: {input_mse}\n")
        f.write(f"input_PSNR: {input_PSNR}\n")

# recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
DC_type = 'prox'
best_PSNR = -float('inf')
best_model_path = None
for lambda_Rtheta, sigma_denoiser, n_iter, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, n_iter_init, lambda_dcs, init_trains):
    path_folder = 'Unrolling_comparison/rician/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}_lambda_dc_{lambda_dc}_init_train_{init_train}'
    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test_PSNR:"):
                PSNR = float(line.split(":")[1].strip())
                if PSNR > best_PSNR:
                    best_PSNR = PSNR
                    best_model_path = path_folder

# On écrit le meilleur modèle dans un fichier texte
with open("Unrolling_comparison/rician/best_model_prox.txt", "w") as f:
    f.write(f"Best model path: {best_model_path}\n")
    f.write(f"Best PSNR: {best_PSNR}\n")