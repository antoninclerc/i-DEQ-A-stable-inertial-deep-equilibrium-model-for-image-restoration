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
noise = 0.1

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

# ------------------------------------------------------------------------------
# Paramètres (facilement modifiables)
# ------------------------------------------------------------------------------
DC_type = 'grad'
lambda_Rthetas = np.logspace(-1.0, 1.0, num=5).tolist()
Network = 'DRUNet'
mode = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 500
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, 5).tolist()
n_iter_init = [20]


for lambda_Rtheta, sigma_denoiser, n_iter in product(lambda_Rthetas, sigma_denoising, n_iter_init):

    lambda_dc = 0.1
    # ------------------------------------------------------------------------------
    # Dossier pour sauvegarde et paramètres du modèle
    # ------------------------------------------------------------------------------

    path_folder = 'Unrolling_comparison/inpainting/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}'

    os.makedirs(path_folder, exist_ok=True)


    CNNBlock_model = dinv.models.GSDRUNet(
            in_channels=3,
            out_channels=3,
            pretrained='download'
        )

    model = DeepEquilibrium(
        Network=CNNBlock_model,
        problem="Inpainting",
        DC_type=DC_type,
        backtracking=backtracking,
        lambda_dc=lambda_dc,
        lambda_Rtheta=lambda_Rtheta,
        learn_lambda_dc=False,
        learn_lambda_Rtheta=True,
        gamma=0.1,
        eta=0.5,
        thresh=1e-4,
        max_iter=max_iter,
        device=device,
        path_folder=path_folder,
        sigma_noise=noise,
        sigma_denoiser=sigma_denoiser,
        theta_interpol=0.2,
        restart=True,
        B_restart=5000,
        learn_theta_interpol=False,
        learn_B_restart=False
    )

    # ------------------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------------------

    pretrained_path = None
    train = False
    init_train = {"epoch_pretraining" : n_iter, "sigma_pretraining" : 0.2}
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=mode,
            init_train=init_train,
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
            max_epochs=1000,
            plot_interval=1,
            pretrained_path=pretrained_path
        )

    test = True

    if test:
        dict = model.evaluate(
            test_loader=val_dataloader,
            n_display=7,
            accelerated=mode,
            init_train=init_train,
            pretrained_path=pretrained_path,
            PnP=False
        )
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
for lambda_Rtheta, sigma_denoiser, n_iter in product(lambda_Rthetas, sigma_denoising, n_iter_init):
    path_folder = 'Unrolling_comparison/inpainting/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}'
    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test_PSNR:"):
                PSNR = float(line.split(":")[1].strip())
                if PSNR > best_PSNR:
                    best_PSNR = PSNR
                    best_model_path = path_folder

# On écrit le meilleur modèle dans un fichier texte
with open("Unrolling_comparison/inpainting/best_model_grad.txt", "w") as f:
    f.write(f"Best model path: {best_model_path}\n")
    f.write(f"Best PSNR: {best_PSNR}\n")

# ------------------------------------------------------------------------------
# Paramètres (facilement modifiables)
# ------------------------------------------------------------------------------
DC_type = 'prox'
lambda_Rthetas = np.linspace(1., 10., num=5).tolist()
Network = 'DRUNet'
mode = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 200
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, 5).tolist()
n_iter_init = [20]


for lambda_Rtheta, sigma_denoiser, n_iter in product(lambda_Rthetas, sigma_denoising, n_iter_init):

    lambda_dc = 5.
    # ------------------------------------------------------------------------------
    # Dossier pour sauvegarde et paramètres du modèle
    # ------------------------------------------------------------------------------

    path_folder = 'Unrolling_comparison/inpainting/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}'

    os.makedirs(path_folder, exist_ok=True)


    CNNBlock_model = dinv.models.GSDRUNet(
            in_channels=3,
            out_channels=3,
            pretrained='download'
        )

    model = DeepEquilibrium(
        Network=CNNBlock_model,
        problem="Inpainting",
        DC_type=DC_type,
        backtracking=backtracking,
        lambda_dc=lambda_dc,
        lambda_Rtheta=lambda_Rtheta,
        learn_lambda_dc=False,
        learn_lambda_Rtheta=True,
        gamma=0.1,
        eta=0.5,
        thresh=1e-4,
        max_iter=max_iter,
        device=device,
        path_folder=path_folder,
        sigma_noise=noise,
        sigma_denoiser=sigma_denoiser,
        theta_interpol=0.2,
        restart=True,
        B_restart=5000,
        learn_theta_interpol=False,
        learn_B_restart=False
    )

    # ------------------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------------------

    pretrained_path = None
    train = False
    init_train = {"epoch_pretraining" : n_iter, "sigma_pretraining" : 0.2}
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=mode,
            init_train=init_train,
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
            max_epochs=1000,
            plot_interval=1,
            pretrained_path=pretrained_path
        )

    test = True

    if test:
        dict = model.evaluate(
            test_loader=val_dataloader,
            n_display=7,
            accelerated=mode,
            init_train=init_train,
            pretrained_path=pretrained_path,
            PnP=False
        )
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
for lambda_Rtheta, sigma_denoiser, n_iter in product(lambda_Rthetas, sigma_denoising, n_iter_init):
    path_folder = 'Unrolling_comparison/inpainting/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_n_iter_init_{n_iter}'
    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test_PSNR:"):
                PSNR = float(line.split(":")[1].strip())
                if PSNR > best_PSNR:
                    best_PSNR = PSNR
                    best_model_path = path_folder

# On écrit le meilleur modèle dans un fichier texte
with open("Unrolling_comparison/inpainting/best_model_prox.txt", "w") as f:
    f.write(f"Best model path: {best_model_path}\n")
    f.write(f"Best PSNR: {best_PSNR}\n")
