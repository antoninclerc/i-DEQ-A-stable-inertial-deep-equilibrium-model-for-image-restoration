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
acceleration = 4
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
    overwrite_existing=False,
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

DC_type = 'grad'
lambda_Rthetas = np.logspace(-1.0, 1.0, num=5).tolist() # 10 valeurs de lambda_Rtheta entre 0.1 et 10
Network = 'DRUNet'
accelerated = True
max_iter = 200
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, 5).tolist()
lambda_dcs = [0.5]
init_trains = [True, False]

for lambda_Rtheta, sigma_denoiser, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, lambda_dcs, init_trains):
    
    # -------------------------------------------------------------------------------
    # Dossier pour sauvegarde et paramètres du modèle
    # -------------------------------------------------------------------------------
    
    path_folder = "Unrolling_comparison/MRI/acceleration_{}/lambda_Rtheta_{:.2f}_sigma_{:.3f}_lambda_dc_{:.2f}".format(
        acceleration, lambda_Rtheta, sigma_denoiser, lambda_dc)
    os.makedirs(path_folder, exist_ok=True)
    
    if Network == 'DRUNet':
        CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')

    model = DeepEquilibrium(
                Network=CNNBlock_model, 
                problem="MRI",
                DC_type=DC_type,
                backtracking=backtracking, 
                lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
                learn_lambda_dc=False, learn_lambda_Rtheta=True,
                gamma=0.1, eta=0.5,
                thresh=1e-4, max_iter=max_iter, 
                device=device, path_folder=path_folder, 
                sigma_noise=0.1,
                sigma_denoiser=sigma_denoiser,
                theta_interpol=0.2,
                restart=True,
                B_restart=5000,
                learn_theta_interpol=False,
                learn_B_restart=False)

    # -------------------------------------------------------------------------------
    # Entraînement
    # -------------------------------------------------------------------------------
    if init_train:
        init_train_params = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2}
    else:
        init_train_params = None

    pretrained_path = None
    train = False
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=accelerated,
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

    test = True
    if test:
        dict = model.evaluate(test_loader=val_dataloader, n_display=7, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

    test_mse = dict["test_mse"]
    test_PSNR = dict["test_PSNR"]
    input_mse = dict["input_mse"]
    input_PSNR = dict["input_PSNR"]

    # On sauvegarde les paramètres et les résultats dans un fichier texte
    with open(os.path.join(path_folder, "results.txt"), "w") as f:
        f.write(f"DC_type: {DC_type}\n")
        f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
        f.write(f"sigma_denoiser: {sigma_denoiser}\n")
        f.write(f"test_mse: {test_mse}\n")
        f.write(f"test_PSNR: {test_PSNR}\n")
        f.write(f"input_mse: {input_mse}\n")
        f.write(f"input_PSNR: {input_PSNR}\n")

# recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
DC_type = 'grad'
best_PSNR = -float('inf')
best_model_path = None
for lambda_Rtheta, sigma_denoiser, n_iter, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, lambda_dcs, init_trains):
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
lambda_Rthetas = np.linspace(-1., 1., num=5).tolist()
Network = 'DRUNet'
accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 200
backtracking = False
sigma_denoising = np.linspace(0.01, 0.1, 5).tolist()
lambda_dc = [1.0]
init_trains = [True, False]

for lambda_Rtheta, sigma_denoiser, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, lambda_dc, init_trains):
    # -------------------------------------------------------------------------------
    # Dossier pour sauvegarde et paramètres du modèle
    # -------------------------------------------------------------------------------
    
    path_folder = "Unrolling_comparison/MRI/acceleration_{}/lambda_Rtheta_{:.2f}_sigma_{:.3f}_lambda_dc_{:.2f}".format(
        acceleration, lambda_Rtheta, sigma_denoiser, lambda_dc)
    os.makedirs(path_folder, exist_ok=True)
    
    if Network == 'DRUNet':
        CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')

    model = DeepEquilibrium(
                Network=CNNBlock_model, 
                problem="MRI",
                DC_type=DC_type,
                backtracking=backtracking, 
                lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
                learn_lambda_dc=False, learn_lambda_Rtheta=True,
                gamma=0.1, eta=0.5,
                thresh=1e-4, max_iter=max_iter, 
                device=device, path_folder=path_folder, 
                sigma_noise=0.1,
                sigma_denoiser=sigma_denoiser,
                theta_interpol=0.2,
                restart=True,
                B_restart=5000,
                learn_theta_interpol=False,
                learn_B_restart=False)
    
    # -------------------------------------------------------------------------------
    # Entraînement
    # -------------------------------------------------------------------------------
    if init_train:
        init_train_params = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2}
    else:
        init_train_params = None
    
    pretrained_path = None
    train = False
    if train:
        model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=accelerated,
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
        
    test = True
    if test:
        dict = model.evaluate(test_loader=val_dataloader, n_display=7, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

    test_mse = dict["test_mse"]
    test_PSNR = dict["test_PSNR"]
    input_mse = dict["input_mse"]
    input_PSNR = dict["input_PSNR"]

    # On sauvegarde les paramètres et les résultats dans un fichier texte
    with open(os.path.join(path_folder, "results.txt"), "w") as f:
            f.write(f"DC_type: {DC_type}\n")
            f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
            f.write(f"sigma_denoiser: {sigma_denoiser}\n")
            f.write(f"lambda_dc: {lambda_dc}\n")
            f.write(f"test_mse: {test_mse}\n")
            f.write(f"test_PSNR: {test_PSNR}\n")
            f.write(f"input_mse: {input_mse}\n")
            f.write(f"input_PSNR: {input_PSNR}\n")
    
# recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
DC_type = 'prox'
best_PSNR = -float('inf')
best_model_path = None
for lambda_Rtheta, sigma_denoiser, lambda_dc, init_train in product(lambda_Rthetas, sigma_denoising, lambda_dc, init_trains):
    path_folder = 'Unrolling_comparison/MRI/' + f'DC_{DC_type}_lambda_Rtheta_{lambda_Rtheta}_sigma_denoiser_{sigma_denoiser}_lambda_dc_{lambda_dc}_init_train_{init_train}'
    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test_PSNR:"):
                PSNR = float(line.split(":")[1].strip())
                if PSNR > best_PSNR:
                    best_PSNR = PSNR
                    best_model_path = path_folder