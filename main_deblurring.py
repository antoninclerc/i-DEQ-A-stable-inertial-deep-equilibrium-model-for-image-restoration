import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from itertools import product

# Force PyTorch to use a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import deepinv as dinv
from models.deep_equilibrium import DeepEquilibrium
from networks.DRUnet import GSDRUNet

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
psf_size = 21
physics_generator = dinv.physics.generator.MotionBlurGenerator(
    (psf_size, psf_size),
    device=device,
    rng=rng,
    dtype=torch.float32,
    sigma=1.,
    l=0.5
)

filter = physics_generator.step()["filter"]

physics = dinv.physics.BlurFFT(
    img_size=img_size,
    filter=physics_generator.step()["filter"],
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
    overwrite_existing=True,
    device=device,
    save_dir=f"datasets/blur",
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

# lambdas_Rtheta = (5 * np.logspace(-1.0, 0.0, num=10)).tolist()
# sigmas_denoiser = np.linspace(0.03, 0.1, num=8).tolist()
# n_iter_init = [False, True]

# for lambda_Rtheta, sigma_denoiser, init_train in product(lambdas_Rtheta, sigma_denoiser, n_iter_init):

#     accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
#     max_iter = 150
#     backtracking = False
#     lambda_dc = min(0.1, 1/lambda_Rtheta)  # Valeur initiale de lambda_dc, peut être ajustée
#     CNNBlock_model = GSDRUNet(in_channels=3, out_channels=3, pretrained='./networks/GS_DRUNet_SPlus.ckpt', act_mode='s')
#     DC_type = 'grad'
#     learn_lambda_dc = True
#     learn_lambda_Rtheta = True
#     learn_theta_interpol = False
#     theta_interpol = 0.2
#     B_restart = 100
#     random_noise = False
#     noise_level = 5/255

#     -------------------------------------------------------------------------------
#     Dossier pour sauvegarde et paramètres du modèle
#     -------------------------------------------------------------------------------

#     path_folder = "Unrolling_comparison/Deblurring/multikernel/psf21/lambda_Rtheta_{:.4f}_sigma_denoiser_{:.4f}_init_train_{}".format(
#         lambda_Rtheta, sigma_denoiser, init_train
#     )

#     os.makedirs(path_folder, exist_ok=True)

#     model = DeepEquilibrium(
#                     Network=CNNBlock_model, 
#                     problem="deblurring",
#                     DC_type=DC_type,
#                     backtracking=backtracking, 
#                     lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
#                     learn_lambda_dc=learn_lambda_dc, learn_lambda_Rtheta=learn_lambda_Rtheta,
#                     gamma=0.01, eta=0.5,
#                     thresh=1e-4, max_iter=max_iter, 
#                     device=device, path_folder=path_folder,
#                     random_noise=random_noise,
#                     sigma_noise=noise_level,
#                     sigma_denoiser=sigma_denoiser,
#                     theta_interpol=theta_interpol,
#                     restart=True,
#                     B_restart=B_restart,
#                     learn_theta_interpol=learn_theta_interpol,
#                     learn_B_restart=False)

#     -------------------------------------------------------------------------------
#     Entraînement
#     -------------------------------------------------------------------------------
#     pretrained = None
#     if init_train:
#         init_train = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2}
#     else:
#         init_train = None

#     train = False
#     if train:
#         model.train_model(
#                 train_loader=train_dataloader,
#                 val_loader=val_dataloader,
#                 accelerated=accelerated,
#                 init_train=init_train,
#                 JFB=True,
#                 K_JFB=0.,
#                 lr=1e-5,
#                 eta_k=None,
#                 eta_TV=None,
#                 eta_l1=None,
#                 optimizer=torch.optim.Adam,
#                 optimizer_kwargs={"betas": (0.9, 0.999)},
#                 scheduler=None,
#                 scheduler_kwargs=None,
#                 max_patience=25,
#                 max_epochs=500,
#                 plot_interval=1,
#                 pretrained_path=pretrained)

#     pretrained = path_folder + "/best_model.pth"  # chemin vers le modèle pré-entraîné, si disponible
#     test = True
#     if test:
#         dict = model.evaluate(test_loader=val_dataloader, 
#                             n_display=7, 
#                             accelerated=accelerated,
#                             noise_test=noise_level,
#                             init_train=init_train, 
#                             pretrained_path=pretrained, 
#                             PnP=False)

#     test_mse = dict["test_mse"]
#     test_PSNR = dict["test_PSNR"]
#     input_mse = dict["input_mse"]
#     input_PSNR = dict["input_PSNR"]
#     input_SSIM = dict["input_SSIM"]
#     test_SSIM = dict["test_SSIM"]
#     PSNR_per_iter = dict["PSNR_list"]
#     print(f"Test MSE: {test_mse}, Test PSNR: {test_PSNR}, Test SSIM: {test_SSIM}")

#     with open(os.path.join(path_folder, "results.txt"), "w") as f:
#         f.write(f"Accelerated: {accelerated}\n")
#         f.write(f"Random noise: {random_noise}\n")
#         f.write(f"Noise level: {noise_level}\n")
#         f.write(f"Andersen_acceleration: {False}\n")
#         f.write(f"Cycle_Andersen: {False}\n")
#         f.write(f"m_Andersen: {0}\n")
#         f.write("backtracking: {}\n".format(backtracking))
#         f.write(f"DC_type: {DC_type}\n")
#         f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
#         f.write(f"theta_interpol: {theta_interpol}\n")
#         f.write(f"learn_lambda_dc: {learn_lambda_dc}\n")
#         f.write(f"learn_lambda_Rtheta: {learn_lambda_Rtheta}\n")
#         f.write(f"learn_theta_interpol: {learn_theta_interpol}\n")
#         f.write(f"sigma_denoiser: {sigma_denoiser}\n")
#         f.write(f"B_restart: {B_restart}\n")
#         f.write(f"lambda_dc: {lambda_dc}\n")
#         f.write(f"test_mse: {test_mse}\n")
#         f.write(f"test_PSNR: {test_PSNR}\n")
#         f.write(f"input_mse: {input_mse}\n")
#         f.write(f"input_PSNR: {input_PSNR}\n")
#         f.write(f"input_SSIM: {input_SSIM}\n")
#         f.write(f"test_SSIM: {test_SSIM}\n")

# Recherche de la meilleure combinaison de paramètres
# best_PSNR = -np.inf
# best_params = None
# for lambda_Rtheta, sigma_denoiser, init_train in product(lambdas_Rtheta, sigmas_denoiser, n_iter_init):
#     path_folder = "Unrolling_comparison/Deblurring/multikernel/psf21/lambda_Rtheta_{:.4f}_sigma_denoiser_{:.4f}_init_train_{}".format(
#         lambda_Rtheta, sigma_denoiser, init_train
#     )
#     results_file = os.path.join(path_folder, "results.txt")
#     if os.path.exists(results_file):
#         with open(results_file, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if line.startswith("test_PSNR:"):
#                     test_PSNR = float(line.split(":")[1].strip())
#                     if test_PSNR > best_PSNR:
#                         best_PSNR = test_PSNR
#                         best_params = (lambda_Rtheta, sigma_denoiser, init_train)

# if best_params is not None:
#     lambda_Rtheta, sigma_denoiser, init_train = best_params
#     print(f"Best parameters found: lambda_Rtheta={lambda_Rtheta}, sigma_denoiser={sigma_denoiser}, init_train={init_train} with test_PSNR={best_PSNR}")
#     with open("Unrolling_comparison/Deblurring/multikernel/psf21/best_params.txt", "w") as f:
#         f.write('lambda_Rtheta: {:.4f}\n'.format(lambda_Rtheta))
#         f.write('sigma_denoiser: {:.4f}\n'.format(sigma_denoiser))
#         f.write('init_train: {}\n'.format(init_train))
#         f.write('test_PSNR: {:.4f}\n'.format(best_PSNR))
# else:
#     print("No results found to determine the best parameters.")

accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
max_iter = 300
backtracking = False
lambda_dc = 0.1  # Valeur initiale de lambda_dc, peut être ajustée
CNNBlock_model = GSDRUNet(in_channels=3, out_channels=3, pretrained='./networks/GS_DRUNet_SPlus.ckpt', act_mode='s')
DC_type = 'grad'
learn_lambda_dc = False
learn_lambda_Rtheta = True
learn_theta_interpol = False
theta_interpol = 0.2
B_restart = 100
random_noise = False
noise_level = 5/255
lambda_Rtheta= 1.4
sigma_denoiser= 0.03
init_train = None
noise_level_bounds = (1/255, 25.5/255) # Lower bound has to be >0 for weighting of the loss reasons
random_noise = False

# -------------------------------------------------------------------------------
# Dossier pour sauvegarde et paramètres du modèle
# -------------------------------------------------------------------------------

path_folder = "Unrolling_comparison/Deblurring/multikernel/psf21/lambda_Rtheta_{:.4f}_sigma_denoiser_{:.4f}_init_train_{}_bactrack".format(
        lambda_Rtheta, sigma_denoiser, init_train
)

os.makedirs(path_folder, exist_ok=True)

model = DeepEquilibrium(
                    Network=CNNBlock_model, 
                    problem="deblurring",
                    DC_type=DC_type,
                    backtracking=backtracking, 
                    lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
                    learn_lambda_dc=learn_lambda_dc, learn_lambda_Rtheta=learn_lambda_Rtheta,
                    gamma=0.01, eta=0.5,
                    thresh=1e-4, max_iter=max_iter, 
                    device=device, path_folder=path_folder,
                    random_noise=random_noise,
                    sigma_noise=noise_level,
                    noise_bounds=noise_level_bounds,
                    sigma_denoiser=sigma_denoiser,
                    theta_interpol=theta_interpol,
                    restart=True,
                    B_restart=B_restart,
                    learn_theta_interpol=learn_theta_interpol,
                    learn_B_restart=False)

# -------------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------------
train = False
pretrained = None
if train:
    model.train_model(
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                accelerated=accelerated,
                init_train=init_train,
                JFB=True,
                K_JFB=0.,
                eigenvalue_tracking=False,
                lr=5e-6,
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
                pretrained_path=pretrained)

pretrained = path_folder + "/best_model.pth"  # chemin vers le modèle pré-entraîné, si disponible

test = True
if test:
    dict = model.evaluate(test_loader=test_dataloader, 
                            n_display=7, 
                            accelerated=accelerated,
                            noise_test=noise_level,
                            init_train=init_train, 
                            pretrained_path=pretrained, 
                            PnP=False)
    
    test_mse = dict["test_mse"]
    test_PSNR = dict["test_PSNR"]
    input_mse = dict["input_mse"]
    input_PSNR = dict["input_PSNR"]
    input_SSIM = dict["input_SSIM"]
    test_SSIM = dict["test_SSIM"]
    PSNR_per_iter = dict["PSNR_list"]
    print(f"Test MSE: {test_mse}, Test PSNR: {test_PSNR}, Test SSIM: {test_SSIM}")
    
test_multinoise = False
if test_multinoise:
    model.evaluate_multinoise(test_loader=test_dataloader, 
                 init_train=None, 
                 accelerated=accelerated,
                 pretrained_path=pretrained)

with open(os.path.join(path_folder, "results.txt"), "w") as f:
    f.write(f"Accelerated: {accelerated}\n")
    f.write(f"Random noise: {random_noise}\n")
    f.write(f"Noise level: {noise_level}\n")
    f.write(f"Andersen_acceleration: {False}\n")
    f.write(f"Cycle_Andersen: {False}\n")
    f.write(f"m_Andersen: {0}\n")
    f.write("backtracking: {}\n".format(backtracking))
    f.write(f"DC_type: {DC_type}\n")
    f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
    f.write(f"theta_interpol: {theta_interpol}\n")
    f.write(f"learn_lambda_dc: {learn_lambda_dc}\n")
    f.write(f"learn_lambda_Rtheta: {learn_lambda_Rtheta}\n")
    f.write(f"learn_theta_interpol: {learn_theta_interpol}\n")
    f.write(f"sigma_denoiser: {sigma_denoiser}\n")
    f.write(f"B_restart: {B_restart}\n")
    f.write(f"lambda_dc: {lambda_dc}\n")
    f.write(f"test_mse: {test_mse}\n")
    f.write(f"test_PSNR: {test_PSNR}\n")
    f.write(f"input_mse: {input_mse}\n")
    f.write(f"input_PSNR: {input_PSNR}\n")
    f.write(f"input_SSIM: {input_SSIM}\n")
    f.write(f"test_SSIM: {test_SSIM}\n")