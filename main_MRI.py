import os
import random
import numpy as np

# Forcer PyTorch à n'utiliser qu'un seul GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # seul le GPU 0 sera visible
device = "cuda:0"

import deepinv as dinv
import torch
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
root = "DATA/MRI/singlecoil_train"
dataset = FastMRISliceDataset(root=root, slice_index="middle")
train_subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt", pad_to_size=img_size)

root = "DATA/MRI/singlecoil_val"
dataset = FastMRISliceDataset(root=root, slice_index="middle")
val_subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt", pad_to_size=img_size)

root = "DATA/MRI/singlecoil_test"
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

# DC_type = 'grad'
# lambda_Rthetas = (5*np.logspace(-1, 0, num=10)).tolist() # 10 valeurs de lambda_Rtheta entre 0.1 et 10
# # lambda_Rthetas = [1.] # PnP
# accelerateds = [True]
# max_iter = 200
# backtracking = False
# sigma_denoising = np.linspace(0.02, 0.08, num=5).tolist()
# noise_level = 5./255
# init_train = False

# for lambda_Rtheta, sigma_denoiser, accelerated in product(lambda_Rthetas, sigma_denoising, accelerateds):
# # for sigma_denoiser, init_train in product(sigma_denoising, init_trains):
#     lambda_dc = min(0.5, 0.5/lambda_Rtheta)  # lambda_dc dépend de lambda_Rtheta pour éviter des valeurs trop grandes
#     # lambda_dc = 1. # PnP
#     # lambda_Rtheta = 1. # PnP
#     # -------------------------------------------------------------------------------
#     # Dossier pour sauvegarde et paramètres du modèle
#     # -------------------------------------------------------------------------------
    
#     path_folder = "Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/grad_lambda_Rtheta_{:.2f}_sigma_{:.2f}".format(
#       acceleration, accelerated, lambda_Rtheta, sigma_denoiser)
#     # path_folder = "Unrolling_comparison/MRI/PnP/acceleration_{}/grad_sigma_{:.2f}_init_train_{}".format(
#     #     acceleration, sigma_denoiser, init_train)
#     os.makedirs(path_folder, exist_ok=True)
    
#     CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')

#     model = DeepEquilibrium(
#                 Network=CNNBlock_model, 
#                 problem="MRI",
#                 DC_type=DC_type,
#                 backtracking=backtracking, 
#                 lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
#                 learn_lambda_dc=False, learn_lambda_Rtheta=True,
#                 gamma=0.1, eta=0.5,
#                 thresh=1e-4, max_iter=max_iter, 
#                 device=device, path_folder=path_folder, 
#                 sigma_noise=noise_level,
#                 sigma_denoiser=sigma_denoiser,
#                 theta_interpol=0.2,
#                 restart=True,
#                 B_restart=5000,
#                 learn_theta_interpol=False,
#                 learn_B_restart=False)

#     # -------------------------------------------------------------------------------
#     # Entraînement
#     # -------------------------------------------------------------------------------
#     if init_train:
#         init_train_params = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2}
#     else:
#         init_train_params = None

#     pretrained_path = None
#     train = False
#     if train:
#         model.train_model(
#             train_loader=train_dataloader,
#             val_loader=val_dataloader,
#             accelerated=accelerated,
#             JFB=True,
#             K_JFB=0,
#             lr=1e-5,
#             eta_k=None,
#             eta_TV=None,
#             eta_l1=None,
#             optimizer=torch.optim.Adam,
#             optimizer_kwargs={"betas": (0.9, 0.999)},
#             scheduler=None,
#             scheduler_kwargs=None,
#             max_patience=25,
#             max_epochs=1000,
#             plot_interval=1,
#             pretrained_path=pretrained_path)

#     test = True
#     if test:
#         dict = model.evaluate(test_loader=val_dataloader, n_display=2, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

#         test_mse = dict["test_mse"]
#         test_PSNR = dict["test_PSNR"]
#         input_mse = dict["input_mse"]
#         input_PSNR = dict["input_PSNR"]

#         # On sauvegarde les paramètres et les résultats dans un fichier texte
#         with open(os.path.join(path_folder, "results.txt"), "w") as f:
#             f.write(f"DC_type: {DC_type}\n")
#             f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
#             f.write(f"sigma_denoiser: {sigma_denoiser}\n")
#             f.write(f"test_mse: {test_mse}\n")
#             f.write(f"test_PSNR: {test_PSNR}\n")
#             f.write(f"input_mse: {input_mse}\n")
#             f.write(f"input_PSNR: {input_PSNR}\n")

# for accelerated in accelerateds:

#     # recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
#     best_PSNR = -float('inf')
#     best_model_path = None
#     for lambda_Rtheta, sigma_denoiser in product(lambda_Rthetas, sigma_denoising):
#         path_folder = "Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/grad_lambda_Rtheta_{:.2f}_sigma_{:.2f}".format(
#         acceleration, accelerated, lambda_Rtheta, sigma_denoiser)
#         # path_folder = "Unrolling_comparison/MRI/PnP/acceleration_{}/grad_sigma_{:.2f}_init_train_{}".format(
#         #     acceleration, sigma_denoiser, init_train)
#         with open(os.path.join(path_folder, "results.txt"), "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if line.startswith("test_PSNR:"):
#                     PSNR = float(line.split(":")[1].strip())
#                     if PSNR > best_PSNR:
#                         best_PSNR = PSNR
#                         best_model_path = path_folder

#     # On écrit le meilleur modèle dans un fichier texte
#     with open("Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/best_model_grad.txt".format(acceleration, accelerated), "w") as f:
#         f.write(f"Best model path: {best_model_path}\n")
#         f.write(f"Best PSNR: {best_PSNR}\n")

# # ------------------------------------------------------------------------------
# # Paramètres (facilement modifiables)
# # ------------------------------------------------------------------------------
# DC_type = 'prox'
# lambda_Rthetas = (5*np.logspace(-1, 0, num=10)).tolist() # 10 valeurs de lambda_Rtheta entre 0.1 et 10
# # lambda_Rthetas = [1.] # PnP
# accelerateds = [True, False]  # True pour entraînement accéléré, False pour entraînement complet
# max_iter = 300
# backtracking = False
# sigma_denoising = np.linspace(0.01, 0.05, num=5).tolist()
# noise_level = 1./255
# init_train = False
# lambdas_dc = (np.logspace(-4, -2, num=10)).tolist() # 10 valeurs de lambda_dc entre 0.01 et 1

# for lambda_Rtheta, sigma_denoiser, accelerated in product(lambda_Rthetas, sigma_denoising, accelerateds):
# # for sigma_denoiser, init_train, lambda_dc in product(sigma_denoising, init_trains, lambdas_dc):
#     # -------------------------------------------------------------------------------
#     # Dossier pour sauvegarde et paramètres du modèle
#     # -------------------------------------------------------------------------------
#     lambda_dc = min(1., 1./lambda_Rtheta)  # lambda_dc dépend de lambda_Rtheta pour éviter des valeurs trop grandes
#     # lambda_Rtheta = 1. # PnP
#     path_folder = "Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/prox_lambda_Rtheta_{:.2f}_sigma_{:.2f}".format(
#         acceleration, accelerated, lambda_Rtheta, sigma_denoiser)
#     # path_folder = "Unrolling_comparison/MRI/PnP/acceleration_{}/prox_sigma_{:.2f}_init_train_{}_lambda_dc_{:.5f}".format(
#     #     acceleration, sigma_denoiser, init_train, lambda_dc)
#     os.makedirs(path_folder, exist_ok=True)
    

#     CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')

#     model = DeepEquilibrium(
#                 Network=CNNBlock_model, 
#                 problem="MRI",
#                 DC_type=DC_type,
#                 backtracking=backtracking, 
#                 lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
#                 learn_lambda_dc=False, learn_lambda_Rtheta=True,
#                 gamma=0.1, eta=0.5,
#                 thresh=1e-4, max_iter=max_iter, 
#                 device=device, path_folder=path_folder, 
#                 sigma_noise=noise_level,
#                 sigma_denoiser=sigma_denoiser,
#                 theta_interpol=0.2,
#                 restart=True,
#                 B_restart=5000,
#                 learn_theta_interpol=False,
#                 learn_B_restart=False)
    
#     # -------------------------------------------------------------------------------
#     # Entraînement
#     # -------------------------------------------------------------------------------
#     if init_train:
#         init_train_params = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2}
#     else:
#         init_train_params = None
    
#     pretrained_path = None
#     train = False
#     if train:
#         model.train_model(
#             train_loader=train_dataloader,
#             val_loader=val_dataloader,
#             accelerated=accelerated,
#             JFB=True,
#             K_JFB=0,
#             lr=1e-5,
#             eta_k=None,
#             eta_TV=None,
#             eta_l1=None,
#             optimizer=torch.optim.Adam,
#             optimizer_kwargs={"betas": (0.9, 0.999)},
#             scheduler=None,
#             scheduler_kwargs=None,
#             max_patience=25,
#             max_epochs=1000,
#             plot_interval=1,
#             pretrained_path=pretrained_path)
        
#     test = True
#     if test:
#         dict = model.evaluate(test_loader=val_dataloader, n_display=2, accelerated=accelerated, init_train=init_train_params, pretrained_path=pretrained_path, PnP=False)

#         test_mse = dict["test_mse"]
#         test_PSNR = dict["test_PSNR"]
#         input_mse = dict["input_mse"]
#         input_PSNR = dict["input_PSNR"]

#         # On sauvegarde les paramètres et les résultats dans un fichier texte
#         with open(os.path.join(path_folder, "results.txt"), "w") as f:
#                 f.write(f"DC_type: {DC_type}\n")
#                 f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
#                 f.write(f"sigma_denoiser: {sigma_denoiser}\n")
#                 f.write(f"lambda_dc: {lambda_dc}\n")
#                 f.write(f"test_mse: {test_mse}\n")
#                 f.write(f"test_PSNR: {test_PSNR}\n")
#                 f.write(f"input_mse: {input_mse}\n")
#                 f.write(f"input_PSNR: {input_PSNR}\n")

# for accelerated in accelerateds:
    
#     # recherche du meilleur modèle (en fonction du PSNR) parmi tous les modèles entraînés
#     best_PSNR = -float('inf')
#     best_model_path = None

#     for lambda_Rtheta, sigma_denoiser in product(lambda_Rthetas, sigma_denoising):
#         path_folder = "Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/prox_lambda_Rtheta_{:.2f}_sigma_{:.2f}".format(
#             acceleration, accelerated, lambda_Rtheta, sigma_denoiser)
#     # for sigma_denoiser, init_train, lambda_dc in product(sigma_denoising, init_trains, lambdas_dc):
#     #     path_folder = "Unrolling_comparison/MRI/PnP/acceleration_{}/prox_sigma_{:.2f}_init_train_{}_lambda_dc_{:.5f}".format(
#     #         acceleration, sigma_denoiser, init_train, lambda_dc)
#         with open(os.path.join(path_folder, "results.txt"), "r") as f:
#             print(path_folder)
#             lines = f.readlines()
#             for line in lines:
#                 print(path_folder)
#                 if line.startswith("test_PSNR:"):
#                     PSNR = float(line.split(":")[1].strip())
#                     if PSNR > best_PSNR:
#                         best_PSNR = PSNR
#                         best_model_path = path_folder

#     # On écrit le meilleur modèle dans un fichier texte
#     with open("Unrolling_comparison/MRI/sigma1255/acceleration_{}/accelerated_{}/best_model_prox.txt".format(acceleration, accelerated), "w") as f:
#         f.write(f"Best model path: {best_model_path}\n")
#         f.write(f"Best PSNR: {best_PSNR}\n")

# lambda_Rtheta = 0.1
# sigma_denoiser = 0.0
# noise_level = 1./255
# accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
# max_iter = 500
# backtracking = False
# init_train = False
# lambda_dc = 1.0 # lambda_dc dépend de lambda_Rtheta pour éviter des valeurs trop grandes
# CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')
# DC_type = 'grad'

# # -------------------------------------------------------------------------------
# # Dossier pour sauvegarde et paramètres du modèle
# # -------------------------------------------------------------------------------
    
# path_folder = "Unrolling_comparison/MRI/DEQ/grad_acceleration_{}_Rtheta_{:.2f}_lambda_dc_{:.2f}_noise_{:.3f}_sigma_denoiser_{:.2f}".format(acceleration, lambda_Rtheta, lambda_dc, noise_level, sigma_denoiser)
# # os.makedirs(path_folder, exist_ok=True)
# os.makedirs(path_folder, exist_ok=True)

# model = DeepEquilibrium(
#                 Network=CNNBlock_model, 
#                 problem="MRI",
#                 DC_type=DC_type,
#                 backtracking=backtracking, 
#                 lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
#                 learn_lambda_dc=True, learn_lambda_Rtheta=True,
#                 gamma=0.01, eta=0.5,
#                 thresh=1e-4, max_iter=max_iter, 
#                 device=device, path_folder=path_folder, 
#                 sigma_noise=noise_level,
#                 sigma_denoiser=sigma_denoiser,
#                 theta_interpol=0.2,
#                 restart=True,
#                 B_restart=100,
#                 learn_theta_interpol=False,
#                 learn_B_restart=False)

# # -------------------------------------------------------------------------------
# # Entraînement
# # -------------------------------------------------------------------------------

# pretrained = path_folder + "/best_model.pth"  # mettre à None si pas de modèle pré-entraîné
# train = False
# if train:
#     model.train_model(
#             train_loader=train_dataloader,
#             val_loader=val_dataloader,
#             accelerated=accelerated,
#             init_train=None,
#             JFB=True,
#             K_JFB=0.,
#             lr=1e-5,
#             eta_k=None,
#             eta_TV=None,
#             eta_l1=None,
#             optimizer=torch.optim.Adam,
#             optimizer_kwargs={"betas": (0.9, 0.999)},
#             scheduler=None,
#             scheduler_kwargs=None,
#             max_patience=25,
#             max_epochs=500,
#             plot_interval=1,
#             pretrained_path=pretrained)

# pretrained = path_folder + "/best_model.pth"  # chemin vers le modèle pré-entraîné, si disponible
# test = True
# if test:
#     dict = model.evaluate(test_loader=test_dataloader, n_display=7, accelerated=accelerated, init_train=None, pretrained_path=pretrained, PnP=False)

# test_mse = dict["test_mse"]
# test_PSNR = dict["test_PSNR"]
# input_mse = dict["input_mse"]
# input_PSNR = dict["input_PSNR"]
# input_SSIM = dict["input_SSIM"]
# test_SSIM = dict["test_SSIM"]

# with open(os.path.join(path_folder, "results.txt"), "w") as f:
#     f.write(f"DC_type: {DC_type}\n")
#     f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
#     f.write(f"sigma_denoiser: {sigma_denoiser}\n")
#     f.write(f"lambda_dc: {lambda_dc}\n")
#     f.write(f"test_mse: {test_mse}\n")
#     f.write(f"test_PSNR: {test_PSNR}\n")
#     f.write(f"input_mse: {input_mse}\n")
#     f.write(f"input_PSNR: {input_PSNR}\n")
#     f.write(f"input_SSIM: {input_SSIM}\n")
#     f.write(f"test_SSIM: {test_SSIM}\n")

# lambda_Rtheta = 0.65
# sigma_denoiser = 0.03
# noise_level = 1./255
# accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
# max_iter = 200
# backtracking = False
# init_train = False
# lambda_dc = .5 # lambda_dc dépend de lambda_Rtheta pour éviter des valeurs trop grandes
# CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')
# DC_type = 'grad'
# learn_lambda_dc = False
# learn_lambda_Rtheta = True
# learn_theta_interpol = False
# B_restart = 100

# # -------------------------------------------------------------------------------
# # Dossier pour sauvegarde et paramètres du modèle
# # -------------------------------------------------------------------------------

# path_folder = "Unrolling_comparison/MRI/DEQ/prox_acceleration_{}_Rtheta_{:.2f}_lambda_dc_{:.2f}_noise_{:.3f}_sigma_denoiser_{:.2f}_max_iter_{}_accelerated_{}_B_restart_{}_backtracking_{}".format(acceleration, lambda_Rtheta, lambda_dc, noise_level, sigma_denoiser, max_iter, accelerated, B_restart, backtracking)
# os.makedirs(path_folder, exist_ok=True)

# model = DeepEquilibrium(
#                 Network=CNNBlock_model, 
#                 problem="MRI",
#                 DC_type=DC_type,
#                 backtracking=backtracking, 
#                 lambda_dc=lambda_dc, lambda_Rtheta=lambda_Rtheta,
#                 learn_lambda_dc=learn_lambda_dc, learn_lambda_Rtheta=learn_lambda_Rtheta,
#                 gamma=0.01, eta=0.5,
#                 thresh=1e-4, max_iter=max_iter, 
#                 device=device, path_folder=path_folder, 
#                 sigma_noise=noise_level,
#                 sigma_denoiser=sigma_denoiser,
#                 theta_interpol=0.2,
#                 restart=True,
#                 B_restart=B_restart,
#                 learn_theta_interpol=learn_theta_interpol,
#                 learn_B_restart=False)

# # -------------------------------------------------------------------------------
# # Entraînement
# # -------------------------------------------------------------------------------
# # pretrained = "Unrolling_comparison/MRI/DEQ/prox_acceleration_8_Rtheta_0.83_lambda_dc_2.00_noise_0.004_sigma_denoiser_0.03_learn_lambda_dc_accelerated_False" + "/best_model.pth"  # mettre à None si pas de modèle pré-entraîné
# pretrained = None
# train = True
# if train:
#     model.train_model(
#             train_loader=train_dataloader,
#             val_loader=val_dataloader,
#             accelerated=accelerated,
#             init_train=None,
#             JFB=True,
#             K_JFB=0.,
#             lr=1e-5,
#             eta_k=None,
#             eta_TV=None,
#             eta_l1=None,
#             optimizer=torch.optim.Adam,
#             optimizer_kwargs={"betas": (0.9, 0.999)},
#             scheduler=None,
#             scheduler_kwargs=None,
#             max_patience=25,
#             max_epochs=500,
#             plot_interval=1,
#             pretrained_path=pretrained)

# pretrained = path_folder + "/best_model.pth"  # chemin vers le modèle pré-entraîné, si disponible
# test = True
# if test:
#     dict = model.evaluate(test_loader=test_dataloader, n_display=7, accelerated=accelerated, init_train=None, pretrained_path=pretrained, PnP=False)

# test_mse = dict["test_mse"]
# test_PSNR = dict["test_PSNR"]
# input_mse = dict["input_mse"]
# input_PSNR = dict["input_PSNR"]
# input_SSIM = dict["input_SSIM"]
# test_SSIM = dict["test_SSIM"]
# PSNR_per_iter = dict["PSNR_list"]

# with open(os.path.join(path_folder, "results.txt"), "w") as f:
#     f.write(f"Accelerated: {accelerated}\n")
#     f.write("backtracking: {}\n".format(backtracking))
#     f.write(f"DC_type: {DC_type}\n")
#     f.write(f"lambda_Rtheta: {lambda_Rtheta}\n")
#     f.write(f"learn_lambda_dc: {learn_lambda_dc}\n")
#     f.write(f"learn_lambda_Rtheta: {learn_lambda_Rtheta}\n")
#     f.write(f"learn_theta_interpol: {learn_theta_interpol}\n")
#     f.write(f"sigma_denoiser: {sigma_denoiser}\n")
#     f.write(f"B_restart: {B_restart}\n")
#     f.write(f"lambda_dc: {lambda_dc}\n")
#     f.write(f"test_mse: {test_mse}\n")
#     f.write(f"test_PSNR: {test_PSNR}\n")
#     f.write(f"input_mse: {input_mse}\n")
#     f.write(f"input_PSNR: {input_PSNR}\n")
#     f.write(f"input_SSIM: {input_SSIM}\n")
#     f.write(f"test_SSIM: {test_SSIM}\n")

lambda_Rtheta = 0.65
sigma_denoiser = 0.04
accelerated = True  # True pour entraînement accéléré, False pour entraînement complet
andersen_acceleration = False
cycle_andersen = False
m_andersen = 5
max_iter = 300
backtracking = False
init_train = False
lambda_dc = .5 # lambda_dc dépend de lambda_Rtheta pour éviter des valeurs trop grandes
CNNBlock_model = GSDRUNet(in_channels=1, out_channels=1, pretrained='networks/GSDRUNet_grayscale_torch.ckpt')
DC_type = 'grad'
learn_lambda_dc = False
learn_lambda_Rtheta = True
learn_theta_interpol = False
theta_interpol = 0.2
B_restart = 100
random_noise = False
noise_level = 5./255
noise_level_bounds = (1/255, 25.5/255) # Lower bound has to be >0 for weighting of the loss reasons

# -------------------------------------------------------------------------------
# Dossier pour sauvegarde et paramètres du modèle
# -------------------------------------------------------------------------------

path_folder = "Unrolling_comparison/MRI/DEQ/tau_decrease_5255"
os.makedirs(path_folder, exist_ok=True)

model = DeepEquilibrium(
                Network=CNNBlock_model, 
                problem="MRI",
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
                learn_B_restart=False,
                m_andersen=m_andersen,
                cycle_andersen=cycle_andersen,)

# -------------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------------
pretrained = None
train = False
if train:
    model.train_model(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            accelerated=accelerated,
            andersen_acceleration=andersen_acceleration,
            init_train=None,
            JFB=True,
            K_JFB=0.,
            lr=1e-5,
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
            pretrained_path=pretrained,
            eigenvalue_tracking=False)

pretrained = path_folder + "/best_model.pth"  # chemin vers le modèle pré-entraîné, si disponible
test = True
if test:
    dict = model.evaluate(test_loader=test_dataloader, 
                          n_display=7, 
                          accelerated=accelerated,
                          andersen_acceleration=andersen_acceleration,
                          noise_test=25.5/255,
                          init_train=None, 
                          pretrained_path=pretrained, 
                          PnP=False)

test_multinoise = False
if test_multinoise:
    model.evaluate_multinoise(test_loader=test_dataloader, 
                 init_train=None, 
                 accelerated=accelerated,
                 andersen_acceleration=andersen_acceleration,
                 pretrained_path=pretrained,
                 number_noises=10)

test_mse = dict["test_mse"]
test_PSNR = dict["test_PSNR"]
input_mse = dict["input_mse"]
input_PSNR = dict["input_PSNR"]
input_SSIM = dict["input_SSIM"]
test_SSIM = dict["test_SSIM"]
PSNR_per_iter = dict["PSNR_list"]

with open(os.path.join(path_folder, "results.txt"), "w") as f:
    f.write(f"Accelerated: {accelerated}\n")
    f.write(f"Random noise: {random_noise}\n")
    f.write(f"Noise level: {noise_level}\n")
    f.write(f"Andersen_acceleration: {andersen_acceleration}\n")
    f.write(f"Cycle_Andersen: {cycle_andersen}\n")
    f.write(f"m_Andersen: {m_andersen}\n")
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