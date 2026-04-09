import os
import random
import numpy as np

# Forcer PyTorch à n'utiliser qu'un seul GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # seul le GPU 0 sera visible
device = "cuda:0"

import deepinv as dinv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deepinv.datasets import FastMRISliceDataset
from utils import ifft2c, PSNR

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
    img_size=img_size, acceleration=8, rng=rng, device=device
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
    save_dir=dinv.utils.get_data_home(),
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
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# On affiche les premières images du train et du val pour vérifier que tout est correct
for batch_target, batch_input, param in test_dataloader:
    B = batch_input.shape[0]
    for i in range(B):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(param['mask'][i, 0].cpu(), cmap="gray")
        plt.title("Mask")
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(batch_input[i, 0].cpu(), cmap="gray")
        plt.title("K-space")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        image_input = ifft2c(batch_input)[i, 0].cpu()
        PSNR_value = PSNR(image_input, batch_target[i, 0].cpu())
        PSNR_value = PSNR_value.item() if isinstance(PSNR_value, torch.Tensor) else PSNR_value
        plt.imshow(image_input, cmap="gray")
        plt.title("Undesampled image,\nPSNR: {:.2f} dB".format(PSNR_value))
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.imshow(batch_target[i, 0].cpu(), cmap="gray")
        plt.title("Target image")
        plt.axis("off")
        plt.savefig(f"train_image_{i}.png")
        plt.close()
    break