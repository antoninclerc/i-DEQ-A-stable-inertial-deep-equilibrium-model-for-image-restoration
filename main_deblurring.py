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
sigma_noise = 0.05

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
# Add blur
# ---------------------------
def blur(x, kernel):
    """
    Apply blur to an image x using a given kernel.
    x: [B,C,H,W]
    kernel: [1,1,kH,kW]
    """
    padding = (kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    blurred = torch.nn.functional.conv2d(x, kernel, padding=padding)
    return blurred

# ---------------------------
# Blur Dataset
# ---------------------------
class BlurDataset(TensorDataset):
    def __init__(self, clean_tensor):
        super().__init__(clean_tensor)

    def __getitem__(self, idx):
        target = self.tensors[0][idx]
        idx_kernel = random.randint(0, 8)
        kernel = dinv.load_degradation('Levin09.npy', 'kernels', index=idx_kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]
        blurred = blur(target, kernel)
        return target, blurred, kernel


# ---------------------------
# Load datasets
# ---------------------------
train_tensor = load_folder_as_tensor("DATA/BSDS500/train")
val_tensor = load_folder_as_tensor("DATA/BSDS500/val")
test_tensor = load_folder_as_tensor("DATA/BSDS500/test")

train_dataset = BlurDataset(train_tensor)
val_dataset = BlurDataset(val_tensor)
test_dataset = BlurDataset(test_tensor)

# ---------------------------
# DataLoaders
# ---------------------------
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)