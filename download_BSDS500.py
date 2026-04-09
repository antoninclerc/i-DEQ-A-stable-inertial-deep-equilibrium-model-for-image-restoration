import os
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

# Root directory for dataset storage
ROOT = Path("DATA/BSDS500")
DOWNLOAD_DIR = ROOT / "raw"

# Output directories
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
VAL_DIR = ROOT / "val"

for d in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)


def save_subset(dataset, indices, output_dir, resize):
    """
    Save a subset of images from a dataset after resizing and normalizing channels to [0,1].
    """
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]  # labels are not needed
        img = resize(img)  # PIL image

        # Convert to tensor in [0,1]
        img_tensor = transforms.ToTensor()(img)  # shape: (C,H,W), values in [0,1]

        # Optional: back to PIL to save as PNG
        img_normalized = transforms.ToPILImage()(img_tensor)
        filename = output_dir / f"img_{i:04d}.png"
        img_normalized.save(filename)


def sample_indices(dataset_size, n):
    """
    Randomly sample n unique indices from a dataset.
    """
    return random.sample(range(dataset_size), n)


def main():

    resize = transforms.Resize((320, 320))

    # Load dataset splits
    train_dataset = datasets.SBDataset(
        root=DOWNLOAD_DIR,
        image_set="train",
        mode="segmentation",
        download=True
    )

    val_dataset = datasets.SBDataset(
        root=DOWNLOAD_DIR,
        image_set="val",
        mode="segmentation",
        download=False
    )

    # For testing we reuse validation images since BSDS500
    test_dataset = val_dataset

    train_idx = sample_indices(len(train_dataset), 100)
    val_idx = sample_indices(len(val_dataset), 10)
    test_idx = sample_indices(len(test_dataset), 10)

    save_subset(train_dataset, train_idx, TRAIN_DIR, resize)
    save_subset(val_dataset, val_idx, VAL_DIR, resize)
    save_subset(test_dataset, test_idx, TEST_DIR, resize)


if __name__ == "__main__":
    main()