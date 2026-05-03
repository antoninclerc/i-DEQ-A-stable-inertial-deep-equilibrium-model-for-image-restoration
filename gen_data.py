import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import deepinv as dinv


# -------------------------
# Utils communs
# -------------------------
def set_seed(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator(device=device).manual_seed(seed)


def load_folder_as_tensor(root, transform):
    root = Path(root)
    images = []

    for path in sorted(root.glob("*")):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

    return torch.stack(images)

# -------------------------
# Rician spécifique
# -------------------------
def add_rician_noise(x, sigma):
    noise_real = torch.randn_like(x) * sigma
    noise_imag = torch.randn_like(x) * sigma
    return torch.sqrt((x + noise_real) ** 2 + noise_imag ** 2)


class RicianDataset(TensorDataset):
    def __init__(self, clean_tensor, sigma):
        super().__init__(clean_tensor, clean_tensor)
        self.sigma = sigma

    def __getitem__(self, idx):
        target = self.tensors[0][idx]
        input_ = add_rician_noise(target, self.sigma)
        mask = {'mask': torch.ones_like(target[0:1])}
        return target, input_, mask


# -------------------------
# Factory principale
# -------------------------
def get_dataloaders(problem_type, config):
    """
    problem_type: "mri" | "inpainting" | "rician"
    config: dict contenant les paramètres
    """

    device = config.get("device", "cuda:0")
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu", "0")

    rng = set_seed(config.get("seed", 42), device)

    # -------------------------
    # MRI
    # -------------------------
    if problem_type == "mri":
        from deepinv.datasets import FastMRISliceDataset

        img_size = config["img_size"]
        acc = config["acceleration"]

        def load_split(path):
            dataset = FastMRISliceDataset(root=path, slice_index="middle")
            return dataset.save_simple_dataset(
                path + "/fastmri.pt", pad_to_size=img_size
            )

        train_subset = load_split(config["train_path"])
        val_subset = load_split(config["val_path"])
        test_subset = load_split(config["test_path"])

        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=img_size, acceleration=acc, rng=rng, device=device
        )

        mask = physics_generator.step()["mask"]

        physics = dinv.physics.MRI(mask=mask, img_size=img_size, device=device)

        dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_subset,
            test_dataset=test_subset,
            val_dataset=val_subset,
            physics=physics,
            physics_generator=physics_generator,
            device=device,
            save_dir=f"datasets/mri_{acc}",
            batch_size=4,
            overwrite_existing=True,
        )

        train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
        val_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="val", load_physics_generator_params=True)
        test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

    # -------------------------
    # Inpainting
    # -------------------------
    elif problem_type == "inpainting":
        transform = transforms.ToTensor()
        img_size = config["img_size"]

        train_tensor = load_folder_as_tensor(config["train_path"], transform)
        val_tensor = load_folder_as_tensor(config["val_path"], transform)
        test_tensor = load_folder_as_tensor(config["test_path"], transform)

        train_subset = TensorDataset(train_tensor, train_tensor)
        val_subset = TensorDataset(val_tensor, val_tensor)
        test_subset = TensorDataset(test_tensor, test_tensor)

        physics_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
            img_size=img_size,
            split_ratio=config["split_ratio"],
            rng=rng,
            device=device,
        )

        mask = physics_generator.step()["mask"]

        physics = dinv.physics.Inpainting(
            img_size=img_size,
            mask=mask,
            device=device,
        )

        dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_subset,
            test_dataset=test_subset,
            val_dataset=val_subset,
            physics=physics,
            physics_generator=physics_generator,
            save_physics_generator_params=True,
            device=device,
            save_dir=f"datasets/inpainting_{config['split_ratio']}",
            batch_size=4,
            overwrite_existing=True,
        )

        train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
        val_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="val", load_physics_generator_params=True)
        test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

    # -------------------------
    # Rician
    # -------------------------
    elif problem_type == "rician":
        transform = transforms.ToTensor()

        train_tensor = load_folder_as_tensor(config["train_path"], transform)
        val_tensor = load_folder_as_tensor(config["val_path"], transform)
        test_tensor = load_folder_as_tensor(config["test_path"], transform)

        train_dataset = RicianDataset(train_tensor, config["sigma"])
        val_dataset = RicianDataset(val_tensor, config["sigma"])
        test_dataset = RicianDataset(test_tensor, config["sigma"])

        physics = None  # Important: pas de physics deepinv ici

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # -------------------------
    # DataLoaders communs
    # -------------------------
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return train_loader, val_loader, test_loader, physics