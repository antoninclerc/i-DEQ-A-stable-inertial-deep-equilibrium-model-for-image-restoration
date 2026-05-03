import os
import random
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"

import torch

from gen_data import get_dataloaders
from models.Unrolling import Unrolling
import deepinv as dinv


# =========================================================
# Seed
# =========================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# Model builder
# =========================================================
def build_model(args, path_folder):

    # DC dépend du mode
    if args.mode.lower() == "varnet":
        dc_type = "grad"
    elif args.mode.lower() == "modl":
        dc_type = "prox"
    else:
        raise ValueError("mode must be VarNet or MoDL")

    cnn = dinv.models.DRUNet(in_channels=1, out_channels=1)

    return Unrolling(
        Network=cnn,
        problem="MRI",
        DC_type=dc_type,
        lambda_dc=args.lambda_dc,
        learn_lambda_dc=True,
        max_iter=args.max_iter,
        device=device,
        path_folder=path_folder,
        sigma_noise=args.noise,
        use_noise=True,
        sigma_denoiser=args.noise
    )


# =========================================================
# Train
# =========================================================
def train_model(model, train_loader, val_loader, args):

    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        mode=args.mode,
        lr=args.lr,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.9, 0.999)},
        scheduler=None,
        scheduler_kwargs=None,
        max_epochs=args.max_epochs,
        max_patience=25,
        plot_interval=1,
        pretrained_path=None
    )

# =========================================================
# Test
# =========================================================
def test_model(model, test_loader, path_folder, args):

    pretrained_path = os.path.join(path_folder, "best_model.pth")

    results = model.evaluate(
        test_loader=test_loader,
        mode=args.mode,
        n_display=5,
        pretrained_path=pretrained_path
    )

    with open(os.path.join(path_folder, "results.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print("TEST PSNR:", results["test_PSNR"])
    print("TEST SSIM:", results["test_SSIM"])


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    # -------------------------
    # mode
    # -------------------------
    parser.add_argument("--mode", type=str, required=True, choices=["VarNet", "MoDL"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # -------------------------
    # hyperparams
    # -------------------------
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--lambda_dc", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=500)

    # -------------------------
    # physics
    # -------------------------
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--noise", type=float, default=1/255)

    # -------------------------
    # paths
    # -------------------------
    parser.add_argument("--data_root", type=str, default="DATA")
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    set_seed()

    # =========================================================
    # DATA (réutilisation de gen_data.py)
    # =========================================================
    config = {
        "problem": "mri",
        "acceleration": args.acceleration,
        "sigma": args.noise,
        "img_size": (320, 320),
        "train_path": os.path.join(args.data_root, "MRI/singlecoil_train"),
        "val_path": os.path.join(args.data_root, "MRI/singlecoil_val"),
        "test_path": os.path.join(args.data_root, "MRI/singlecoil_test"),
        "device": device,
        "seed": 42,
    }

    train_loader, val_loader, test_loader, _ = get_dataloaders("mri", config)

    # =========================================================
    # PATH
    # =========================================================
    path_folder = os.path.join(
        args.save_dir,
        f"{args.mode}_acc{args.acceleration}_dc{args.lambda_dc}_noise{args.noise:.4f}"
    )

    os.makedirs(path_folder, exist_ok=True)

    # =========================================================
    # MODEL
    # =========================================================
    model = build_model(args, path_folder)

    # =========================================================
    # TRAIN
    # =========================================================
    if args.train:
        train_model(model, train_loader, val_loader, args)

    # =========================================================
    # TEST
    # =========================================================
    if args.test:
        test_model(model, test_loader, path_folder, args)


if __name__ == "__main__":
    main()