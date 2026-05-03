import os
import torch
import numpy as np
from itertools import product
from networks.DRUnet import GSDRUNet
from models.deep_equilibrium import DeepEquilibrium


class GridSearch:

    def __init__(self, config, val_loader):
        self.cfg = config
        self.val_loader = val_loader

    # =========================================================
    # 1. CONFIG
    # =========================================================
    def compute_lambda_dc(self, R, lambda_dc):
        p = self.cfg["problem"]
        dc = self.cfg["DC_type"]

        if p == "MRI":
            if dc == "grad":
                return min(0.5, 0.5 / R)
            else:
                return min(1.0, 1.0 / R)

        elif p == "inpainting":
            return min(0.1, 1 / R)

        elif p == "rician":
            return min(lambda_dc, 1 / (5 * R))

        else:
            raise ValueError(p)

    def get_model_hyperparams(self):
        if self.cfg["problem"] == "rician":
            return dict(gamma=0.01, theta=0.01, restart=100, learn_R=False)
        else:
            return dict(gamma=0.1, theta=0.2, restart=5000, learn_R=True)

    # =========================================================
    # 2. GRID
    # =========================================================
    def generate_grid(self):
        return product(
            self.cfg["lambda_Rtheta"],
            self.cfg["sigma_denoiser"],
            self.cfg["n_iter_init"],
            self.cfg["lambda_dc"],
            self.cfg["init_train"],
        )

    # =========================================================
    # 3. MODEL
    # =========================================================
    def build_model(self, path, R, sigma, lambda_dc):
        hp = self.get_model_hyperparams()

        in_ch = 1 if self.cfg["problem"] == "MRI" else 3

        net = GSDRUNet(
            in_channels=in_ch,
            out_channels=in_ch,
            pretrained=self.cfg["pretrained"]
        )

        return DeepEquilibrium(
            Network=net,
            problem=self.cfg["problem"],
            DC_type=self.cfg["DC_type"],
            backtracking=self.cfg["backtracking"],
            lambda_dc=lambda_dc,
            lambda_Rtheta=R,
            learn_lambda_dc=False,
            learn_lambda_Rtheta=False,
            gamma=hp["gamma"],
            eta=0.5,
            thresh=1e-4,
            max_iter=self.cfg["max_iter"],
            device=self.cfg["device"],
            path_folder=path,
            sigma_noise=self.cfg["sigma_noise"],
            sigma_denoiser=sigma,
            theta_interpol=hp["theta"],
            restart=True,
            B_restart=hp["restart"],
        )

    # =========================================================
    # 4. PATH
    # =========================================================
    def build_path(self, R, sigma, n_iter, lambda_dc, init_train):
        return (
            f"Unrolling_comparison/{self.cfg['problem']}/"
            f"{self.cfg['DC_type']}_"
            f"R_{R:.2e}_"
            f"s_{sigma:.3f}_"
            f"n_{n_iter}_"
            f"dc_{lambda_dc:.2e}_"
            f"init_{init_train}"
        )

    # =========================================================
    # 5. RUN
    # =========================================================
    def run(self):

        for R, sigma, n_iter, lambda_dc, init_flag in self.generate_grid():

            lambda_dc = self.compute_lambda_dc(R, lambda_dc)

            path = self.build_path(R, sigma, n_iter, lambda_dc, init_flag)
            os.makedirs(path, exist_ok=True)

            model = self.build_model(path, R, sigma, lambda_dc)

            init_params = (
                {"epoch_pretraining": n_iter, "sigma_pretraining": 0.2}
                if init_flag else None
            )

            # -------- TEST --------
            metrics = model.evaluate(
                test_loader=self.val_loader,
                accelerated=self.cfg["accelerated"],
                init_train=init_params,
                PnP=self.cfg["PnP"],
            )

            self.save(path, metrics)

    # =========================================================
    # 6. SAVE
    # =========================================================
    def save(self, path, metrics):
        with open(os.path.join(path, "results.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

    # =========================================================
    # 7. BEST MODEL
    # =========================================================
    def find_best(self):
        best_psnr = -float("inf")
        best_path = None

        for root, _, files in os.walk(f"Unrolling_comparison/{self.cfg['problem']}"):
            if "results.txt" not in files:
                continue

            with open(os.path.join(root, "results.txt")) as f:
                for line in f:
                    if line.startswith("test_PSNR:"):
                        psnr = float(line.split(":")[1])
                        if psnr > best_psnr:
                            best_psnr = psnr
                            best_path = root

        print("BEST:", best_path, best_psnr)

def build_config(problem, DC_type, accelerated):

    base = {
        "problem": problem,
        "DC_type": DC_type,
        "accelerated": accelerated,
        "device": "cuda:0",
        "train": False,
        "backtracking": False,
        "pretrained": "./networks/GS_DRUNet_SPlus.ckpt" if problem != "MRI" else "./networks/GSDRUNet_grayscale_torch.ckpt",
    }

    if problem == "MRI":
        base.update({
            "lambda_Rtheta": (5*np.logspace(-1, 0, 10)).tolist(),
            "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
            "max_iter": 200,
            "sigma_noise": 1./255,
            "n_iter_init": [None],
            "lambda_dc": [0.0],
            "init_train": [False],
        })

    elif problem == "inpainting":
        base.update({
            "lambda_Rtheta": (5*np.logspace(-1, 1, 10)).tolist(),
            "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
            "max_iter": 200,
            "sigma_noise": 5/255,
            "n_iter_init": [20],
            "lambda_dc": [0.0],
            "init_train": [True],
        })

    elif problem == "rician":
        base.update({
            "lambda_Rtheta": (
                np.logspace(-1, 1, 10).tolist()
                if DC_type == "grad"
                else (5*np.logspace(1, 3, 10)).tolist()
            ),
            "sigma_denoiser": np.linspace(0.01, 0.1, 10).tolist(),
            "max_iter": 200,
            "sigma_noise": 0.1,
            "n_iter_init": [20],
            "lambda_dc": [0.03] if DC_type == "grad" else [5e-4],
            "init_train": [False],
        })

    else:
        raise ValueError(problem)

    return base