import argparse
import os
import numpy as np
from itertools import product
from networks.DRUnet import GSDRUNet
from models.deep_equilibrium import DeepEquilibrium
from gen_data import get_dataloaders
from utils import str2bool, normalize_problem

class GridSearch:

    def __init__(self, config, val_loader):
        self.cfg = config
        self.val_loader = val_loader

    # =========================================================
    # 1. CONFIG
    # =========================================================
    def compute_lambda_dc(self, R):
        p = self.cfg["problem"]
        dc = self.cfg["DC_type"]

        if p == "MRI":
            if dc == "grad":
                return min(0.5, 0.5 / R)
            else:
                return min(1.0, 1.0 / R)

        elif p == "inpainting":
            return min(0.1, 1 / R)

        elif p == "deblurring":
            return min(0.1, 1 / R)

        elif p == "rician":
            if dc == "grad":
                return 0.03
            else:
                return 1e-2 / R

        else:
            raise ValueError(p)

    def get_model_hyperparams(self):
        if self.cfg["problem"] == "rician":
            return dict(gamma=0.01, theta=0.01, restart=100, learn_R=False)
        elif self.cfg["problem"] == "deblurring":
            return dict(gamma=0.1, theta=0.2, restart=100, learn_R=False)
        else:
            return dict(gamma=0.1, theta=0.2, restart=5000, learn_R=False)

    # =========================================================
    # 2. GRID
    # =========================================================
    def generate_grid(self):
        return product(
            self.cfg["lambda_Rtheta"],
            self.cfg["sigma_denoiser"],
            self.cfg["n_iter_init"],
            self.cfg["init_train"],
        )

    # =========================================================
    # 3. MODEL
    # =========================================================
    def build_model(self, path, R, sigma, lambda_dc):
        hp = self.get_model_hyperparams()

        in_ch = 1 if self.cfg["problem"] == "MRI" else 3

        if in_ch == 1:
            net = GSDRUNet(
                in_channels=in_ch,
                out_channels=in_ch,
                pretrained=self.cfg["pretrained"],
            )
        else:
            net = GSDRUNet(
                in_channels=in_ch,
                out_channels=in_ch,
                 pretrained=self.cfg["pretrained"], 
                 act_mode='s'
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
            f"grid_search/{self.cfg['problem']}"
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

        for R, sigma, n_iter, init_flag in self.generate_grid():

            lambda_dc= self.compute_lambda_dc(R)

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
                    if line.startswith("test_PSNR: "):
                        psnr = float(line.split(": ")[1])
                        if psnr > best_psnr:
                            best_psnr = psnr
                            best_path = root

        print("BEST:", best_path, best_psnr)

def build_config(problem, DC_type, accelerated, sigma_noise=1.0):

    base = {
        "problem": problem,
        "DC_type": DC_type,
        "accelerated": accelerated,
        "device": "cuda:0",
        "sigma_noise": sigma_noise/255,
        "train": False,
        "backtracking": False,
        "pretrained": "./networks/GS_DRUNet_SPlus.ckpt" if problem != "MRI" else "./networks/GSDRUNet_grayscale_torch.ckpt",
    }

    if problem == "MRI":
        base.update({
            "lambda_Rtheta": (5*np.logspace(-1, 0, 10)).tolist(),
            "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
            "max_iter": 200,
            "sigma_noise": sigma_noise/255,
            "n_iter_init": [None],
            "init_train": [False],
        })

    elif problem == "inpainting":
        base.update({
            "lambda_Rtheta": (5*np.logspace(-1, 1, 10)).tolist(),
            "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
            "max_iter": 200,
            "sigma_noise": sigma_noise/255,
            "n_iter_init": [20],
            "init_train": [True],
        })

    elif problem == "deblurring":
            base.update({
                "lambda_Rtheta": (5*np.logspace(-1, 0, 10)).tolist(),
                "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
                "max_iter": 200,
                "sigma_noise": sigma_noise/255,
                "n_iter_init": [None],
                "init_train": [False],
            })

    elif problem == "rician":
        base.update({
            "lambda_Rtheta": (
                np.logspace(-1, 1, 10).tolist()
                if DC_type == "grad"
                else (5*np.logspace(2, 4, 10)).tolist()
            ),
            "sigma_denoiser": np.linspace(0.01, 0.05, 5).tolist(),
            "max_iter": 200,
            "sigma_noise": sigma_noise/255,
            "n_iter_init": [None],
            "init_train": [False],
        })

    else:
        raise ValueError(problem)

    return base

# =========================================================
# Config DATASET (séparée de la grid)
# =========================================================
def build_data_config(problem):

    if problem == "MRI":
        return {
            "img_size": (320, 320),
            "acceleration": 8,
            "train_path": "DATA/MRI/singlecoil_train",
            "val_path": "DATA/MRI/singlecoil_val",
            "test_path": "DATA/MRI/singlecoil_test",
        }

    elif problem == "inpainting":
        return {
            "img_size": (3, 320, 320),
            "split_ratio": 0.5,
            "train_path": "DATA/BSDS500/train",
            "val_path": "DATA/BSDS500/val",
            "test_path": "DATA/BSDS500/test",
        }

    elif problem == "deblurring":
            return {
                "img_size": (3, 320, 320),
                "kernel_size": 21,
                "train_path": "DATA/BSDS500/train",
                "val_path": "DATA/BSDS500/val",
                "test_path": "DATA/BSDS500/test",
            }

    elif problem == "rician":
        return {
            "train_path": "DATA/BSDS500/train",
            "val_path": "DATA/BSDS500/val",
            "test_path": "DATA/BSDS500/test",
        }

    else:
        raise ValueError(problem)


# =========================================================
# MAIN
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", required=True, choices=["mri", "inpainting", "deblurring", "rician"])
    parser.add_argument("--dc", required=True, choices=["grad", "prox"])
    parser.add_argument("--accelerated", type=str2bool, default=True)
    parser.add_argument("--noise_level", type=float, default=1.0, help="Noise level, divided by 255 later on")

    args = parser.parse_args()

    # -------------------------
    # Normalisation
    # -------------------------
    problem = normalize_problem(args.problem)
    DC_type = args.dc
    accelerated = args.accelerated
    noise_level = args.noise_level

    print(f"Running: {problem} | {DC_type} | accelerated={accelerated}")

    # -------------------------
    # Config grid
    # -------------------------
    grid_config = build_config(problem, DC_type, accelerated, noise_level)

    # -------------------------
    # Config data
    # -------------------------
    data_config = build_data_config(problem)

    # merge configs
    full_data_config = {**data_config, **grid_config}

    # -------------------------
    # Data
    # -------------------------
    train_loader, val_loader, test_loader, physics = get_dataloaders(
        args.problem, full_data_config
    )

    # -------------------------
    # Grid Search
    # -------------------------
    engine = GridSearch(grid_config, val_loader)

    engine.run()
    engine.find_best()


if __name__ == "__main__":
    main()
    # python main_gridsearch.py --problem mri --dc grad --accelerated True
