import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import time

import deepinv as dinv
from networks.DRUnet import GSDRUNet

# ---------------------------
# Setup & Device
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"

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
# Dataset Parameters & Helpers
# ---------------------------
img_size = (3, 320, 320)
transform = transforms.ToTensor()


def load_folder_as_tensor(root):
    """Charge toutes les images d'un dossier sous forme de Tensor [N, C, H, W]."""
    root = Path(root)
    images = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = Image.open(path).convert("RGB")
            img = transform(img)  # [C, H, W] dans [0, 1]
            images.append(img)
    return torch.stack(images)


def get_h5_file_from_dir(dataset_dir):
    """Trouve le fichier .h5 dans un dossier pour éviter le IsADirectoryError."""
    dataset_path = Path(dataset_dir)
    if dataset_path.is_dir():
        h5_files = list(dataset_path.glob("*.h5"))
        if len(h5_files) == 0:
            raise FileNotFoundError(f"Aucun fichier .h5 trouvé dans {dataset_dir}")
        return str(h5_files[0])
    return str(dataset_path)


# ---------------------------
# Setup Physics & HDF5 Generation
# ---------------------------
def setup_physics(problem_type="deblurring", sigma=1.0 / 255.0):
    print(f"\n--- Chargement des images BSDS500 pour {problem_type} ---")
    train_tensor = load_folder_as_tensor("DATA/BSDS500/train")
    val_tensor = load_folder_as_tensor("DATA/BSDS500/val")
    test_tensor = load_folder_as_tensor("DATA/BSDS500/test")

    train_subset = TensorDataset(train_tensor, train_tensor)
    val_subset = TensorDataset(val_tensor, val_tensor)
    test_subset = TensorDataset(test_tensor, test_tensor)

    if problem_type == "deblurring":
        psf_size = 21
        physics_generator = dinv.physics.generator.MotionBlurGenerator(
            (psf_size, psf_size),
            device=device,
            rng=rng,
            dtype=torch.float32,
            sigma=1.0,
            l=0.5,
        )

        physics = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=physics_generator.step()["filter"],
            device=device,
        )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)
        save_dir = "datasets/blur"

    elif problem_type == "inpainting":
        split_ratio = 0.5
        physics_generator = (
            dinv.physics.generator.BernoulliSplittingMaskGenerator(
                img_size=img_size,
                split_ratio=split_ratio,
                device=device,
                rng=rng,
            )
        )

        mask = physics_generator.step()["mask"]
        physics = dinv.physics.Inpainting(
            img_size=img_size,
            mask=mask,
            noise_model=dinv.physics.GaussianNoise(sigma=sigma),
            device=device,
        )
        save_dir = f"datasets/{split_ratio}"

    else:
        raise ValueError(f"Problème inconnu : {problem_type}")

    # ---------------------------
    # Génération du Dataset HDF5
    # ---------------------------
    print(f"Génération du dataset HDF5 dans : {save_dir}")
    generated_path = dinv.datasets.generate_dataset(
        train_dataset=train_subset,
        test_dataset=test_subset,
        val_dataset=val_subset,
        physics=physics,
        physics_generator=physics_generator,
        save_physics_generator_params=True,
        overwrite_existing=True,
        device=device,
        save_dir=save_dir,
        batch_size=4,
    )

    # Résolution du chemin .h5
    h5_path = get_h5_file_from_dir(generated_path)

    # ---------------------------
    # Chargement HDF5 Test Dataset
    # ---------------------------
    test_dataset = dinv.datasets.HDF5Dataset(
        h5_path,
        split="test",
        load_physics_generator_params=True,
    )

    val_dataset = dinv.datasets.HDF5Dataset(
        h5_path,
        split="val",
        load_physics_generator_params=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return physics, test_dataloader, val_dataloader


# ---------------------------
# Wrapper GSDRUNet pour DeepInv
# ---------------------------
class GSDRUNetWrapper(torch.nn.Module):
    def __init__(
        self, ckpt_path="./networks/GS_DRUNet_SPlus.ckpt", device="cuda:0"
    ):
        super().__init__()
        # self.net = GSDRUNet(
        #     in_channels=3, out_channels=3, pretrained=ckpt_path, act_mode="s"
        # ).to(device)
        self.net = dinv.models.DRUNet(device=device)

    def forward(self, x, sigma):
        return self.net(x, sigma)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


# ---------------------------
# Reconstruction Loop (DDRM / DPS)
# ---------------------------
def run_reconstruction(
    problem_type="deblurring",
    algo="DDRM",
    ckpt_path="./networks/GS_DRUNet_SPlus.ckpt",
    save_dir="results",
    sigma=1.0 / 255.0,
    DPS_weight=5.0,
):
    os.makedirs(save_dir, exist_ok=True)
    physics, test_dataloader, val_dataloader = setup_physics(problem_type, sigma=sigma)

    denoiser = GSDRUNetWrapper(ckpt_path=ckpt_path, device=device)
    denoiser.eval()

    # Choix du Sampler DeepInv selon l'algorithme
    if algo.upper() == "DDRM":
        # DDRM s'appuie sur une grille de bruits discrétisée (sigmas)
        sigmas = np.linspace(1.0, 0.01, 300)
        model = dinv.sampling.DDRM(
            denoiser=denoiser,
            sigmas=sigmas,
            etab=1.0,  # 1.0 = stochastique (DDPM), 0.0 = déterministe (DDIM)
            verbose=True,
        ).to(device)

    elif algo.upper() == "DPS":
        # DPS utilise la signature officielle SDE (Schedule, num_steps, weight)
        model = dinv.sampling.DPS(
            denoiser,
            schedule="ve",  # Noise schedule (Variance Preserving par défaut)
            weight=DPS_weight,  # Poids hyperparamètre lambda de la data fidelity
            num_steps=300,  # Nombre de pas de diffusion (ou 100/500 selon le besoin)
            alpha=1.0,  # 1.0 pour DDPM, 0.0 pour DDIM
            verbose=True,
            device=device,
            dtype=torch.float64,  # Recommandé pour éviter les instabilités numériques
            minus_one_one=False,  # Indique si les images sont dans [-1, 1] ou [0, 1]
            rng=torch.Generator(device=device),  # <--- Ajouté pour corriger le warning
        )
    elif algo.upper() == "RAM":
        model = dinv.models.RAM(
            device=device,
        )
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")

    psnr_metric = dinv.metric.PSNR()
    ssim_metric = dinv.metric.SSIM()
    psnrs, ssims = [], []
    input_psnr, input_ssim = [], []
    saved_count = 0

    print(f"\n=== Lancement {algo} | Problème: {problem_type} ===")

    time_start = time.time()
    for i, batch in enumerate(test_dataloader):
        # Unpack des données depuis HDF5Dataset
        if isinstance(batch, (list, tuple)):
            x_true = batch[0].to(device)
            y = batch[1].to(device)
            params = batch[2] if len(batch) > 2 else {}
        elif isinstance(batch, dict):
            x_true = batch["x"].to(device)
            y = batch["y"].to(device)
            params = batch.get("params", {})

        batch_size = x_true.shape[0]
        if batch_size == 1:
            if problem_type == "deblurring":
                if "filter" in params:
                    # Réassignation directe pour éviter l'accumulation de tenseurs par update_parameters
                    physics.filter = params["filter"].to(device)

            elif problem_type == "inpainting":
                if "mask" in params:
                    physics.mask = params["mask"].to(device)

        # Inférence selon l'algo
        if algo.upper() == "DPS":
            # DPS effectue la rétropropagation sur x_t, donc ne pas utiliser torch.no_grad()
            with torch.no_grad():
                x_hat = model(y, physics, seed=seed)
        else:
            with torch.no_grad():
                x_hat = model(y, physics)

        x_hat = torch.clamp(x_hat, 0, 1)

        # Métriques du batch
        cur_psnr = psnr_metric(x_true, x_hat).mean().item()
        cur_ssim = ssim_metric(x_true, x_hat).mean().item()
        curr_input_psnr = psnr_metric(x_true, y).mean().item()
        curr_input_ssim = ssim_metric(x_true, y).mean().item()

        input_psnr.append(curr_input_psnr)
        input_ssim.append(curr_input_ssim)
        psnrs.append(cur_psnr)
        ssims.append(cur_ssim)

        # --- Sauvegarde des 5 premières images avec Matplotlib ---
        batch_size = x_true.shape[0]
        for b in range(batch_size):
            if saved_count < 5:
                single_psnr = psnr_metric(
                    x_true[b : b + 1], x_hat[b : b + 1]
                ).item()
                single_ssim = ssim_metric(
                    x_true[b : b + 1], x_hat[b : b + 1]
                ).item()

                single_input_psnr = psnr_metric(
                    x_true[b : b + 1], y[b : b + 1]
                ).item()
                single_input_ssim = ssim_metric(
                    x_true[b : b + 1], y[b : b + 1]
                ).item()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(tensor_to_numpy(x_true[b]))
                axes[0].set_title("Original ($x$)")
                axes[0].axis("off")

                axes[1].imshow(tensor_to_numpy(y[b]))
                axes[1].set_title(
                    f"Dégradé ($y$)\nPSNR: {single_input_psnr:.2f} dB | SSIM: {single_input_ssim:.4f}"
                )
                axes[1].axis("off")

                axes[2].imshow(tensor_to_numpy(x_hat[b]))
                axes[2].set_title(
                    f"{algo} ($\hat{{x}}$)\nPSNR: {single_psnr:.2f} dB | SSIM: {single_ssim:.4f}"
                )
                axes[2].axis("off")

                plt.tight_layout()

                out_path = os.path.join(
                    save_dir,
                    f"{algo}_{problem_type}_fig_{saved_count + 1}.pdf",
                )
                plt.savefig(out_path, bbox_inches="tight", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(tensor_to_numpy(x_hat[b]))
                ax.axis("off")
                out_path_single = os.path.join(
                    save_dir,
                    f"{algo}_{problem_type}_single_{saved_count + 1}.pdf",
                )
                plt.title(
                    f"PSNR: {single_psnr:.2f} dB | SSIM: {single_ssim:.4f}"
                )
                plt.savefig(out_path_single, bbox_inches="tight", dpi=300)
                plt.close(fig)

                print(
                    f"  -> Figure {saved_count + 1}/5 sauvegardée : {out_path_single}"
                )

                saved_count += 1

        print(
            f"Batch {i+1}/{len(test_dataloader)} | PSNR: {cur_psnr:.2f} dB | SSIM: {cur_ssim:.4f}"
        )
    time_end = time.time()
    total_time = time_end - time_start
    print(
        f"\n--> [Résultat final {algo}] Mean PSNR: {np.mean(psnrs):.2f} dB | Mean SSIM: {np.mean(ssims):.4f}"
    )
    print(
        f"Total time: {total_time:.2f} seconds"
    )
    with open(os.path.join(save_dir, f"{algo}_{problem_type}_metrics.txt"), "w") as f:
        f.write(
            f"Mean PSNR: {np.mean(psnrs):.2f} dB | Mean SSIM: {np.mean(ssims):.4f}\n"
        )
        f.write(
            f"Input Mean PSNR: {np.mean(input_psnr):.2f} dB | Input Mean SSIM: {np.mean(input_ssim):.4f}\n"
        )

# ---------------------------
# Grid-Search Routine pour DPS
# ---------------------------
def run_dps_grid_search():
    # 20 valeurs de poids réparties sur une échelle logarithmique [0.1, 10.0]
    weights = np.logspace(np.log10(0.1), np.log10(10.0), num=20)

    sigmas = [5/255]
    problems = ["deblurring"]

    psnr_metric = dinv.metric.PSNR()
    denoiser = GSDRUNetWrapper(device=device)
    denoiser.eval()

    best_configurations = {}

    for prob in problems:
        for sigma in sigmas:
            sigma_str = f"{int(round(sigma*255))}/255"
            config_label = f"Problème: {prob.capitalize():<10} | Bruit: {sigma_str:<5}"

            print("\n" + "=" * 65)
            print(f"Lancement Grid-Search DPS | {config_label}")
            print("=" * 65)

            physics, test_dataloader, val_dataloader = setup_physics(
                problem_type=prob, sigma=sigma
            )

            best_psnr = -float("inf")
            best_weight = None

            # On teste chaque poids sur l'ensemble du jeu de validation
            for idx, w in enumerate(weights, 1):
                model = dinv.sampling.DPS(
                    denoiser=denoiser,
                    schedule="vp",
                    weight=float(w),
                    num_steps=200,
                    alpha=1.0,  # 1.0 (DDPM stochastique) stabilise fortement les gradients
                    verbose=False,
                    device=device,
                    dtype=torch.float64,
                    minus_one_one=False,
                    rng=torch.Generator(device=device).manual_seed(seed),
                )

                batch_psnrs = []

                # Évaluation du poids w sur TOUS les batchs de val_dataloader
                for batch in val_dataloader:
                    if isinstance(batch, (list, tuple)):
                        x_true = batch[0].to(device)
                        y = batch[1].to(device)
                        params = batch[2] if len(batch) > 2 else {}
                    elif isinstance(batch, dict):
                        x_true = batch["x"].to(device)
                        y = batch["y"].to(device)
                        params = batch.get("params", {})
                    else:
                        raise ValueError("Format de batch non reconnu")

                    # Réassignation directe des paramètres de dégradation pour ce batch
                    if prob == "deblurring":
                        if "filter" in params:
                            physics.filter = params["filter"].to(device)
                    elif prob == "inpainting":
                        if "mask" in params:
                            physics.mask = params["mask"].to(device)

                    with torch.no_grad():
                        x_hat = model(y, physics, seed=seed)

                    x_hat = torch.clamp(x_hat, 0, 1)
                    batch_psnr = psnr_metric(x_true, x_hat).mean().item()
                    batch_psnrs.append(batch_psnr)

                    print(
                        f"  Batch {len(batch_psnrs)}/{len(val_dataloader)} | Poids = {w:.4f} | PSNR: {batch_psnr:.2f} dB"
                    )

                # PSNR moyen sur tout le jeu de validation pour ce poids w
                mean_psnr_for_w = np.mean(batch_psnrs)

                print(
                    f"[{idx:02d}/20] Poids = {w:.4f}  --->  PSNR moyen (val) = {mean_psnr_for_w:.2f} dB"
                )

                if mean_psnr_for_w > best_psnr:
                    best_psnr = mean_psnr_for_w
                    best_weight = w

            best_configurations[config_label] = {
                "best_weight": best_weight,
                "best_psnr": best_psnr,
            }

    # ---------------------------
    # Bilan Final
    # ---------------------------
    print("\n" + "#" * 65)
    print("      RÉSULTATS DE LA GRID SEARCH (MEILLEURES COMBINAISONS)      ")
    print("#" * 65)
    print(f"{'Configuration':<42} | {'Meilleur Poids':<14} | {'PSNR (dB)':<8}")
    print("-" * 68)
    for cfg, res in best_configurations.items():
        print(
            f"{cfg:<42} | {res['best_weight']:<14.4f} | {res['best_psnr']:<8.2f}"
        )
    print("#" * 65)

if __name__ == "__main__":
    # Test : RAM sur deblurring
    # run_reconstruction(
    #     problem_type="inpainting",
    #     algo="DDRM",
    #     save_dir="results/DDRM_inpainting/sigma_1/",
    #     sigma=1. / 255.0
    # )

    # run_reconstruction(
    #         problem_type="inpainting",
    #         algo="DDRM",
    #         save_dir="results/DDRM_inpainting/sigma_5/",
    #         sigma=5. / 255.0
    #     )

    # Test : DPS sur deblurring
    run_reconstruction(
        problem_type="inpainting",
        algo="DPS",
        save_dir="results/DPS_inpainting/sigma_1/",
        sigma=1.0 / 255.0,
        DPS_weight=4.8,
    )

    run_reconstruction(
            problem_type="inpainting",
            algo="DPS",
            save_dir="results/DPS_inpainting/sigma_5/",
            sigma=5.0 / 255.0,
            DPS_weight=4.8,
        )

    # run_dps_grid_search()