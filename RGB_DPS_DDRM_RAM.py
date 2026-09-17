import os
import random
import argparse
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

DEFAULT_GPU = "0"

img_size = (3, 320, 320)
transform = transforms.ToTensor()


def setup_device(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    print("CUDA unavailable, using CPU.")
    return torch.device("cpu")


def setup_seed(seed, device):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

    rng_device = "cuda:0" if device.type == "cuda" else "cpu"
    return torch.Generator(device=rng_device).manual_seed(seed)


def load_folder_as_tensor(root):
    root = Path(root)
    images = []

    for path in sorted(root.glob("*")):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

    if not images:
        raise FileNotFoundError(
            f"No images found in directory: {root}"
        )

    return torch.stack(images)


def get_h5_file_from_dir(dataset_dir):
    dataset_path = Path(dataset_dir)

    if dataset_path.is_dir():
        h5_files = list(dataset_path.glob("*.h5"))

        if not h5_files:
            raise FileNotFoundError(
                f"No .h5 file found in {dataset_dir}"
            )

        return str(h5_files[0])

    return str(dataset_path)


def setup_physics(
    problem_type="deblurring",
    sigma=1.0 / 255.0,
    device=None,
    rng=None,
):
    print(
        f"\n--- Loading BSDS500 images for {problem_type} ---"
    )

    train_tensor = load_folder_as_tensor(
        "DATA/BSDS500/train"
    )

    val_tensor = load_folder_as_tensor(
        "DATA/BSDS500/val"
    )

    test_tensor = load_folder_as_tensor(
        "DATA/BSDS500/test"
    )

    train_subset = TensorDataset(
        train_tensor,
        train_tensor,
    )

    val_subset = TensorDataset(
        val_tensor,
        val_tensor,
    )

    test_subset = TensorDataset(
        test_tensor,
        test_tensor,
    )

    if problem_type == "deblurring":
        psf_size = 21

        physics_generator = (
            dinv.physics.generator.MotionBlurGenerator(
                (psf_size, psf_size),
                device=device,
                rng=rng,
                dtype=torch.float32,
                sigma=1.0,
                l=0.5,
            )
        )

        physics = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=physics_generator.step()["filter"],
            device=device,
        )

        physics.noise_model = dinv.physics.GaussianNoise(
            sigma=sigma
        )

        save_dir = "DATA/datasets/blur"

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
            noise_model=dinv.physics.GaussianNoise(
                sigma=sigma
            ),
            device=device,
        )

        save_dir = (
            "DATA/datasets" + f"inpainting_{split_ratio}"
        )

    else:
        raise ValueError(
            f"Unknown problem type: {problem_type}"
        )

    print(
        f"Generating HDF5 dataset in: {save_dir}"
    )

    generated_path = dinv.datasets.generate_dataset(
        train_dataset=train_subset,
        test_dataset=test_subset,
        val_dataset=val_subset,
        physics=physics,
        physics_generator=physics_generator,
        save_physics_generator_params=True,
        overwrite_existing=True,
        device=device,
        save_dir=str(save_dir),
        batch_size=4,
    )

    h5_path = get_h5_file_from_dir(generated_path)

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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    return physics, test_dataloader, val_dataloader


class GSDRUNetWrapper(torch.nn.Module):
    def __init__(
        self,
        ckpt_path=None,
        device="cuda:0",
    ):
        super().__init__()

        self.net = dinv.models.DRUNet(device=device)

    def forward(self, x, sigma):
        return self.net(x, sigma)


def tensor_to_numpy(tensor):
    return (
        tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


def run_reconstruction(
    problem_type="deblurring",
    algo="DDRM",
    ckpt_path=None,
    save_dir="results",
    sigma=1.0 / 255.0,
    dps_weight=5.0,
    device=None,
    rng=None,
    seed=42,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    physics, test_dataloader, val_dataloader = setup_physics(
        problem_type=problem_type,
        sigma=sigma,
        device=device,
        rng=rng,
    )

    denoiser = GSDRUNetWrapper(
        ckpt_path=ckpt_path,
        device=device,
    )

    denoiser.eval()

    if algo.upper() == "DDRM":
        sigmas = np.linspace(
            1.0,
            0.01,
            300,
        )

        model = dinv.sampling.DDRM(
            denoiser=denoiser,
            sigmas=sigmas,
            etab=1.0,
            verbose=True,
        ).to(device)

    elif algo.upper() == "DPS":
        model = dinv.sampling.DPS(
            denoiser,
            schedule="ve",
            weight=dps_weight,
            num_steps=300,
            alpha=1.0,
            verbose=True,
            device=device,
            dtype=torch.float64,
            minus_one_one=False,
            rng=torch.Generator(
                device=device
            ).manual_seed(seed),
        )

    elif algo.upper() == "RAM":
        model = dinv.models.RAM(
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algo}"
        )

    psnr_metric = dinv.metric.PSNR()
    ssim_metric = dinv.metric.SSIM()

    psnrs = []
    ssims = []
    input_psnr = []
    input_ssim = []

    saved_count = 0

    print(
        f"\n=== Running {algo} | "
        f"Problem: {problem_type} | "
        f"Noise: {sigma * 255:.0f}/255 ==="
    )

    time_start = time.time()

    for i, batch in enumerate(test_dataloader):
        if isinstance(batch, (list, tuple)):
            x_true = batch[0].to(device)
            y = batch[1].to(device)
            params = (
                batch[2]
                if len(batch) > 2
                else {}
            )

        elif isinstance(batch, dict):
            x_true = batch["x"].to(device)
            y = batch["y"].to(device)
            params = batch.get("params", {})

        else:
            raise ValueError(
                "Unsupported batch format"
            )

        batch_size = x_true.shape[0]

        if batch_size == 1:
            if problem_type == "deblurring":
                if "filter" in params:
                    physics.filter = (
                        params["filter"].to(device)
                    )

            elif problem_type == "inpainting":
                if "mask" in params:
                    physics.mask = (
                        params["mask"].to(device)
                    )

        with torch.no_grad():
            if algo.upper() == "DPS":
                x_hat = model(
                    y,
                    physics,
                    seed=seed,
                )
            else:
                x_hat = model(
                    y,
                    physics,
                )

        x_hat = torch.clamp(
            x_hat,
            0,
            1,
        )

        cur_psnr = (
            psnr_metric(
                x_true,
                x_hat,
            )
            .mean()
            .item()
        )

        cur_ssim = (
            ssim_metric(
                x_true,
                x_hat,
            )
            .mean()
            .item()
        )

        curr_input_psnr = (
            psnr_metric(
                x_true,
                y,
            )
            .mean()
            .item()
        )

        curr_input_ssim = (
            ssim_metric(
                x_true,
                y,
            )
            .mean()
            .item()
        )

        psnrs.append(cur_psnr)
        ssims.append(cur_ssim)
        input_psnr.append(curr_input_psnr)
        input_ssim.append(curr_input_ssim)

        for b in range(batch_size):
            if saved_count >= 5:
                break

            single_psnr = (
                psnr_metric(
                    x_true[b:b + 1],
                    x_hat[b:b + 1],
                )
                .item()
            )

            single_ssim = (
                ssim_metric(
                    x_true[b:b + 1],
                    x_hat[b:b + 1],
                )
                .item()
            )

            single_input_psnr = (
                psnr_metric(
                    x_true[b:b + 1],
                    y[b:b + 1],
                )
                .item()
            )

            single_input_ssim = (
                ssim_metric(
                    x_true[b:b + 1],
                    y[b:b + 1],
                )
                .item()
            )

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(12, 4),
            )

            axes[0].imshow(
                tensor_to_numpy(x_true[b])
            )

            axes[0].set_title(
                "Original ($x$)"
            )

            axes[0].axis("off")

            axes[1].imshow(
                tensor_to_numpy(y[b])
            )

            axes[1].set_title(
                f"Degraded ($y$)\n"
                f"PSNR: {single_input_psnr:.2f} dB | "
                f"SSIM: {single_input_ssim:.4f}"
            )

            axes[1].axis("off")

            axes[2].imshow(
                tensor_to_numpy(x_hat[b])
            )

            axes[2].set_title(
                f"{algo} ($\\hat{{x}}$)\n"
                f"PSNR: {single_psnr:.2f} dB | "
                f"SSIM: {single_ssim:.4f}"
            )

            axes[2].axis("off")

            plt.tight_layout()

            out_path = (
                save_dir
                / f"{algo}_{problem_type}_"
                f"fig_{saved_count + 1}.pdf"
            )

            plt.savefig(
                out_path,
                bbox_inches="tight",
                dpi=300,
            )

            plt.close(fig)

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )

            ax.imshow(
                tensor_to_numpy(x_hat[b])
            )

            ax.axis("off")

            out_path_single = (
                save_dir
                / f"{algo}_{problem_type}_"
                f"single_{saved_count + 1}.pdf"
            )

            plt.title(
                f"PSNR: {single_psnr:.2f} dB | "
                f"SSIM: {single_ssim:.4f}"
            )

            plt.savefig(
                out_path_single,
                bbox_inches="tight",
                dpi=300,
            )

            plt.close(fig)

            print(
                f"  -> Figure {saved_count + 1}/5 saved: "
                f"{out_path_single}"
            )

            saved_count += 1

        print(
            f"Batch {i + 1}/{len(test_dataloader)} | "
            f"PSNR: {cur_psnr:.2f} dB | "
            f"SSIM: {cur_ssim:.4f}"
        )

    total_time = time.time() - time_start

    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    mean_input_psnr = np.mean(input_psnr)
    mean_input_ssim = np.mean(input_ssim)

    print(
        f"\nFinal result [{algo}] "
        f"Mean PSNR: {mean_psnr:.2f} dB | "
        f"Mean SSIM: {mean_ssim:.4f}"
    )

    print(
        f"Input Mean PSNR: {mean_input_psnr:.2f} dB | "
        f"Input Mean SSIM: {mean_input_ssim:.4f}"
    )

    print(
        f"Total time: {total_time:.2f} seconds"
    )

    metrics_path = (
        save_dir
        / f"{algo}_{problem_type}_metrics.txt"
    )

    with open(metrics_path, "w") as f:
        f.write(
            f"Algorithm: {algo}\n"
            f"Problem: {problem_type}\n"
            f"Noise level: {sigma * 255:.0f}/255\n"
            f"DPS weight: {dps_weight}\n"
            f"Mean PSNR: {mean_psnr:.2f} dB\n"
            f"Mean SSIM: {mean_ssim:.4f}\n"
            f"Input Mean PSNR: {mean_input_psnr:.2f} dB\n"
            f"Input Mean SSIM: {mean_input_ssim:.4f}\n"
            f"Total time: {total_time:.2f} seconds\n"
        )


def run_dps_grid_search(
    device,
    rng,
    seed=42,
):
    weights = np.logspace(
        np.log10(0.1),
        np.log10(10.0),
        num=20,
    )

    sigmas = [5 / 255]
    problems = ["deblurring"]

    psnr_metric = dinv.metric.PSNR()

    denoiser = GSDRUNetWrapper(
        device=device
    )

    denoiser.eval()

    best_configurations = {}

    for prob in problems:
        for sigma in sigmas:
            sigma_level = int(round(sigma * 255))

            config_label = (
                f"Problem: {prob} | "
                f"Noise: {sigma_level}/255"
            )

            print("\n" + "=" * 65)
            print(
                f"DPS grid search | {config_label}"
            )
            print("=" * 65)

            physics, _, val_dataloader = setup_physics(
                problem_type=prob,
                sigma=sigma,
                device=device,
                rng=rng,
            )

            best_psnr = -float("inf")
            best_weight = None

            for idx, w in enumerate(weights, 1):
                model = dinv.sampling.DPS(
                    denoiser=denoiser,
                    schedule="vp",
                    weight=float(w),
                    num_steps=200,
                    alpha=1.0,
                    verbose=False,
                    device=device,
                    dtype=torch.float64,
                    minus_one_one=False,
                    rng=torch.Generator(
                        device=device
                    ).manual_seed(seed),
                )

                batch_psnrs = []

                for batch in val_dataloader:
                    if isinstance(batch, (list, tuple)):
                        x_true = batch[0].to(device)
                        y = batch[1].to(device)
                        params = (
                            batch[2]
                            if len(batch) > 2
                            else {}
                        )

                    elif isinstance(batch, dict):
                        x_true = batch["x"].to(device)
                        y = batch["y"].to(device)
                        params = batch.get(
                            "params",
                            {},
                        )

                    else:
                        raise ValueError(
                            "Unsupported batch format"
                        )

                    if prob == "deblurring":
                        if "filter" in params:
                            physics.filter = (
                                params["filter"].to(device)
                            )

                    elif prob == "inpainting":
                        if "mask" in params:
                            physics.mask = (
                                params["mask"].to(device)
                            )

                    with torch.no_grad():
                        x_hat = model(
                            y,
                            physics,
                            seed=seed,
                        )

                    x_hat = torch.clamp(
                        x_hat,
                        0,
                        1,
                    )

                    batch_psnr = (
                        psnr_metric(
                            x_true,
                            x_hat,
                        )
                        .mean()
                        .item()
                    )

                    batch_psnrs.append(
                        batch_psnr
                    )

                    print(
                        f"  Batch "
                        f"{len(batch_psnrs)}/"
                        f"{len(val_dataloader)} | "
                        f"Weight = {w:.4f} | "
                        f"PSNR = {batch_psnr:.2f} dB"
                    )

                mean_psnr_for_w = np.mean(
                    batch_psnrs
                )

                print(
                    f"[{idx:02d}/20] "
                    f"Weight = {w:.4f} -> "
                    f"Validation PSNR = "
                    f"{mean_psnr_for_w:.2f} dB"
                )

                if mean_psnr_for_w > best_psnr:
                    best_psnr = mean_psnr_for_w
                    best_weight = w

            best_configurations[
                config_label
            ] = {
                "best_weight": best_weight,
                "best_psnr": best_psnr,
            }

    print("\n" + "#" * 65)
    print("DPS GRID SEARCH RESULTS")
    print("#" * 65)

    for cfg, res in best_configurations.items():
        print(
            f"{cfg} | "
            f"Best weight = {res['best_weight']:.4f} | "
            f"PSNR = {res['best_psnr']:.2f} dB"
        )

    print("#" * 65)


def build_save_dir(
    algo,
    problem_type,
    sigma,
    dps_weight=None,
    grid_search=False,
):
    noise_level = int(round(sigma * 255))

    if algo.upper() in ["DDRM", "DPS"]:
        path_root = Path("diffusion_models")
    else:
        path_root = Path("e2e_models")
    path = (
        path_root
        / algo.upper()
        / problem_type
        / f"noise_{noise_level}"
    )

    if algo.upper() == "DPS" and dps_weight is not None:
        path = path / f"weight_{dps_weight:g}"

    if grid_search:
        path = path / "grid_search"

    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run image reconstruction with "
            "DDRM, DPS or RAM."
        )
    )

    parser.add_argument(
        "--problem",
        choices=[
            "deblurring",
            "inpainting",
        ],
        default="inpainting",
    )

    parser.add_argument(
        "--algo",
        choices=[
            "DDRM",
            "DPS",
            "RAM",
        ],
        default="DPS",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--noise-level",
        type=float,
        default=1.0,
        help="Noise level expressed as x/255.",
    )

    parser.add_argument(
        "--dps-weight",
        type=float,
        default=4.8,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_GPU,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--grid-search",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = setup_device(args.gpu)

    rng = setup_seed(
        args.seed,
        device,
    )

    if args.sigma is not None:
        sigma = args.sigma
    else:
        sigma = args.noise_level / 255.0

    if args.grid_search:
        run_dps_grid_search(
            device=device,
            rng=rng,
            seed=args.seed,
        )

    else:
        if args.save_dir is None:
            save_dir = build_save_dir(
                algo=args.algo,
                problem_type=args.problem,
                sigma=sigma,
                dps_weight=(
                    args.dps_weight
                    if args.algo.upper() == "DPS"
                    else None
                ),
            )
        else:
            save_dir = Path(args.save_dir)

        print("\nExperiment configuration:")
        print(f"  Algorithm    : {args.algo}")
        print(f"  Problem      : {args.problem}")
        print(
            f"  Noise level  : "
            f"{sigma * 255:.0f}/255"
        )
        print(f"  DPS weight   : {args.dps_weight}")
        print(f"  GPU          : {args.gpu}")
        print(f"  Device       : {device}")
        print(f"  Output       : {save_dir}")

        run_reconstruction(
            problem_type=args.problem,
            algo=args.algo,
            ckpt_path=args.ckpt,
            save_dir=save_dir,
            sigma=sigma,
            dps_weight=args.dps_weight,
            device=device,
            rng=rng,
            seed=args.seed,
        )