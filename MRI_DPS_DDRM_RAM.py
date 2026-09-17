import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import deepinv as dinv

from torch.utils.data import DataLoader
from deepinv.datasets import FastMRISliceDataset

from networks.GSPnP_MRI import GSDRUNet_MRI
from utils import ifft2c, PSNR, SSIM, image_2ch_to_magnitude


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        choices=["DPS", "DDRM", "RAM"],
        required=True,
    )

    parser.add_argument(
        "--acceleration",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sigma-level",
        type=float,
        default=1.0,
        help="Noise level expressed as sigma * 255.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--dps-weight",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[320, 320],
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/MRI",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="DATA/MRI",
    )

    parser.add_argument(
        "--denoiser-ckpt",
        type=str,
        default="networks/GSDRUNet_grayscale_torch.ckpt",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
    )

    return parser.parse_args()


def setup_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return torch.Generator(device=device).manual_seed(seed)


def build_save_dir(args):
    noise_level = int(round(args.sigma_level))

    if args.save_dir is not None:
        return Path(args.save_dir)
    if args.algo == "RAM":
        return (
            Path("E2E")
            / args.algo
            / "MRI"
            / f"acceleration_{args.acceleration}"
            / f"noise_{noise_level}"
        )
    else:
        return (
            Path("diffusion_models")
            / args.algo
            / "MRI"
            / f"acceleration_{args.acceleration}"
            / f"noise_{noise_level}"
        )


def build_dataset(
    data_root,
    dataset_dir,
    img_size,
    acceleration,
    sigma,
    rng,
    device,
):
    data_root = Path(data_root)
    dataset_dir = Path(dataset_dir)

    train_root = data_root / "singlecoil_train"
    val_root = data_root / "singlecoil_val"
    test_root = data_root / "singlecoil_test"

    train_dataset_raw = FastMRISliceDataset(
        root=str(train_root),
        slice_index="middle",
    )

    val_dataset_raw = FastMRISliceDataset(
        root=str(val_root),
        slice_index="middle",
    )

    test_dataset_raw = FastMRISliceDataset(
        root=str(test_root),
        slice_index="middle",
    )

    train_subset = train_dataset_raw.save_simple_dataset(
        str(train_root / "fastmri_brain_singlecoil.pt"),
        pad_to_size=img_size,
    )

    val_subset = val_dataset_raw.save_simple_dataset(
        str(val_root / "fastmri_brain_singlecoil.pt"),
        pad_to_size=img_size,
    )

    test_subset = test_dataset_raw.save_simple_dataset(
        str(test_root / "fastmri_brain_singlecoil.pt"),
        pad_to_size=img_size,
    )

    physics_generator = dinv.physics.generator.GaussianMaskGenerator(
        img_size=img_size,
        acceleration=acceleration,
        rng=rng,
        device=device,
    )

    mask = physics_generator.step()["mask"]

    physics = dinv.physics.MRI(
        mask=mask,
        img_size=img_size,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    noise_level = int(round(sigma * 255))

    save_dir = (
        dataset_dir
        / f"acceleration_{acceleration}"
        / f"noise_{noise_level}"
    )

    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_subset,
        test_dataset=test_subset,
        val_dataset=val_subset,
        physics=physics,
        physics_generator=physics_generator,
        save_physics_generator_params=True,
        overwrite_existing=False,
        device=device,
        save_dir=str(save_dir),
        batch_size=4,
    )

    return dataset_path, physics


def build_model(args, device, rng):
    if args.algo == "RAM":
        model = dinv.models.RAM(
            device=device,
            pretrained=True,
        ).to(device)

        model.eval()

        return model

    denoiser = GSDRUNet_MRI(
        in_channels=1,
        out_channels=1,
        pretrained=args.denoiser_ckpt,
    ).to(device)

    denoiser.eval()

    if args.algo == "DPS":
        model = dinv.sampling.DPS(
            denoiser=denoiser,
            schedule="vp",
            num_steps=args.steps,
            weight=args.dps_weight,
            alpha=0.8,
            verbose=True,
            device=device,
            dtype=torch.float64,
            rng=rng,
            minus_one_one=False,
        ).to(device)

    elif args.algo == "DDRM":
        model = dinv.sampling.DDRM(
            denoiser,
            sigmas=torch.linspace(
                1,
                0,
                args.steps,
                device=device,
            ),
            verbose=True,
        ).to(device)

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model.eval()

    return model


def evaluate(
    model,
    physics,
    test_dataloader,
    device,
    algo,
    save_dir,
    num_images,
):
    psnr_sum = 0.0
    ssim_sum = 0.0
    psnr_input_sum = 0.0

    num_samples = 0
    reconstruction_time = 0.0

    reconstructions = []
    inputs = []
    targets = []
    psnrs = []
    ssims = []
    input_psnrs = []

    with torch.no_grad():
        for batch_target, batch_input, batch_params in test_dataloader:
            batch_target = batch_target.to(device)
            batch_input = batch_input.to(device)

            batch_target_im = image_2ch_to_magnitude(batch_target)

            physics.update(**batch_params)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            xhat = model(
                batch_input,
                physics=physics,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            reconstruction_time += end_time - start_time

            xhat_im = image_2ch_to_magnitude(xhat)

            batch_psnr = PSNR(
                xhat_im,
                batch_target_im,
            )

            batch_ssim = SSIM(
                xhat_im,
                batch_target_im,
            )

            input_im = image_2ch_to_magnitude(
                ifft2c(batch_input)
            )

            batch_psnr_input = PSNR(
                input_im,
                batch_target_im,
            )

            psnr_sum += batch_psnr.sum().item()
            ssim_sum += batch_ssim.sum().item()
            psnr_input_sum += batch_psnr_input.sum().item()

            psnrs.extend(batch_psnr.cpu().tolist())
            ssims.extend(batch_ssim.cpu().tolist())
            input_psnrs.extend(batch_psnr_input.cpu().tolist())

            reconstructions.extend(xhat_im.cpu())
            inputs.extend(input_im.cpu())
            targets.extend(batch_target_im.cpu())

            num_samples += batch_target.shape[0]

    mean_psnr = psnr_sum / num_samples
    mean_ssim = ssim_sum / num_samples
    mean_psnr_input = psnr_input_sum / num_samples

    mean_time = reconstruction_time / num_samples

    print()
    print("=" * 60)
    print(f"Algorithm            : {algo}")
    print(f"Number of images     : {num_samples}")
    print(f"PSNR input           : {mean_psnr_input:.2f} dB")
    print(f"PSNR reconstruction   : {mean_psnr:.2f} dB")
    print(f"SSIM reconstruction   : {mean_ssim:.4f}")
    print(f"Total reconstruction  : {reconstruction_time:.4f} s")
    print(f"Time / image          : {mean_time:.4f} s")
    print(f"Images / second       : {1.0 / mean_time:.2f}")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    n = min(
        num_images,
        len(reconstructions),
        len(inputs),
        len(targets),
    )

    for i in range(n):
        reconstruction = reconstructions[i]
        input_image = inputs[i]
        target = targets[i]

        psnr_rec = PSNR(
            reconstruction.unsqueeze(0),
            target.unsqueeze(0),
        ).item()

        ssim_rec = SSIM(
            reconstruction.unsqueeze(0),
            target.unsqueeze(0),
        ).item()

        psnr_in = PSNR(
            input_image.unsqueeze(0),
            target.unsqueeze(0),
        ).item()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(
            input_image.squeeze().numpy(),
            cmap="gray",
        )
        plt.title(f"Input\nPSNR: {psnr_in:.2f} dB")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(
            reconstruction.squeeze().numpy(),
            cmap="gray",
        )
        plt.title(
            f"{algo}\n"
            f"PSNR: {psnr_rec:.2f} dB, "
            f"SSIM: {ssim_rec:.4f}"
        )
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(
            target.squeeze().numpy(),
            cmap="gray",
        )
        plt.title("Target")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            save_dir / f"{i}.pdf",
            dpi=300,
        )

        plt.close()

        plt.figure(figsize=(5, 5))

        plt.imshow(
            reconstruction.squeeze().numpy(),
            cmap="gray",
        )

        plt.title(
            f"PSNR: {psnr_rec:.2f} dB / "
            f"SSIM: {ssim_rec:.4f}"
        )

        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            save_dir / f"{i}_reconstruction.pdf",
            dpi=300,
        )

        plt.close()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    img_size = tuple(args.img_size)
    sigma = args.sigma_level / 255.0

    rng = setup_seed(
        args.seed,
        device,
    )

    print("=" * 60)
    print("MRI reconstruction")
    print("=" * 60)
    print(f"Algorithm       : {args.algo}")
    print(f"Device          : {device}")
    print(f"Acceleration    : {args.acceleration}")
    print(f"Sigma           : {sigma:.6f}")
    print(f"Steps           : {args.steps}")
    print(f"Image size      : {img_size}")
    print("=" * 60)

    dataset_path, physics = build_dataset(
        data_root=args.data_root,
        dataset_dir=args.dataset_dir,
        img_size=img_size,
        acceleration=args.acceleration,
        sigma=sigma,
        rng=rng,
        device=device,
    )

    test_dataset = dinv.datasets.HDF5Dataset(
        dataset_path,
        split="test",
        load_physics_generator_params=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(
        args,
        device,
        rng,
    )

    save_dir = build_save_dir(args)

    evaluate(
        model=model,
        physics=physics,
        test_dataloader=test_dataloader,
        device=device,
        algo=args.algo,
        save_dir=save_dir,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()