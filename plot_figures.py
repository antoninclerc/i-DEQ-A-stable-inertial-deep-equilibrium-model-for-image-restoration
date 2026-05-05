import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from networks.DRUnet import GSDRUNet
from models.deep_equilibrium import DeepEquilibrium
from gen_data import get_dataloaders
from utils import PSNR


# ============================================================
# 1. EXPERIMENTS
# ============================================================

EXPERIMENTS = [
    # {"name": "DEQ-RISP (100)", 
    #  "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_100/"},
    {"name": "DEQ (100)",
     "path": "Unrolling_comparison/MRI/DEQ/ELDER_maxiter_100/"},
    {"name": "DEQ (200)",
     "path": "Unrolling_comparison/MRI/DEQ/ELDER_maxiter_200/"},
    {"name": "i-DEQ (100)", 
     "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_100_plus/"},
    {"name": "i-DEQ (200)", 
     "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_200_learn_all/"},
    {"name": "RISP",
     "path": "Unrolling_comparison/MRI/DEQ/RISP_GRAD_B_5000/"},
    #  {"name": "No Plot",
    #  "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_100_plus_e5/"},
    # {"name": "DEQ-RISP (200, alpha=0.2)", 
    #  "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_200_learn_lambda_and_tau/"},
    # {"name": "DEQ-RISP 0", 
    #  "path": "Unrolling_comparison/MRI/DEQ/DEQ_RISP_Rtheta_0.10_lambda_dc_1.00_sigma_denoiser_0.00/"},
    # {"name": "P-RISP-DEQ",
    #  "path": "Unrolling_comparison/MRI/DEQ/PROX_DEQ_RISP_Rtheta_0.10_lambda_dc_1.00_sigma_denoiser_0.00/"},
    # {"name": "RISP (B=100)",
    #  "path": "Unrolling_comparison/MRI/DEQ/RISP_GRAD_B_100/"},
    # {"name": "P-RED",
    #  "path": "Unrolling_comparison/MRI/DEQ/RED_PROX/"},
    # {"name": "P-RISP",
    #  "path": "Unrolling_comparison/MRI/DEQ/RISP_PROX/"},
    # {"name": "RED",
    #  "path": "Unrolling_comparison/MRI/DEQ/RED/"},
    # {"name": "ELDER 0",
    #  "path": "Unrolling_comparison/MRI/DEQ/ELDER_Rtheta_0.1_sigma_denoiser_0.00/"}
]
DEVICE = "cuda:1"

TARGET_ITERS = [50, 100, 150, 200, 500]



# ============================================================
# 2. LOAD CONFIG
# ============================================================

# ============================================================
# CONFIG LOADER
# ============================================================

def load_times(path_folder):
    file_path = os.path.join(path_folder, "test_times.txt")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        line = f.readline()

    # Extraction de la liste depuis la string
    # "Time per iteration: [....]"
    times_str = line.split(":", 1)[1].strip()

    try:
        times = np.array(eval(times_str))
    except:
        raise ValueError(f"Failed to parse times in {file_path}")

    return times

def cumulative_time(times):
    return np.cumsum(times)

def load_config(path_folder):
    cfg = {}

    with open(os.path.join(path_folder, "results.txt"), "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                cfg[k.strip()] = v.strip()

    def to_float(x):
        try:
            return float(x)
        except:
            return x

    def to_bool(x):
        return x.lower() in ["true", "yes", "1"]

    for k in ["lambda_Rtheta", "sigma_denoiser", "lambda_dc", "B_restart"]:
        if k in cfg:
            cfg[k] = to_float(cfg[k])

    for k in ["learn_lambda_dc", "learn_lambda_Rtheta",
              "learn_theta_interpol", "accelerated", "backtracking"]:
        if k in cfg:
            cfg[k] = to_bool(cfg[k])

    return cfg


# ============================================================
# DATASET
# ============================================================

def build_test_loader():
    _, _, test_loader, _ = get_dataloaders(
        problem_type="mri",
        config={
            "img_size": (320, 320),
            "acceleration": 8,
            "train_path": "DATA/MRI/singlecoil_train",
            "val_path": "DATA/MRI/singlecoil_val",
            "test_path": "DATA/MRI/singlecoil_test",
            "sigma": 1. / 255,
            "device": DEVICE,
        },
    )
    return test_loader


TEST_LOADER = build_test_loader()


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(cfg, path):

    CNN = GSDRUNet(
        in_channels=1,
        out_channels=1,
        pretrained="networks/GSDRUNet_grayscale_torch.ckpt",
    )

    return DeepEquilibrium(
        Network=CNN,
        problem="MRI",
        DC_type=cfg["DC_type"],
        backtracking=cfg.get("backtracking", False),
        lambda_dc=float(cfg.get("lambda_dc", 0.5)),
        lambda_Rtheta=float(cfg["lambda_Rtheta"]),
        learn_lambda_dc=cfg.get("learn_lambda_dc", False),
        learn_lambda_Rtheta=cfg.get("learn_lambda_Rtheta", False),
        gamma=0.01,
        eta=0.5,
        thresh=1e-4,
        max_iter=500,
        device=DEVICE,
        path_folder=path,
        sigma_noise=1. / 255,
        sigma_denoiser=float(cfg["sigma_denoiser"]),
        theta_interpol=0.2,
        restart=True,
        B_restart=cfg.get("B_restart", 5000),
        learn_theta_interpol=cfg.get("learn_theta_interpol", False),
    )


# ============================================================
# UTILS
# ============================================================

def extract_psnr_at_iters(psnr_curve, target_iters):
    T = len(psnr_curve)
    values = []

    for it in target_iters:
        idx = it - 1
        if idx < T:
            values.append(psnr_curve[idx])
        else:
            values.append(psnr_curve[-1])

    return np.array(values)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(exp, PSNR, energy):

    path = exp["path"]
    cache_file_psnr = os.path.join(path, "psnr_curve.npy")
    cache_file_energy = os.path.join(path, "energy_curve.npy")
    model_path = os.path.join(path, "best_model.pth")

    # -------------------------
    # CACHE
    # -------------------------
    if PSNR and os.path.exists(cache_file_psnr):
        print(f"[CACHE PSNR] {exp['name']}")
        full_psnr_curve = np.load(cache_file_psnr)
        psnr_selected = extract_psnr_at_iters(full_psnr_curve, TARGET_ITERS)
        psnr_loaded = True
    else:
        psnr_loaded = False

    if energy and os.path.exists(cache_file_energy):
        print(f"[CACHE ENERGY] {exp['name']}")
        energy_curve = np.load(cache_file_energy)
        energy_loaded = True
    else:
        energy_loaded = False

    if (PSNR and psnr_loaded) and (energy and energy_loaded):
        times = load_times(path)
        time_cum = cumulative_time(times) if times is not None else None
        return {
            "PSNR": full_psnr_curve,
            "PSNR_selected": psnr_selected,
            "Energy": energy_curve,
            "time": time_cum
        }
    
    if (PSNR and psnr_loaded) and (not energy):
        times = load_times(path)
        time_cum = cumulative_time(times) if times is not None else None
        return {
            "PSNR": full_psnr_curve,
            "PSNR_selected": psnr_selected,
            "time": time_cum
        }
    
    if (not PSNR) and (energy and energy_loaded):
        times = load_times(path)
        time_cum = cumulative_time(times) if times is not None else None
        return {
            "Energy": energy_curve,
            "time": time_cum
        }

    print(f"[RUN] {exp['name']}")

    cfg = load_config(path)
    model = build_model(cfg, path)

    if not os.path.exists(model_path):
        model_path = None

    result = model.evaluate(
        test_loader=TEST_LOADER,
        n_display=7,
        accelerated=cfg.get("accelerated", True),
        init_train=None,
        pretrained_path=model_path,
        PnP=False,
    )

    # -------------------------
    # PSNR
    # -------------------------
    if PSNR:
        full_psnr_curve = np.array(result["PSNR_list"])
        np.save(cache_file_psnr, full_psnr_curve)
        psnr_selected = extract_psnr_at_iters(full_psnr_curve, TARGET_ITERS)

    # -------------------------
    # ENERGY
    # -------------------------
    if energy:
        energy_curve = np.array(result["Energy_list"])
        np.save(cache_file_energy, energy_curve)

    times = load_times(path)
    time_cum = cumulative_time(times) if times is not None else None

    # -------------------------
    # RETURN
    # -------------------------
    out = {}
    if PSNR:
        out["PSNR"] = full_psnr_curve
        out["PSNR_selected"] = psnr_selected
        out["time"] = time_cum
    if energy:
        out["Energy"] = energy_curve
        out["time"] = time_cum

    return out


# ============================================================
# RUN ALL
# ============================================================

def run_all(PSNR, energy):
    return {exp["name"]: run_experiment(exp, PSNR, energy)
            for exp in EXPERIMENTS}


# ============================================================
# TABLE
# ============================================================

def build_psnr_table(curves):
    names = []
    table = []

    for name in curves:
        if "PSNR_selected" in curves[name]:
            names.append(name)
            table.append(curves[name]["PSNR_selected"])

    return names, np.array(table)


def save_psnr_table(names, table, filename="psnr_table.txt"):

    with open(filename, "w") as f:
        header = "Method\t" + "\t".join([f"PSNR@{it}" for it in TARGET_ITERS])
        f.write(header + "\n")

        for name, row in zip(names, table):
            row_str = "\t".join([f"{v:.4f}" for v in row])
            f.write(f"{name}\t{row_str}\n")

# ============================================================
# PLOT
# ============================================================

def plot_curves(curves, PSNR=True, energy=True):

    if PSNR:
        plt.figure(figsize=(6, 5))

        PSNR_cible = 28.2
        plt.axhline(y=PSNR_cible, color='gray', linestyle='--')

        for name in curves:
            if "PSNR" in curves[name]:
                if name == "No Plot":
                    plt.plot(curves[name]["PSNR"], color="green", linestyle="--")
                else:
                    plt.plot(curves[name]["PSNR"], label=name, linewidth=3)

        plt.xlabel("Iteration", fontsize=16, labelpad=-10)
        plt.ylabel("PSNR", fontsize=16, labelpad=-25)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks([0, 300], fontsize=14)
        plt.yticks([24., 28.2], fontsize=14)

        plt.xlim(left=0, right=400)
        plt.ylim(bottom=24, top=28.3)
        
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig("Figures/psnr_comparison.pdf", dpi=300)
        plt.close()

    if energy:
        plt.figure(figsize=(6, 3.5))

        for name in curves:
            if "Energy" in curves[name]:
                e = curves[name]["Energy"]
                plt.plot(e - np.min(e), label=name)

        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Energy", fontsize=16)
        plt.yscale("log")
        plt.legend(fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Figures/energy_comparison.pdf", dpi=300)
        plt.close()

def plot_psnr_vs_time(curves):

    plt.figure(figsize=(6, 3.5))

    x_cut = 60

    for name in curves:
        if "PSNR" in curves[name] and "time" in curves[name]:
            psnr = np.array(curves[name]["PSNR"])
            time = np.array(curves[name]["time"])

            T = min(len(psnr), len(time))
            psnr = psnr[:T]
            time = time[:T]

            linestyle = "--" if name == "DEQ (100)" or name == "DEQ (200)" else "-"
            if name == "No Plot":
                line, = plt.plot(time, psnr, color="green", linestyle=":")
                color = "green"
            else:
                line, = plt.plot(time, psnr, label=name, linewidth=3, linestyle=linestyle)
                color = line.get_color()

            # intersection at x = 60
            if np.min(time) <= x_cut <= np.max(time):
                psnr_at_cut = np.interp(x_cut, time, psnr)
                plt.plot(x_cut, psnr_at_cut, 'o', color=color, markersize=6)

    plt.axvline(x=x_cut, color='gray', linestyle='--')

    plt.xlabel("Time (s)", fontsize=16, labelpad=-10)
    plt.ylabel("PSNR", fontsize=16, labelpad=-25)
    plt.legend(fontsize=14)

    plt.xlim(left=0, right=400)
    plt.ylim(bottom=24, top=28.3)
    plt.xticks([0, 60, 300], fontsize=14)
    plt.yticks([24., 28.3], fontsize=14)

    plt.tight_layout()
    plt.savefig("Figures/psnr_vs_time_inference.pdf", dpi=300)
    plt.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    psnr = True
    energy = False
    curves = run_all(PSNR=psnr, energy=energy)

    plot_curves(curves, PSNR=psnr, energy=energy)

    names, table = build_psnr_table(curves)
    save_psnr_table(names, table)
    plot_psnr_vs_time(curves)