import re
import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(filepath):
    pattern = re.compile(
        r"Epoch (\d+):.*?"
        r"Train MSE=([0-9.eE+-]+), Val MSE=([0-9.eE+-]+), "
        r"Train PSNR=([0-9.eE+-]+) dB, Val PSNR=([0-9.eE+-]+) dB, "
        r"Train SSIM=([0-9.eE+-]+), Val SSIM=([0-9.eE+-]+), .*?"
        r"Time Epoch=([0-9.eE+-]+) sec"
    )

    epochs, val_mse, val_psnr, val_ssim, times = [], [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                val_mse.append(float(match.group(3)))
                val_psnr.append(float(match.group(5)))
                val_ssim.append(float(match.group(7)))
                times.append(float(match.group(8)))

    return {
        "epochs": np.array(epochs),
        "mse": np.array(val_mse),
        "psnr": np.array(val_psnr),
        "ssim": np.array(val_ssim),
        "time": np.cumsum(times)
    }


def plot_metric(
    data,
    labels,
    x_key,
    y_key,
    xlabel,
    ylabel,
    filename,
    linestyles=None,
    colors=None,
    markers=None
):
    plt.figure(figsize=(6, 5))

    for i, (d, label) in enumerate(zip(data, labels)):
        plt.plot(
            d[x_key],
            d[y_key],
            label=label,
            linestyle=linestyles[i] if linestyles else "-",
            # color=colors[i] if colors else None,
            # marker=markers[i] if markers else None,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("Figures/" + filename, dpi=300)
    plt.close()

def plot_psnr_time(data, labels, filename, linestyles, colors, markers):
    plt.figure(figsize=(6, 3.5))

    for i, (d, label) in enumerate(zip(data, labels)):
        plt.plot(
            d["time"],
            d["psnr"],
            label=label,
            linestyle=linestyles[i],
            linewidth=3,
            # color=colors[i],
            # marker=markers[i],
            # markevery=0.1
        )
    
    plt.axvline(x=4.8*10**4, color='gray', linestyle='--')

    # ---- Axes labels (identique à ton code) ----
    plt.xlabel("Time (s)", fontsize=16, labelpad=-10)
    plt.ylabel("PSNR (dB)", fontsize=16, labelpad=-25)

    # ---- Ticks fixés (spécifiques) ----
    plt.xticks([3*10**3, 10**5], [r"$3 \times 10^3$", r"$10^5$"], fontsize=14)
    plt.yticks([27.5, 28.7], fontsize=14)

    plt.xlim(left=3000, right=100000)
    plt.ylim(bottom=27.4, top=28.7)

    # plt.xscale('log')

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/" + filename, dpi=300)
    plt.close()


def plot_metrics(log_files, labels, colors, linestyles, markers):
    data = [parse_log_file(f) for f in log_files]

    # ----- Version générique -----
    plot_metric(data, labels, "epochs", "mse",
                "Epoch", "MSE", "mse_vs_epochs.pdf",
                linestyles, colors, markers)

    plot_metric(data, labels, "epochs", "psnr",
                "Epoch", "PSNR (dB)", "psnr_vs_epochs.pdf",
                linestyles, colors, markers)

    plot_metric(data, labels, "epochs", "ssim",
                "Epoch", "SSIM", "ssim_vs_epochs.pdf",
                linestyles, colors, markers)

    plot_metric(data, labels, "time", "mse",
                "Time (s)", "MSE", "mse_vs_time.pdf",
                linestyles, colors, markers)

    plot_metric(data, labels, "time", "ssim",
                "Time (s)", "SSIM", "ssim_vs_time.pdf",
                linestyles, colors, markers)

    # ----- Version custom PSNR -----
    plot_psnr_time(data, labels,
                   "psnr_vs_time.pdf",
                   linestyles, colors, markers)


if __name__ == "__main__":
    log_files = [
        "Unrolling_comparison/MRI/DEQ/ELDER_maxiter_100/training_stats.txt",
        "Unrolling_comparison/MRI/DEQ/ELDER_maxiter_200/training_stats.txt",
        "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_100_plus/training_stats.txt",
        "Unrolling_comparison/MRI/DEQ/DEQ_RISP_maxiter_200_learn_all/training_stats.txt",
    ]
    colors = [
    "blue",  # presque noir
    "lightblue",  # bleu atténué
    "lightcoral",  # rouge doux
    "lightcyan",  # cyan/gris
]
    linestyles = ['--', '--', '-', '-']
    markers = ['o', 's', 'o', 's']

    labels = [
        "DEQ (100)",
        "DEQ (200)",
        "i-DEQ (100)",
        "i-DEQ (200)",
    ]

    plot_metrics(log_files, labels, colors, linestyles, markers)