import os
import re
import ast

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PARAMÈTRES À MODIFIER
# =============================================================================

# Dossiers contenant les résultats
RESULT_DIRS = [
    "Unrolling_comparison/MRI/DEQ/theta_0",
    "Unrolling_comparison/MRI/DEQ/theta_001",
    "Unrolling_comparison/MRI/DEQ/theta_01",
    "Unrolling_comparison/MRI/DEQ/run_paper/DEQ_RISP_maxiter_100_plus_B50",
    "Unrolling_comparison/MRI/DEQ/theta_05",
]

# Nom des fichiers dans chaque dossier
RESULTS_FILE = "results.txt"
TIME_FILE = "test_times.txt"


# -----------------------------------------------------------------------------
# Paramètres du graphique
# -----------------------------------------------------------------------------

PSNR_COLOR = "C0"
TIME_COLOR = "C1"

LINE_WIDTH = 2.5
MARKER_SIZE = 7
MARKER = "o"

AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 13
LEGEND_FONT_SIZE = 13

FIGSIZE = (8, 5)

AXIS_WIDTH = 1.5

SHOW_GRID = False

# Valeur de alpha à mettre en évidence
ALPHA_DEFAULT = 0.2


# =============================================================================
# FONCTIONS DE LECTURE
# =============================================================================

def read_results_file(filepath):
    """
    Lit results.txt et extrait theta_interpol et test_PSNR.
    """
    theta_interpol = None
    psnr = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("theta_interpol:"):
                theta_interpol = float(line.split(":", 1)[1].strip())

            elif line.startswith("test_PSNR:"):
                psnr = float(line.split(":", 1)[1].strip())

    if theta_interpol is None:
        raise ValueError(
            f"'theta_interpol' introuvable dans {filepath}"
        )

    if psnr is None:
        raise ValueError(
            f"'test_PSNR' introuvable dans {filepath}"
        )

    return theta_interpol, psnr


def read_inference_time(filepath):
    """
    Lit le fichier contenant par exemple :

        Time per iteration: [0.50, 0.42, ...]

    et retourne la somme des temps.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Cherche la liste entre crochets
    match = re.search(
        r"Time per iteration:\s*(\[.*?\])",
        content,
        re.DOTALL,
    )

    if match is None:
        raise ValueError(
            f"'Time per iteration' introuvable dans {filepath}"
        )

    time_list = ast.literal_eval(match.group(1))

    return sum(time_list)


# =============================================================================
# LECTURE DES DONNÉES
# =============================================================================

alphas = []
psnrs = []
inference_times = []

for directory in RESULT_DIRS:

    results_path = os.path.join(directory, RESULTS_FILE)
    time_path = os.path.join(directory, TIME_FILE)

    if not os.path.isfile(results_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {results_path}"
        )

    if not os.path.isfile(time_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {time_path}"
        )

    alpha, psnr = read_results_file(results_path)
    inference_time = read_inference_time(time_path)

    alphas.append(alpha)
    psnrs.append(psnr)
    inference_times.append(inference_time)


# =============================================================================
# TRI DES RÉSULTATS
# =============================================================================

data = sorted(
    zip(alphas, psnrs, inference_times)
)

alphas = np.array([x[0] for x in data])
psnrs = np.array([x[1] for x in data])
inference_times = np.array([x[2] for x in data])


# =============================================================================
# AFFICHAGE DES RÉSULTATS DANS LE TERMINAL
# =============================================================================

print("\nResults:")
print("-" * 55)
print(f"{'alpha':>10} {'PSNR (dB)':>15} {'Time (s)':>15}")
print("-" * 55)

for alpha, psnr, time in zip(
    alphas,
    psnrs,
    inference_times,
):
    print(
        f"{alpha:>10.3f} "
        f"{psnr:>15.3f} "
        f"{time:>15.3f}"
    )

print("-" * 55)


# =============================================================================
# POSITIONS ARTIFICIELLES SUR L'AXE X
# =============================================================================
#
# Les valeurs réelles de alpha ne déterminent plus la distance horizontale.
#
# Exemple :
#
#   alpha réel :       0    0.01    0.1    0.2    0.5
#   position affichée: 0      1      2      3      4
#
# Cela permet notamment d'éloigner artificiellement 0 et 0.01.
# =============================================================================

x_positions = np.arange(len(alphas))


# =============================================================================
# TRACÉ
# =============================================================================

fig, ax1 = plt.subplots(figsize=FIGSIZE)


# -----------------------------------------------------------------------------
# AXE DE GAUCHE : PSNR
# -----------------------------------------------------------------------------

ax1.plot(
    x_positions,
    psnrs,
    color=PSNR_COLOR,
    linewidth=LINE_WIDTH,
    marker=MARKER,
    markersize=MARKER_SIZE,
)

ax1.set_xlabel(
    r"$\alpha$",
    fontsize=AXIS_LABEL_SIZE,
)

ax1.set_ylabel(
    "PSNR (dB)",
    fontsize=AXIS_LABEL_SIZE,
    color=PSNR_COLOR,
    labelpad=-25,
)

# Ticks uniquement aux valeurs de alpha observées
ax1.set_xticks(x_positions)

ax1.set_xticklabels(
    [f"{a:g}" for a in alphas],
    fontsize=TICK_LABEL_SIZE,
)

ax1.tick_params(
    axis="y",
    labelsize=TICK_LABEL_SIZE,
    labelcolor=PSNR_COLOR,
)

ax1.tick_params(
    axis="x",
    labelsize=TICK_LABEL_SIZE,
)


ax1.set_yticks([27.7, 28.4])

# Couleur et épaisseur de l'axe gauche
ax1.spines["left"].set_color(PSNR_COLOR)
ax1.spines["left"].set_linewidth(AXIS_WIDTH)


# -----------------------------------------------------------------------------
# LIGNE VERTICALE POUR ALPHA = 0.2
# -----------------------------------------------------------------------------

# On cherche la position artificielle correspondant à alpha = 0.2
alpha_default_indices = np.where(
    np.isclose(alphas, ALPHA_DEFAULT)
)[0]

if len(alpha_default_indices) > 0:

    alpha_default_position = x_positions[
        alpha_default_indices[0]
    ]

    ax1.axvline(
        alpha_default_position,
        color="black",
        linewidth=1.5,
        linestyle="--",
    )


# -----------------------------------------------------------------------------
# AXE DE DROITE : TEMPS D'INFÉRENCE
# -----------------------------------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    x_positions,
    inference_times,
    color=TIME_COLOR,
    linewidth=LINE_WIDTH,
    marker=MARKER,
    markersize=MARKER_SIZE,
)

ax2.set_ylabel(
    "Inference time (s)",
    fontsize=AXIS_LABEL_SIZE,
    color=TIME_COLOR,
    labelpad=-20,
)

ax2.tick_params(
    axis="y",
    labelsize=TICK_LABEL_SIZE,
    labelcolor=TIME_COLOR,
)

ax2.set_yticks([70, 150])

ax2.spines["right"].set_color(TIME_COLOR)
ax2.spines["right"].set_linewidth(AXIS_WIDTH)


# =============================================================================
# GRILLE
# =============================================================================

if SHOW_GRID:
    ax1.grid(
        True,
        axis="y",
        alpha=0.3,
    )


# =============================================================================
# LÉGENDE
# =============================================================================

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# ax1.legend(
#     lines1 + lines2,
#     labels1 + labels2,
#     fontsize=LEGEND_FONT_SIZE,
# )


# =============================================================================
# FINITION
# =============================================================================

plt.tight_layout()

plt.savefig(
    "Figures/alpha_ablation.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()