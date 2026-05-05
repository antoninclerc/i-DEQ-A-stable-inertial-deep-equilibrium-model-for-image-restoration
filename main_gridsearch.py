import argparse

from grid_search import GridSearch, build_config
from gen_data import get_dataloaders
from utils import str2bool, normalize_problem

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

    elif problem == "rician":
        return {
            "sigma": 0.1,
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

    parser.add_argument("--problem", required=True, choices=["mri", "inpainting", "rician"])
    parser.add_argument("--dc", required=True, choices=["grad", "prox"])
    parser.add_argument("--accelerated", type=str2bool, default=True)

    args = parser.parse_args()

    # -------------------------
    # Normalisation
    # -------------------------
    problem = normalize_problem(args.problem)
    DC_type = args.dc
    accelerated = args.accelerated

    print(f"Running: {problem} | {DC_type} | accelerated={accelerated}")

    # -------------------------
    # Config grid
    # -------------------------
    grid_config = build_config(problem, DC_type, accelerated)

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
