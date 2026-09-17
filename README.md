# i-DEQ: A Stable Inertial Deep Equilibrium Model for Image Restoration

This repository contains the implementation of **i-DEQ** and several reconstruction methods:

- Deep Equilibrium Models (DEQ)
- Unrolled networks (VarNet / MoDL)
- Diffusion-based reconstruction (DiffPIR)
- Diffusion reconstruction with **DPS / DDRM / RAM**
- Grid search for hyperparameter selection

Experiments cover **MRI reconstruction (FastMRI)**, **image inpainting (BSDS500)**, and **Rician denoising (BSDS500)**.

## 1. Installation

```bash
pip install -r requirements.txt
```

Datasets and pretrained weights:

- DATA: <https://drive.google.com/file/d/1LYmLACEJou3TyT2qEEfRE6wPOQgB1-WT/view?usp=share_link>
- DEQ weights: <https://drive.google.com/file/d/1Sgk_J26LMeSwMJc0fHQvv1qGjsWb7hKv/view?usp=share_link>
- Denoiser weights: <https://drive.google.com/file/d/1kXpB_9MiTomWg0kDmTayu5qess75KmTP/view?usp=sharing>

## 2. Dataset Structure

```text
DEQS/
├── DATA/
│   ├── MRI/
│   │   ├── singlecoil_train/
│   │   ├── singlecoil_val/
│   │   └── singlecoil_test/
│   └── BSDS500/
│       ├── train/
│       ├── val/
│       └── test/
├── DEQ_weights/
└── networks/
    ├── GS_DRUNet_SPlus.ckpt
    └── GSDRUNet_grayscale_torch/
```

## 3. Main Components

## Train DEQ models — `run_training.py`

Train or evaluate DEQ models for MRI, inpainting, and Rician denoising.

### Arguments

```text
--problem {mri,inpainting,rician}
--dc {grad,prox}
--data_root DATA
--save_dir PATH
--train BOOL
--lr 1e-5
--max_epochs 500
--max_iter 200
--init_train False
--lambda_dc 0.1
--backtracking True
--lambda_Rtheta 0.83
--learn_lambda_dc True
--learn_lambda_Rtheta True
--accelerated True
--B_restart 100
--theta_interpol 0.2
--learn_theta_interpol True
--andersen_acceleration False
--cycle_andersen False
--m_andersen 5
--noise 1/255
--sigma_denoiser 0.03
--device cuda:0
```

Example:

```bash
python run_training.py --problem mri --dc grad --train False --max_iter 500 --lambda_dc 0.5 --backtracking False --lambda_Rtheta 0.65 --accelerated True --noise 0.004 --sigma_denoiser 0.03 --data_root DATA --save_dir DEQ_weights/MRI/iDEQ_200_plus --device cuda:0
```

## Grid Search — `main_gridsearch.py`

Grid search for DEQ/PnP hyperparameters.

```text
--problem {mri,inpainting,rician}
--dc {grad,prox}
--accelerated True
--PnP False
```

Example:

```bash
python main_gridsearch.py --problem mri --dc grad --accelerated True
```

## Unrolling — `run_unrolling.py`

Supports **VarNet** and **MoDL** for MRI reconstruction.

```text
--mode {VarNet,MoDL}
--train
--test
--max_iter 5
--lambda_dc 0.5
--lr 1e-4
--max_epochs 500
--acceleration 8
--noise 1/255
--data_root DATA
--save_dir PATH
```

Example:

```bash
python run_unrolling.py --mode VarNet --train --test --max_iter 5 --lambda_dc 0.5 --lr 1e-4 --max_epochs 500 --acceleration 8 --noise 0.00390625 --data_root DATA --save_dir runs/unrolling_experiment
```

## MRI Diffusion/E2E Reconstruction — `MRI_DPS_DDRM_RAM.py`

Unified MRI pipeline for **DPS, DDRM, and RAM**.

```text
--algo {DPS,DDRM,RAM}
--acceleration 8
--sigma-level 1
--steps 500
--dps-weight 0.04
--gpu 0
--seed 42
--batch-size 1
--img-size 320 320
--save-dir PATH
--dataset-dir datasets/MRI
--data-root DATA/MRI
--denoiser-ckpt networks/GSDRUNet_grayscale_torch.ckpt
--num-images 5
```

Examples:

```bash
python MRI_DPS_DDRM_RAM.py --algo DPS --acceleration 8 --sigma-level 1 --steps 200 --dps-weight 0.04 --gpu 0 --batch-size 1
python MRI_DPS_DDRM_RAM.py --algo DDRM --acceleration 8 --sigma-level 1 --steps 200 --gpu 0 --batch-size 1
python MRI_DPS_DDRM_RAM.py --algo RAM --acceleration 8 --sigma-level 1 --gpu 0 --batch-size 1
```

Results are automatically organized by algorithm, acceleration, and noise level.

## RGB Diffusion/E2E Reconstruction — `RGB_DPS_DDRM_RAM.py`

Unified RGB pipeline for **DPS, DDRM, and RAM** on inpainting and Rician denoising.

```text
--algo {DPS,DDRM,RAM}
--problem {inpainting,rician}
--sigma-level 1
--steps 500
--dps-weight 4.8
--gpu 0
--seed 42
--batch-size 1
--save-dir PATH
--data-root DATA
--num-images 5
```

Examples:

```bash
python RGB_DPS_DDRM_RAM.py --algo DPS --problem inpainting --sigma-level 1 --steps 200 --dps-weight 4.8 --gpu 0 --batch-size 1
python RGB_DPS_DDRM_RAM.py --algo DDRM --problem inpainting --sigma-level 1 --steps 200 --gpu 0 --batch-size 1
python RGB_DPS_DDRM_RAM.py --algo RAM --problem inpainting --sigma-level 1 --gpu 0 --batch-size 1
```

## DiffPIR — `run_DIFFPIR.py`

DiffPIR supports test and grid-search modes.

```text
--problem {mri,inpainting,rician}
--mode {grid,test}
--device cuda:0
--noise_level 12.75
--lambda_ 10.0
--zeta 0.5
--lambda_min 3.0
--lambda_max 25.0
--n_lambda 10
--zeta_min 0.0
--zeta_max 1.0
--n_zeta 10
```

Example:

```bash
python run_DIFFPIR.py --problem mri --mode test --lambda_ 10.0 --zeta 0.5 --noise_level 12.75 --device cuda:0
```

## References

[1] J. Zbontar et al. *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*. arXiv:1811.08839, 2018.

[2] P. Arbelaez et al. *Contour Detection and Hierarchical Image Segmentation*. IEEE TPAMI, 33(5):898–916, 2011.
