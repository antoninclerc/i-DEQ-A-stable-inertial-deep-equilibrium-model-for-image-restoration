# i-DEQ: A Stable Inertial Deep Equilibrium Model for Image Restoration

This repository contains the official implementation of **i-DEQ: A Stable Inertial Deep Equilibrium Model for Image Restoration**.

Implemented methods include:

- Deep Equilibrium Models (DEQ)
- Unrolled networks (VarNet / MoDL)
- Diffusion-based reconstruction (DiffPIR-style)
- Grid search for hyperparameter selection

The methods are evaluated on:

- MRI reconstruction (FastMRI)
- Image inpainting (BSDS500)
- Rician denoising (BSDS500)

---

# 1. Installation

```bash
pip install -r requirements.txt
```

# 2. Dataset Structure
Expected dataset layout:

DATA/
├── MRI/
│   ├── singlecoil_train/
│   ├── singlecoil_val/
│   └── singlecoil_test/
│
└── BSDS500/
    ├── train/
    ├── val/
    └── test/

# 3. Main components

## Train DEQ models – run_training.py
Allow users to train differents DEQ models.

### CLI Arguments

#### Required Arguments

##### `--problem` (str)
Options: `mri | inpainting | rician`  
Type of inverse problem.

##### `--dc` (str)
Options: `grad | prox`  
Data-consistency operator.

##### `--data_root` (str)
Path to dataset root directory. 
Defaults : 'DATA'

##### `--save_dir` (str)
Directory where checkpoints and logs are saved.

---

#### Training hyperparameters

##### `--lr` (float, default=1e-5)
Learning rate.

##### `--max_epochs` (int, default=500)
Number of training epochs.

##### `--max_iter` (int, default=200)
Maximum number of DEQ iterations.

##### `--init_train` (bool, default=False)
Enables 20 iterations with higher denoising parameter sigma_denoiser = 0.2. Recommanded for inpainting tasks.

---

#### Model parameters

##### `--lambda_dc` (float, default=0.1)
Weight of the data-consistency term.

##### `--backtracking` (bool, default=True)
Enables backtracking line-search.

##### `--lambda_Rtheta` (float, default=0.83)
Weight of the network regularization term.


---

#### Optimization / stability

##### `--learn_lambda_dc` (bool, default=True)
Whether λ_dc is learned during training.

##### `--learn_lambda_Rtheta` (bool, default=True)
Whether λ_Rθ is learned during training.

---

#### DEQ dynamics / restart

##### `--accelerated` (bool, default=True)
Enables accelerated DEQ with restart mechanism.

##### `--B_restart` (int, default=100)
Restart period for DEQ iterations.

##### `--theta_interpol` (float, default=0.2)
Interpolation parameter θ.

##### `--learn_theta_interpol` (bool, default=True)
Whether θ is learned during training.

---

#### Noise model

##### `--noise` (float, default=1/255)
Observation noise level.

##### `--sigma_denoiser` (float, default=0.03)
Noise level used in the denoiser.

---

#### System

##### `--device` (str, default=cuda:0)
Device used for training.

##### `--PnP` (bool, default=False)
Use the PnP (and not RED) approach.
---

## Grid Search – main_gridsearch.py

This project includes an automated grid search procedure to tune the hyperparameters of the PnP-based reconstruction models for MRI, inpainting, and Rician denoising tasks.

---

### Overview

The grid search evaluates multiple combinations of:
- regularization strength (`lambda_Rtheta`)
- denoiser noise level (`sigma_denoiser`)
- data-consistency weight (`lambda_dc`)

Each configuration is trained/evaluated independently, and metrics (PSNR/SSIM) are stored per run.

---
### CLI Arguments

#### Required arguments

##### `--problem` (str)
Options: `mri | inpainting | rician`  
Defines the inverse problem to solve.

- `mri`: accelerated MRI reconstruction
- `inpainting`: image inpainting with missing pixels
- `rician`: denoising with Rician noise model

---

##### `--dc` (str)
Options: `grad | prox`  
Defines the data-consistency operator used in the reconstruction model.

- `grad`: gradient-based data consistency
- `prox`: proximal operator-based data consistency

---

#### Training / execution mode

##### `--accelerated` (bool, default=True)
Enables accelerated variants of the reconstruction algorithm:
- faster convergence
- modified iteration scheme

---

##### `--PnP` (bool, default=False)
Enables Plug-and-Play formulation instead of the default reconstruction framework.

- `True`: Plug-and-Play denoiser-based reconstruction
- `False`: standard learned/unrolled formulation

---

## Unrolling Experiments (VarNet / MoDL) - run_unrolling.py

This section describes classical unrolled reconstruction models for MRI.

---

### Overview

Two architectures are supported:

- **VarNet**
- **MoDL**

Both combine:
- learned denoising network
- physics-based data consistency
- iterative reconstruction

---

### CLI Arguments


#### Mode selection

##### `--mode` (str, required)
Options: `VarNet | MoDL`  
Defines the type of unrolled architecture.

- `VarNet`: gradient-based data consistency (`grad`)
- `MoDL`: proximal data consistency (`prox`)

---

#### Execution flags

##### `--train` (flag)
If set, enables training phase.

- runs optimization over training set
- validates on validation set
- saves best model to disk

---

##### `--test` (flag)
If set, runs evaluation using the best saved checkpoint.

- computes PSNR / SSIM
- saves quantitative results in `results.txt`

---

#### Optimization hyperparameters

##### `--max_iter` (int, default=5)
Number of unrolling iterations inside the network.

---

##### `--lambda_dc` (float, default=0.5)
Weight of the data-consistency term.

Higher values enforce stronger fidelity to measurements.

---

##### `--lr` (float, default=1e-4)
Learning rate used for training.

---

##### `--max_epochs` (int, default=500)
Maximum number of training epochs.

---

#### Physics configuration

##### `--acceleration` (int, default=8)
MRI undersampling factor.

---

##### `--noise` (float, default=1/255)
Noise level added to measurements.

Used both in:
- training simulation
- evaluation consistency

---

#### Dataset configuration

##### `--data_root` (str, default=`DATA`)
Root directory of the dataset.

Expected structure:
- `MRI/singlecoil_train`
- `MRI/singlecoil_val`
- `MRI/singlecoil_test`

---

#### Output configuration

##### `--save_dir` (str, required)
Directory where:

- trained models
- checkpoints
- evaluation results

are stored.

## DIFFPIR experiments - run_DIFFPIR.py

---

### Overview

This framework supports two execution modes for diffusion-based reconstruction:

- **grid search mode**: hyperparameter selection
- **test mode**: final evaluation

Both modes use the same reconstruction pipeline, but differ in how parameters are chosen.

---

### Grid search mode

Grid search is used to select good values for the hyperparameters:
- `lambda` (data-consistency / regularization strength)
- `zeta` (stochasticity level in reverse diffusion)

The method evaluates multiple configurations on the **validation set**.

For each pair \((\lambda, \zeta)\):
- full reconstruction is run
- PSNR and SSIM are computed
- results are stored on disk

The best configuration is selected based on validation PSNR.

This step is necessary because performance is highly sensitive to noise level and degradation type.

---

### Test mode

Test mode evaluates a single fixed configuration:
- one value of `lambda`
- one value of `zeta`

No parameter search is performed.

This mode is used for:
- final reporting
- comparison with other methods
- reproducible evaluation

---

### Arguments

---

#### Global setup

##### `--problem` (str)
Options: `mri | inpainting | rician`  
Defines the inverse problem to solve.

---

#####  `--mode` (str)
Options: `grid | test`  
Execution mode:
- `grid`: hyperparameter search on validation set
- `test`: single-run evaluation

---

#####  `--device` (str, default=`cuda:0`)
Device used for computation.

---

#### Noise model

#####  `--noise_level` (float, default=`12.75`)
Standard deviation of the observation noise (image space scale).

---

##### Test mode parameters (`--mode test`)

Used only when running a single reconstruction.

---

#####  `--lambda_` (float, default=`10.0`)
Regularization / data-consistency weighting parameter.

---

#####  `--zeta` (float, default=`0.5`)
Stochasticity parameter in the reverse diffusion process.

---

##### Grid search parameters (`--mode grid`)

Used only when performing hyperparameter search.

---

#####  `--lambda_min` (float, default=`3.0`)
Minimum value of λ tested.

#####  `--lambda_max` (float, default=`25.0`)
Maximum value of λ tested.

#####  `--n_lambda` (int, default=`10`)
Number of sampled λ values (uniform grid between min and max).

#####  `--zeta_min` (float, default=`0.0`)
Minimum value of ζ tested.

#####  `--zeta_max` (float, default=`1.0`)
Maximum value of ζ tested.

#####  `--n_zeta` (int, default=`10`)
Number of sampled ζ values (uniform grid between min and max).