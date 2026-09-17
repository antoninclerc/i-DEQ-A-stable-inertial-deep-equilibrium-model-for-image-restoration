import argparse
import os
import torch

from grid_search import build_config
from gen_data import get_dataloaders
from models.deep_equilibrium import DeepEquilibrium
from networks.DRUnet import GSDRUNet
from utils import str2bool, normalize_problem


# -------------------------
# main
# -------------------------
def main():

    parser = argparse.ArgumentParser()

    # problem setup
    parser.add_argument("--problem", required=True, choices=["mri", "inpainting", "deblurring", "rician"])
    parser.add_argument("--dc", required=True, choices=["grad", "prox"])
    parser.add_argument("--train", type=str2bool, default=True)

    # training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--init_train", type=str2bool, default=False)

    # model params
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--backtracking", type=str2bool, default=True)
    parser.add_argument("--lambda_Rtheta", type=float, default=0.83)

    # stability / training behaviour
    parser.add_argument("--learn_lambda_dc", type=str2bool, default=True)
    parser.add_argument("--learn_lambda_Rtheta", type=str2bool, default=True)

    # restart
    parser.add_argument("--accelerated", type=str2bool, default=True)
    parser.add_argument("--B_restart", type=int, default=100)
    parser.add_argument("--theta_interpol", type=float, default=0.2)
    parser.add_argument("--learn_theta_interpol", type=str2bool, default=True)

    parser.add_argument("--andersen_acceleration", type=str2bool, default=False)
    parser.add_argument("--cycle_andersen", type=str2bool, default=False)
    parser.add_argument("--m_andersen", type=int, default=5)

    # noise
    parser.add_argument("--noise", type=float, default=1/255)
    parser.add_argument("--sigma_denoiser", type=float, default=0.03)

    # paths
    parser.add_argument("--data_root", type=str, required=True, default="DATA")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    problem = normalize_problem(args.problem)

    if args.accelerated and args.backtracking:
        print("Running in accelerated setting (with restart).")
        print("Backtracking will be disabled.")
        args.backtracking = False
    
    if args.backtracking and args.learn_lambda_dc:
        print("Backtracking is enabled, but lambda_dc will not be learned.")
        args.learn_lambda_dc = False

    if args.backtracking and args.learn_theta_interpol:
        print("Backtracking is enabled, but theta_interpol will not be learned.")
        args.learn_theta_interpol = False

    # -------------------------
    # dataset config
    # -------------------------
    config = {
        "train_path": os.path.join(args.data_root, "MRI/singlecoil_train" if problem == "MRI" else "BSDS500/train"),
        "val_path": os.path.join(args.data_root, "MRI/singlecoil_val" if problem == "MRI" else "BSDS500/val"),
        "test_path": os.path.join(args.data_root, "MRI/singlecoil_test" if problem == "MRI" else "BSDS500/test"),
        "sigma": args.noise,
        "split_ratio": 0.5,
        "img_size": (3, 320, 320) if problem != "MRI" else (320, 320),
        "acceleration": 8,
        "device": "cuda:0",
        "seed": 42,
    }

    train_loader, val_loader, test_loader, physics = get_dataloaders(problem.lower(), config)

    # -------------------------
    # model
    # -------------------------
    if problem == "MRI":
        cnn = GSDRUNet(
            in_channels=1,
            out_channels=1,
            pretrained="networks/GSDRUNet_grayscale_torch.ckpt"
        )
    else:
        cnn = GSDRUNet(
            in_channels=3,
            out_channels=3,
            pretrained="networks/GS_DRUNet_SPlus.ckpt",
            act_mode="s"
        )

    model = DeepEquilibrium(
        Network=cnn,
        problem=problem,
        DC_type=args.dc,
        backtracking=args.backtracking,
        lambda_dc=args.lambda_dc,
        lambda_Rtheta=args.lambda_Rtheta,
        learn_lambda_dc=args.learn_lambda_dc,
        learn_lambda_Rtheta=args.learn_lambda_Rtheta,
        gamma=0.01,
        eta=0.5,
        thresh=1e-4,
        max_iter=args.max_iter,
        device=args.device,
        path_folder=args.save_dir,
        sigma_noise=args.noise,
        sigma_denoiser=args.sigma_denoiser,
        restart=True,
        B_restart=args.B_restart,
        theta_interpol=args.theta_interpol,
        learn_theta_interpol=args.learn_theta_interpol,
        cycle_andersen=args.cycle_andersen,
        m_andersen=args.m_andersen,
    )

    if args.init_train:
        init_train_params = {"epoch_pretraining" : 20, "sigma_pretraining" : 0.2, "tau0_pretraining" : 0.1}
    else:
        init_train_params = None

    # -------------------------
    # training
    # -------------------------
    if args.train:
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            accelerated=args.accelerated,
            andersen_acceleration=args.andersen_acceleration,
            init_train=init_train_params,
            JFB=True,
            K_JFB=0.,
            lr=args.lr,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={"betas": (0.9, 0.999)},
            scheduler=None,
            scheduler_kwargs=None,
            max_patience=25,
            max_epochs=args.max_epochs,
            plot_interval=1,
            pretrained_path=args.pretrained,
        )

    # -------------------------
    # evaluation
    # -------------------------
    results = model.evaluate(
        test_loader=test_loader,
        accelerated=args.accelerated,
        andersen_acceleration=args.andersen_acceleration,
        init_train=init_train_params,
        n_display=5,
        pretrained_path=os.path.join(args.save_dir, "best_model.pth")
    )

    print("TEST PSNR:", results["test_PSNR"])
    print("TEST SSIM:", results["test_SSIM"])


if __name__ == "__main__":
    main()