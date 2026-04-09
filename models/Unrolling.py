from turtle import mode

import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from models.data_consistency import (
    DC_prox_MRI, DC_grad_MRI, Adjoint_MRI, Forward_MRI, 
    DC_prox_inpainting, DC_grad_inpainting, Forward_inpainting, Adjoint_inpainting,
    DC_prox_Rician, DC_grad_Rician, Forward_Rician, Adjoint_Rician)

from utils import (
    image_2ch_to_magnitude,
    ifft2c,
    setup_training,
    validation_and_checkpoint,
    plot_training_state,
    compute_batch_metrics,
    load_pretrained,
    add_zero_channel,
    show_image
)

class Unrolling(nn.Module):
    def __init__(self,
                 # -----------------------------
                 # Network and problem
                 # -----------------------------
                 Network,
                 problem,

                 # -----------------------------
                 # Data consistency
                 # -----------------------------
                 DC_type,

                 # -----------------------------
                 # Regularization parameters
                 # -----------------------------
                 lambda_dc=0.1,
                 learn_lambda_dc=False,


                 # -----------------------------
                 # Iteration / convergence
                 # -----------------------------
                 max_iter=1000,

                 # -----------------------------
                 # Device / path / noise
                 # -----------------------------
                 device='cpu',
                 path_folder=None,
                 sigma_noise=0.0,
                 ):

        super(Unrolling, self).__init__()

        # -----------------------------
        # Device and paths
        # -----------------------------
        self.device = device
        self.path = path_folder if path_folder is not None else "nopath"

        # -----------------------------
        # Regularization / DC
        # -----------------------------
        self.lambda_dc = lambda_dc if not learn_lambda_dc else torch.nn.Parameter(torch.tensor(lambda_dc))
        self.DC_type = DC_type

        # -----------------------------
        # Network and problem
        # -----------------------------
        self.Network = Network
        self.problem = problem

        # -----------------------------
        # Step / iteration
        # -----------------------------
        self.max_iter = max_iter
        self.sigma_noise = sigma_noise

        # -----------------------------
        # Directory for models
        # -----------------------------
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.get_dc()

    def get_dc(self):
        if self.problem == "MRI":
            if self.DC_type == "prox":
                self.DC = DC_prox_MRI
            elif self.DC_type == "grad":
                self.DC = DC_grad_MRI
            self.Adjoint = Adjoint_MRI
            self.forward_op = Forward_MRI
            self.adjoint_op = Adjoint_MRI

        elif self.problem == "Inpainting":
            if self.DC_type == "prox":
                self.DC = DC_prox_inpainting
            elif self.DC_type == "grad":
                self.DC = DC_grad_inpainting
            self.Adjoint = Adjoint_inpainting
            self.forward_op = Forward_inpainting
            self.adjoint_op = Adjoint_inpainting
        else:            
            raise ValueError("Unsupported problem type")
        
    def denoiser(self, x):
        if self.problem == "MRI":
            x_in = x[:, 0, :, :].unsqueeze(1)  # Extract real part
        else:
            x_in = x
       
        x_out = self.Network(x_in, self.sigma_noise)
        
        if self.problem == "MRI":
            x_out = add_zero_channel(x_out)  # Add zero imaginary part
        
        return x_out
    
    def forward_MoDL(self, y, mask):

        if self.problem == "MRI":
            image = self.Adjoint(y, mask) 
        elif self.problem == "Inpainting":
            image = self.Adjoint(y, mask) + 0.5 * (1 - mask) * y  # Start with zero-filled image for inpainting
        
        niter = 0
        intermediate_outputs = []

        while niter < self.max_iter:

            image = self.DC(image, y, mask, self.lambda_dc)    
                
            # Denoising step
            image = self.denoiser(image)

            niter += 1

            intermediate_outputs.append(image.detach())
        
        return image, intermediate_outputs
    
    def forward_Varnet(self, y, mask):

        if self.problem == "MRI":
            image = self.Adjoint(y, mask) 
        elif self.problem == "Inpainting":
            image = self.Adjoint(y, mask) + 0.5 * (1 - mask) * y  # Start with zero-filled image for inpainting
        
        niter = 0
        intermediate_outputs = []

        while niter < self.max_iter:

            image_grad = self.DC(image, y, mask, self.lambda_dc)

            image_denoised = self.denoiser(image)

            image = image_grad - image_denoised

            niter += 1
            intermediate_outputs.append(image.detach())
        
        return image, intermediate_outputs

    def train_model(
        self,
        train_loader,
        val_loader,
        mode="MoDL",
        lr=1e-3,
        eta_k=None,
        eta_TV=None,
        eta_l1=None,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
        max_patience=10,
        max_epochs=100,
        plot_interval=10,
        pretrained_path=None):
        
        criterion, optimizer, scheduler = setup_training(
            model=self,
            device=self.device,
            network=self.Network,
            lr=lr,
            eta_k=eta_k,
            eta_TV=eta_TV,
            eta_l1=eta_l1,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        if self.DC_type != "grad" and mode == "Varnet":
            print("Warning: Varnet mode is designed for gradient-based DC. Switching to MoDL mode.")
            mode = "MoDL"

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        # -------------------------------------------------
        # Optional pretrained loading
        # -------------------------------------------------
        if pretrained_path is not None:
            load_pretrained(
                model=self,
                checkpoint_path=pretrained_path,
                device=self.device,
                optimizer=self.optimizer,  # facultatif
                load_optimizer=True
            )

        best_val_loss = float("inf")
        patience = 0

        train_losses, val_losses = [], []
        train_PSNRs, val_PSNRs = [], []
        train_ssim, val_ssim = [], []

        iter_nums = []
        iter_nums_val = []

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        # =================================================
        # Training loop
        # =================================================
        for epoch in range(1, max_epochs+1):

            if patience >= max_patience:
                break

            # ---------------------
            # Train phase
            # ---------------------
            self.train()

            epoch_train_mse = []
            epoch_train_psnr = []
            epoch_train_ssim = []

            for batch_nr, (batch_target, batch_input, batch_mask) in enumerate(train_loader):

                print(f"Epoch {epoch}, Batch {batch_nr + 1}/{len(train_loader)}", end='\r')
                
                batch_input = batch_input.to(self.device).float()

                B = batch_input.shape[0]

                if self.sigma_noise == "random":
                    noise = torch.rand(B, 1, 1, 1, device=self.device) * 0.1
                else:
                    noise = torch.full((B, 1, 1, 1), self.sigma_noise, device=self.device)

                batch_input = batch_input + noise * torch.randn_like(batch_input)
                batch_input = torch.clamp(batch_input, 0, 1.0)

                batch_mask = batch_mask["mask"].to(self.device).float()
                if self.problem == "MRI":
                    batch_input = batch_input * batch_mask  # Ensure masked input is consistent in k-space for MRI

                batch_target = batch_target.to(self.device).float()

                epoch_iter_nums = []
                epoch_iter_nums_val = []
                if mode == "MoDL":
                    outputs, _ = self.forward_MoDL(batch_input, batch_mask)
                else:
                    outputs, _ = self.forward_Varnet(batch_input, batch_mask)

                # 2) Compute loss
                self.optimizer.zero_grad()

                # Standard backward on loss
                loss = self.criterion(outputs, batch_target)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                mse_list, psnr_list, ssim_list, _ = compute_batch_metrics(
                    outputs, batch_target
                )

                epoch_train_mse.extend(mse_list)
                epoch_train_psnr.extend(psnr_list)
                epoch_train_ssim.extend(ssim_list)

            epoch_train_mse = np.mean(epoch_train_mse)
            epoch_train_psnr = np.mean(epoch_train_psnr)
            epoch_train_ssim = np.mean(epoch_train_ssim)

            train_losses.append(epoch_train_mse)
            train_PSNRs.append(epoch_train_psnr)
            train_ssim.append(epoch_train_ssim)

            # ---------------------
            # Validation phase
            # ---------------------
            self.eval()

            epoch_val_mse = []
            epoch_val_psnr = []
            epoch_val_ssim = []
            with torch.no_grad():
                for batch_target, batch_input, batch_mask in val_loader:

                    batch_input = batch_input.to(self.device).float()
                    
                    B = batch_input.shape[0]

                    if self.sigma_noise == "random":
                        noise = torch.linspace(0, 0.1, B, device=self.device).view(B,1,1,1)
                    else:
                        noise = torch.full((B,1,1,1), self.sigma_noise, device=self.device)

                    batch_input = batch_input + noise * torch.randn_like(batch_input)

                    batch_input = torch.clamp(batch_input, 0, 1.0)  # Ensure input is in valid range

                    batch_mask = batch_mask["mask"].to(self.device).float()
                    if self.problem == "MRI":
                        batch_input = batch_input * batch_mask  # Ensure masked input is consistent in k-space

                    batch_target = batch_target.to(self.device).float()

                    if mode == "MoDL":
                        outputs, _ = self.forward_MoDL(batch_input, batch_mask)
                    else:
                        outputs, _ = self.forward_Varnet(batch_input, batch_mask)

                    mse_list, psnr_list, ssim_list, _ = compute_batch_metrics(
                        outputs, batch_target
                    )

                    epoch_val_mse.extend(mse_list)
                    epoch_val_psnr.extend(psnr_list)
                    epoch_val_ssim.extend(ssim_list)

                epoch_val_mse = np.mean(epoch_val_mse)
                epoch_val_psnr = np.mean(epoch_val_psnr)
                epoch_val_ssim = np.mean(epoch_val_ssim)

                val_losses.append(epoch_val_mse)
                val_PSNRs.append(epoch_val_psnr)
                val_ssim.append(epoch_val_ssim)

                print(
                    f"Epoch {epoch}/{max_epochs} | "
                    f"Train MSE: {epoch_train_mse:.6f} | "
                    f"Train PSNR: {epoch_train_psnr:.2f} dB | "
                    f"Val MSE: {epoch_val_mse:.6f} | "
                    f"Val PSNR: {epoch_val_psnr:.2f} dB | "
                    f"Val SSIM: {epoch_val_ssim:.4f} | "
                    f"Patience: {patience}"
                )

                # ---------------------
                # Plot
                # ---------------------
                if plot_interval is not None and epoch % plot_interval == 0:
                    plot_training_state(
                        epoch,
                        train_losses,
                        val_losses,
                        train_PSNRs,
                        val_PSNRs,
                        outputs,
                        batch_target,
                        self.path,
                    )

                # ---------------------
                # Early stopping + scheduler + checkpoint
                # ---------------------
                best_val_loss, patience = validation_and_checkpoint(
                    model=self,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    epoch_val_mse=epoch_val_mse,
                    best_val_loss=best_val_loss,
                    patience=patience,
                    max_epochs=max_epochs,
                    save_path=self.path,
                    min_iter=0,
                )

            epoch += 1

        print("Training complete.")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_PSNRs": train_PSNRs,
            "val_PSNRs": val_PSNRs,
            "train_ssim": train_ssim,
            "val_ssim": val_ssim,
            "iter_nums": iter_nums,
            "iter_nums_val": iter_nums_val,
        }
    
    def evaluate(self, test_loader, mode="MoDL", n_display=1, pretrained_path=None):

        self.eval()
        self.to(self.device)

        if pretrained_path is not None:
            load_pretrained(model=self, checkpoint_path=pretrained_path, device=self.device)

        test_mse, test_PSNR = [], []
        input_mse, input_PSNR = [], []
        test_SSIM, input_SSIM = [], []

        # buffers
        display_outputs, display_targets, display_inputs = [], [], []
        display_psnr, display_input_psnr = [], []
        display_ssim, display_input_ssim = [], []
        n_collected = 0

        # convergence plots
        psnr_per_iter_accumulator = None
        eps_per_iter_accumulator = None
        n_batches = 0

        with torch.no_grad():
            for batch_target, batch_input, batch_mask in test_loader:

                batch_input = batch_input.to(self.device).float()
                    
                B = batch_input.shape[0]

                if self.sigma_noise == "random":
                    noise = torch.linspace(0, 0.1, B, device=self.device).view(B,1,1,1)
                else:
                    noise = torch.full((B,1,1,1), self.sigma_noise, device=self.device)

                batch_input = batch_input + noise * torch.randn_like(batch_input)

                batch_input = torch.clamp(batch_input, 0, 1.0)  # Ensure input is in valid range
                    
                batch_mask = batch_mask["mask"].to(self.device).float()
                if self.problem == "MRI":
                    batch_input = batch_input * batch_mask  # Ensure masked input is consistent in k-space

                batch_target = batch_target.to(self.device).float()              

                if mode == "MoDL":
                    outputs, intermediates = self.forward_MoDL(batch_input, batch_mask)
                else:
                    outputs, intermediates = self.forward_Varnet(batch_input, batch_mask)

                # -----------------------------
                # Metrics
                # -----------------------------
                if self.problem == "MRI":
                    batch_input_in = ifft2c(batch_input)
                else:
                    batch_input_in = batch_input
                mse_list, psnr_list, ssim_list, _ = compute_batch_metrics(outputs, batch_target)
                mse_input, psnr_input, ssim_input, _ = compute_batch_metrics(batch_input_in, batch_target)

                test_mse.extend(mse_list)
                test_PSNR.extend(psnr_list)
                test_SSIM.extend(ssim_list)

                input_mse.extend(mse_input)
                input_PSNR.extend(psnr_input)
                input_SSIM.extend(ssim_input)

                # -----------------------------
                # PSNR evolution
                # -----------------------------
                n_iter = len(intermediates)

                if psnr_per_iter_accumulator is None:
                    psnr_per_iter_accumulator = np.zeros(n_iter)

                for k in range(n_iter):
                    interm_k = intermediates[k]
                    _, psnr_k, _, _ = compute_batch_metrics(interm_k, batch_target)
                    psnr_per_iter_accumulator[k] += float(np.mean(psnr_k))

                # -----------------------------
                # epsilon evolution
                # -----------------------------

                n_batches += 1

                # -----------------------------
                # Collect images
                # -----------------------------
                if n_display > 0 and n_collected < n_display:

                    batch_size = outputs.shape[0]
                    n_to_take = min(batch_size, n_display - n_collected)

                    display_outputs.append(outputs[:n_to_take].detach().cpu())
                    display_targets.append(batch_target[:n_to_take].detach().cpu())
                    display_inputs.append(batch_input[:n_to_take].detach().cpu())

                    display_psnr.extend(psnr_list[:n_to_take])
                    display_input_psnr.extend(psnr_input[:n_to_take])
                    display_ssim.extend(ssim_list[:n_to_take])
                    display_input_ssim.extend(ssim_input[:n_to_take])

                    n_collected += n_to_take

            # ---------------------------------------
            # Global metrics
            # ---------------------------------------
            test_mse = float(np.mean(test_mse))
            test_PSNR = float(np.mean(test_PSNR))
            test_SSIM = float(np.mean(test_SSIM))

            input_mse = float(np.mean(input_mse))
            input_PSNR = float(np.mean(input_PSNR))
            input_SSIM = float(np.mean(input_SSIM))

            print(f"Test MSE: {test_mse:.6f}, Test PSNR: {test_PSNR:.2f} dB")
            print(f"Input MSE: {input_mse:.6f}, Input PSNR: {input_PSNR:.2f} dB")
            print(f"Test SSIM: {test_SSIM:.4f}, Input SSIM: {input_SSIM:.4f}")

            # ---------------------------------------
            # Reconstructions
            # ---------------------------------------
            if n_collected > 0:

                display_outputs = torch.cat(display_outputs, dim=0)
                display_targets = torch.cat(display_targets, dim=0)
                display_inputs = torch.cat(display_inputs, dim=0)

                if self.problem == "MRI":
                    display_outputs = image_2ch_to_magnitude(display_outputs)
                    display_targets = image_2ch_to_magnitude(display_targets)
                    display_inputs = image_2ch_to_magnitude(ifft2c(display_inputs))

                display_inputs = torch.clamp(display_inputs, 0, 1.0)
                display_targets = torch.clamp(display_targets, 0, 1.0)
                display_outputs = torch.clamp(display_outputs, 0, 1.0)

                for i in range(n_collected):

                    image = display_outputs[i]
                    target = display_targets[i]
                    input_image = display_inputs[i]

                    error_map = np.abs(image - target)

                    psnr_recon = display_psnr[i]
                    psnr_input_img = display_input_psnr[i]
                    ssim_recon = display_ssim[i]
                    ssim_input_img = display_input_ssim[i]

                    plt.figure(figsize=(10,4))

                    ax = plt.subplot(1,4,1)
                    show_image(
                        ax,
                        input_image,
                        f"Zero-Filled\nPSNR {psnr_input_img:.2f}, SSIM {ssim_input_img:.4f}"
                    )

                    ax = plt.subplot(1,4,2)
                    show_image(
                        ax,
                        image,
                        f"Reconstruction\nPSNR {psnr_recon:.2f}, SSIM {ssim_recon:.4f}"
                    )

                    ax = plt.subplot(1,4,3)
                    show_image(
                        ax,
                        target,
                        "Ground Truth"
                    )

                    ax = plt.subplot(1,4,4)
                    show_image(
                        ax,
                        error_map,
                        "Error map",
                        is_error=True
                    )

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.path, f"test_reconstruction_{i}.png"))
                    plt.close()

        return {
            "test_mse": test_mse,
            "test_PSNR": test_PSNR,
            "input_mse": input_mse,
            "input_PSNR": input_PSNR,
        }
