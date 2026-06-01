import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from models.data_consistency import (
    DC_prox_MRI, DC_grad_MRI, Adjoint_MRI, Forward_MRI, 
    DC_prox_inpainting, DC_grad_inpainting, Forward_inpainting, Adjoint_inpainting,
    DC_prox_Rician, DC_grad_Rician, Forward_Rician, Adjoint_Rician)

from utils import (
    image_2ch_to_magnitude,
    ifft2c,
    one_step,
    one_step_GD_back,
    setup_training,
    validation_and_checkpoint,
    plot_training_state,
    compute_batch_metrics,
    load_pretrained,
    one_step_PGD_back,
    one_step_GD_back,
    one_step,
    restart_condition,
    add_zero_channel,
    jacobian_free_backpropagation,
    show_image
)

class DeepEquilibrium(nn.Module):
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
                 backtracking=True,

                 # -----------------------------
                 # Regularization parameters
                 # -----------------------------
                 lambda_dc=0.1,
                 lambda_Rtheta=1.0,
                 learn_lambda_dc=False,
                 learn_lambda_Rtheta=False,

                 # -----------------------------
                 # Step parameters
                 # -----------------------------
                 gamma=0.5,
                 eta=0.9,

                 # -----------------------------
                 # Iteration / convergence
                 # -----------------------------
                 thresh=1e-5,
                 max_iter=1000,

                 # -----------------------------
                 # Device / path / noise
                 # -----------------------------
                 device='cpu',
                 path_folder=None,
                 sigma_noise=0.0,
                 sigma_denoiser=0.05,

                 # -----------------------------
                 # Acceleration / restart
                 # -----------------------------
                 theta_interpol=0.1,
                 restart=True,
                 B_restart=100,
                 learn_theta_interpol=False,
                 learn_B_restart=False,
                 ):

        super(DeepEquilibrium, self).__init__()

        # -----------------------------
        # Device and paths
        # -----------------------------
        self.device = device
        self.path = path_folder if path_folder is not None else "nopath"

        # -----------------------------
        # Regularization / DC
        # -----------------------------
        if backtracking and learn_lambda_dc:
            print("Warning: lambda_dc is not learnable when backtracking is enabled. Setting lambda_dc to fixed value.")
            self.lambda_dc = lambda_dc
        else:
            self.lambda_dc = lambda_dc if not learn_lambda_dc else torch.nn.Parameter(torch.tensor(lambda_dc))
            self.tau0 = self.lambda_dc  # Initialize tau0 to lambda_dc for backtracking

        self.lambda_Rtheta = lambda_Rtheta if not learn_lambda_Rtheta else torch.nn.Parameter(torch.tensor(lambda_Rtheta))
        self.DC_type = DC_type
        self.backtracking = backtracking

        # -----------------------------
        # Network and problem
        # -----------------------------
        self.Network = Network
        self.problem = problem

        # -----------------------------
        # Step / iteration
        # -----------------------------
        self.gamma = gamma
        self.eta = eta
        self.thresh = thresh
        self.max_iter = max_iter
        self.sigma_noise = sigma_noise
        self.sigma_denoiser = sigma_denoiser

        self.learn_theta_interpol = learn_theta_interpol

        # -----------------------------
        # Acceleration / restart
        # -----------------------------
        self.theta = theta_interpol if not learn_theta_interpol else nn.Parameter(torch.tensor(theta_interpol, dtype=torch.float32))
        self.restart = restart
        self.B_restart = B_restart if not learn_B_restart else nn.Parameter(torch.tensor(B_restart, dtype=torch.float32))

        # -----------------------------
        # Directory for models
        # -----------------------------
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.get_dc()

        self.to(self.device)

    def get_dc(self):
        if self.problem == "MRI":
            if self.DC_type == "prox":
                self.DC = DC_prox_MRI
            elif self.DC_type == "grad":
                self.DC = DC_grad_MRI
            self.Adjoint = Adjoint_MRI
            self.forward_op = Forward_MRI
            self.adjoint_op = Adjoint_MRI
            self.noise_type = 'gaussian'
        elif self.problem == "inpainting":
            if self.DC_type == "prox":
                self.DC = DC_prox_inpainting
            elif self.DC_type == "grad":
                self.DC = DC_grad_inpainting
            self.Adjoint = Adjoint_inpainting
            self.forward_op = Forward_inpainting
            self.adjoint_op = Adjoint_inpainting
            self.noise_type = 'gaussian'
        elif self.problem == "rician":
            if self.DC_type == "prox":
                self.DC = DC_prox_Rician
            elif self.DC_type == "grad":
                self.DC = DC_grad_Rician
            self.Adjoint = Adjoint_Rician
            self.forward_op = Forward_Rician
            self.adjoint_op = Adjoint_Rician
            self.noise_type = 'rician'
        else:
            raise ValueError("Unsupported problem type")


    def Rtheta(self, x):
        if self.problem == "MRI":
            x_in = x[:, 0, :, :].unsqueeze(1)  # Extract real part
        else:
            x_in = x
            
        return self.Network.potential(x_in, self.sigma_denoiser)

    def nabla_x_network(self, x):
        grad = x - self.denoiser(x)
        return grad
        
    def denoiser(self, x):
        if self.problem == "MRI":
            x_in = x[:, 0, :, :].unsqueeze(1)  # Extract real part
        else:
            x_in = x
       
        x_out = self.Network(x_in, self.sigma_denoiser)
        
        if self.problem == "MRI":
            x_out = add_zero_channel(x_out)  # Add zero imaginary part
        
        return x_out
    
    def forward(self, y, mask):

        y = y.clone()

        # Initial reconstruction
        if self.problem == "MRI" or self.problem == "rician":
            x_curr = self.Adjoint(y, mask) # Start from zero-filled reconstruction for MRI and Rician
        elif self.problem == "inpainting":
            x_curr = self.Adjoint(y, mask)  + (1 - mask) * 0.5  # Start from a constant image (could also start from zero-filled)

        x_curr = torch.clamp(x_curr, 0, 1.0)  # Ensure initial image is in valid range
        x_prev = x_curr.detach()
        image0 = x_curr.clone().detach()

        taux_real_iter = []
        epsilons = []

        iter_num = 0
        total_iter = 0
        eps = torch.tensor(float('inf'), device=self.device)

        inter_restart = [x_curr.detach()]

        intermediate_outputs = []
        
        if self.init_train is not None:
            iter_pretraining = self.init_train.get("epoch_pretraining", 10)
            sigma_pretraining = self.init_train.get("sigma_pretraining", 0.2)
            tau0 = self.init_train.get("tau0_pretraining", self.lambda_dc)
            
            train_sigma = self.sigma_denoiser
            train_backtracking = self.backtracking
            
            self.backtracking = False  # Disable backtracking during pretraining for stability
            self.sigma_denoiser = sigma_pretraining  # Use higher noise during pretraining for better convergence
            print(f"Starting pretraining for {iter_pretraining} iterations with sigma={sigma_pretraining}...")
        else:
            iter_pretraining = 0
            tau0 = torch.tensor(self.lambda_dc, device=self.device) if self.backtracking else self.lambda_dc
        if self.andersen_acceleration:
            list_x = []
            list_r = []

        with torch.no_grad():

            if self.accelerated:
                condition = lambda: (iter_num < self.max_iter and eps > self.thresh)
            else:
                condition = lambda: (iter_num < self.max_iter and eps > self.thresh)
            times = [] 
            while condition():
                time_start_iter = time.time()
                if self.accelerated:
                    # inertial extrapolation
                    z = x_curr + (1 - self.theta) * (x_curr - x_prev)
                else:
                    z = x_curr

                # --- one-step update ---
                if self.backtracking:
                    if self.DC_type == 'prox':
                        x_next, tau = one_step_PGD_back(
                            image=z, obs=y, mask=mask, DC=self.DC, R=self.Rtheta,
                            nabla_R=self.nabla_x_network,
                            lambda_dc=tau0, lambda_Rtheta=self.lambda_Rtheta,
                            gamma=self.gamma, eta=self.eta,
                            forward_op=self.forward_op, noise_type=self.noise_type, sigma=self.sigma_noise
                        )
                        
                    elif self.DC_type == 'grad':
                        x_next, tau = one_step_GD_back(
                            image=z, obs=y, mask=mask, R=self.Rtheta,
                            nabla_R=self.nabla_x_network,
                            lambda_dc=tau0, lambda_Rtheta=self.lambda_Rtheta,
                            gamma=self.gamma, eta=self.eta,
                            forward_op=self.forward_op, adjoint_op=self.adjoint_op,
                            noise_type=self.noise_type, sigma=self.sigma_noise
                        )
                    else:
                        raise ValueError("Unsupported DC for backtracking")

                    taux_real_iter.append(tau)

                else:
                    x_next, _ = one_step(
                        image=z, obs=y, mask=mask, DC=self.DC, nabla_R=self.nabla_x_network, 
                        lambda_dc=tau0, lambda_Rtheta=self.lambda_Rtheta, 
                        DC_type=self.DC_type, noise_type=self.noise_type, sigma=self.sigma_noise,
                        forward_op=self.forward_op, adjoint_op=self.adjoint_op
                    )
                
                if self.andersen_acceleration:
                    r_k = (x_next - x_curr).detach()
                    r_k_flat = r_k.view(r_k.size(0), -1).detach()
                    x_next_flat = x_next.view(x_next.size(0), -1)
                    list_x.append(x_next_flat.detach())
                    list_r.append(r_k_flat.detach())
                    
                    if len(list_x) > self.m_andersen:
                        if self.cycle_andersen:
                            list_x = [x_next_flat.detach()]
                            list_r = [r_k_flat.detach()]
                        else:
                            list_x.pop(0)
                            list_r.pop(0)

                    if len(list_r) > 1:
                        R = torch.stack(list_r, dim=1)
                        b, m, d = R.shape
                        beta = []
                        for i in range(b):
                            Ri = R[i].T
                            G = Ri.T @ Ri
                            ones = torch.ones((m, 1), device=R.device)

                            KKT = torch.cat([
                                torch.cat([G, ones], dim=1),
                                torch.cat([ones.T, torch.zeros((1, 1), device=R.device)], dim=1)
                            ], dim=0)

                            rhs = torch.zeros(m + 1, device=R.device)
                            rhs[-1] = 1.0
                            try:
                                sol = torch.linalg.solve(KKT, rhs)
                                beta.append(sol[:-1])
                            except RuntimeError:
                                print("Linear system solve failed in Anderson acceleration, using fallback.")
                                beta.append(torch.zeros(m, device=R.device))
                            
                        beta = torch.stack(beta, dim=0)  # Shape: (B, m_andersen)
                        x_next = torch.sum(beta.view(b, m, 1) * torch.stack(list_x, dim=0).permute(1, 0, 2), dim=1).view_as(x_next)

                tau0 = tau if self.backtracking else tau0

                # --- update variables ---
                x_prev = x_curr.detach()
                x_curr = x_next.detach()

                # --- convergence ---
                eps = torch.norm(x_curr - x_prev) / (torch.norm(x_curr))
                epsilons.append(eps.item())

                # --- restart (only accelerated) ---
                if self.accelerated and self.B_restart > 0:
                    inter_restart.append(x_curr.detach())
                    if restart_condition(inter_restart, self.B_restart):
                        inter_restart = [x_curr.clone().detach()]
                        x_prev = x_curr.clone().detach()
                        iter_num = 0

                iter_num += 1
                total_iter += 1

                if total_iter > iter_pretraining and self.init_train is not None:
                    self.backtracking = train_backtracking  # Restore backtracking setting after pretraining
                    self.sigma_denoiser = train_sigma  # Restore denoiser noise level after pretraining

                    if total_iter == iter_pretraining + 1:
                        print(f"Pretraining complete. Resuming with backtracking={self.backtracking} and sigma_denoiser={self.sigma_denoiser}.")
                        tau0 = torch.tensor(self.lambda_dc, device=self.device) if self.backtracking else self.lambda_dc
                        print(f"Reset tau0 after pretraining.")

                if total_iter > 2000:
                    raise ValueError("Exceeded maximum allowed iterations, possible divergence.")
                
                intermediate_outputs.append(x_curr.detach())

                print(
                    f"Iteration {iter_num} - {total_iter}, "
                    f"norm diff: {torch.norm(x_curr - x_prev).item():.3f}, "
                    f"norm input image: {torch.norm(image0).item():.3f}, "
                    f"norm old image: {torch.norm(x_prev).item():.3f}, "
                    f"norm new image: {torch.norm(x_curr).item():.3f}, "
                    f"tau: {taux_real_iter[-1] if taux_real_iter else 'N/A'}, "
                    f"eps: {eps.item() if eps is not None else 'N/A'}"
                )

                if eps.item() > 5:
                    raise ValueError(
                        f"Divergence detected at iteration {iter_num} with eps={eps.item()}"
                    )
                time_end_iter = time.time()
                times.append(time_end_iter - time_start_iter)
        stats = {
            "iter_num": total_iter,
            "epsilons": epsilons,
            "times": times
        }
        self.tau0 = tau0 if self.backtracking else self.lambda_dc

        outputs = x_curr
        return outputs, intermediate_outputs, stats
    
    def forward_PnP(self, y, mask):

        if self.problem == "MRI" or self.problem == "rician":
            image = self.Adjoint(y, mask) 
        elif self.problem == "inpainting":
            image = self.Adjoint(y, mask)  + (1 - mask) * 0.5  # Start from a constant image (could also start from zero-filled)

        image = torch.clamp(image, 0, 1.0)  # Ensure initial image is in valid range
        niter = 0
        eps = float('inf')
        intermediate_outputs = []
        epsilons = []

        tau0 = torch.tensor(self.lambda_dc, device=self.device)

        if self.init_train is not None:
            iter_pretraining = self.init_train.get("epoch_pretraining", 10)
            sigma_pretraining = self.init_train.get("sigma_pretraining", 0.2)
            
            train_sigma = self.sigma_denoiser
            train_backtracking = self.backtracking
            
            self.backtracking = False  # Disable backtracking during pretraining for stability
            self.sigma_denoiser = sigma_pretraining  # Use higher noise during pretraining for better convergence
            print(f"Starting pretraining for {iter_pretraining} iterations with sigma={sigma_pretraining}...")
        else:
            iter_pretraining = 0
            tau0 = torch.tensor(self.lambda_dc, device=self.device)

        with torch.no_grad():
            while niter < self.max_iter and eps > self.thresh:

                image_prev = image.clone()

                if self.problem != "rician":
                    image = self.DC(image, y, mask, tau0)    
                else:
                    image = self.DC(image, y, self.sigma_noise, tau0)
                
                # Denoising step
                image = self.denoiser(image)

                image = torch.clamp(image, 0, 1.0)

                # Convergence check
                eps = torch.norm(image - image_prev) / torch.norm(image_prev)
                niter += 1

                if niter > iter_pretraining and self.init_train is not None:
                    self.backtracking = train_backtracking  # Restore backtracking setting after pretraining
                    self.sigma_denoiser = train_sigma  # Restore denoiser noise level after pretraining

                    if niter == iter_pretraining + 1:
                        print(f"Pretraining complete. Resuming with backtracking={self.backtracking} and sigma_denoiser={self.sigma_denoiser}.")
                print(f"Iteration {niter}: eps = {eps:.6f}")
                intermediate_outputs.append(image.detach())
                epsilons.append(eps.item())

        stats = {
            "iter_num": niter,
            "epsilons": epsilons
        }
        return image, intermediate_outputs, stats

    def train_model(
        self,
        train_loader,
        val_loader,
        accelerated=False,
        andersen_acceleration=False,
        m_andersen=5,
        cycle_andersen=False,
        init_train=None,
        JFB=True,
        K_JFB=3,
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

        self.K_JFB = K_JFB
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

        self.accelerated = accelerated
        self.andersen_acceleration = andersen_acceleration
        self.m_andersen = m_andersen
        self.cycle_andersen = cycle_andersen
        self.init_train = init_train

        if self.accelerated and self.andersen_acceleration:
            print("Use of Andersen and Inertia. Inertia will be desabled")
            self.accelerated = False  # Disable inertia if Anderson acceleration is enabled, as they can interfere with each other

        train_losses, val_losses = [], []
        train_PSNRs, val_PSNRs = [], []
        train_ssim, val_ssim = [], []

        iter_nums = []
        iter_nums_val = []

        times_epoch = []

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        timer_start_training = time.time()

        # =================================================
        # Training loop
        # =================================================
        for epoch in range(1, max_epochs+1):
            timer_start_epoch = time.time()
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
                
                if self.noise_type == 'gaussian':
                    batch_input = batch_input + self.sigma_noise * torch.randn_like(batch_input)
                    
                batch_mask = batch_mask["mask"].to(self.device).float()
                if self.problem == "MRI":
                    batch_input = batch_input * batch_mask  # For MRI, ensure input is consistent with mask
                else:
                    batch_input = torch.clamp(batch_input, 0, 1.0)  # Ensure input is in valid range

                batch_target = batch_target.to(self.device).float()    

                epoch_iter_nums = []
                epoch_iter_nums_val = []            

                outputs, intermediates, stats = self.forward(batch_input, batch_mask)
                
                epoch_iter_nums.append(stats["iter_num"])

                # 2) Compute loss
                self.optimizer.zero_grad()

                if JFB:
                    # JFB uses fixed-point z_fixed and single-step one_step internally
                    jacobian_free_backpropagation(
                        z_fixed=outputs,
                        z_fixed_before=intermediates[-2] if len(intermediates) > 1 else outputs,  # Use previous intermediate as z_fixed_before for acceleration
                        mask=batch_mask,
                        loss_fn=self.criterion,
                        target=batch_target,
                        y=batch_input,
                        params=list(self.parameters()),
                        DC=self.DC,
                        DC_type=self.DC_type,
                        forward_op=self.forward_op,
                        adjoint_op=self.adjoint_op,
                        noise_type=self.noise_type,
                        sigma=self.sigma_noise,
                        Rtheta=self.Rtheta,
                        nabla_x_network=self.nabla_x_network,
                        lambda_dc=self.tau0,
                        lambda_Rtheta=self.lambda_Rtheta,
                        gamma=self.gamma,
                        eta=self.eta,
                        theta=self.theta,
                        accelerated=self.accelerated,
                        backtracking=self.backtracking,
                        K_JFB=self.K_JFB,
                    )

                else:
                    # Standard backward on loss
                    loss = self.criterion(outputs, batch_target)
                    loss.backward()
                
                self.optimizer.step()

                with torch.no_grad():
                    if self.learn_theta_interpol:
                        self.theta.clamp_(0.0, 1.0)

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

            iter_nums.append(np.mean(epoch_iter_nums))

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
                    
                    if self.noise_type == 'gaussian':
                        batch_input = batch_input + self.sigma_noise * torch.randn_like(batch_input)
                    
                    batch_mask = batch_mask["mask"].to(self.device).float()
                    if self.problem == "MRI":
                        batch_input = batch_input * batch_mask  # For MRI, ensure input is consistent with mask
                    else:
                        batch_input = torch.clamp(batch_input, 0, 1.0)  # Ensure input is in valid range

                    batch_target = batch_target.to(self.device).float()     

                    outputs, _, stats = self.forward(batch_input, batch_mask)
                        
                    epoch_iter_nums_val.append(stats["iter_num"])

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

                iter_nums_val.append(np.mean(epoch_iter_nums_val))

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
                
                if plot_interval is not None and epoch % plot_interval == 0:
                    plt.figure()
                    plt.plot(iter_nums, label='Train Iterations')
                    plt.plot(iter_nums_val, label='Val Iterations')
                    plt.xlabel('Epochs')
                    plt.ylabel('Number of Iterations to Convergence')
                    plt.title('Iterations to Convergence')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.path, f'convergence_stats_epoch_{epoch}.pdf'))
                    plt.close()

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
            timer_end_epoch = time.time()
            times_epoch.append(timer_end_epoch - timer_start_epoch)

        print("Training complete.")
        timer_end_training = time.time()
        time_total = timer_end_training - timer_start_training
        print(f"Total training time: {time_total:.2f} seconds")
        mean_time_epoch = np.mean(times_epoch)
        print(f"Mean time per epoch: {mean_time_epoch:.2f} seconds")

        # on écrit les statistiques dans un fichier txt
        with open(os.path.join(self.path, "training_stats.txt"), "w") as f:
            for epoch in range(len(train_losses)):
                f.write(
                    f"Epoch {epoch+1}: Train MSE={train_losses[epoch]:.6f}, "
                    f"Val MSE={val_losses[epoch]:.6f}, "
                    f"Train PSNR={train_PSNRs[epoch]:.2f} dB, "
                    f"Val PSNR={val_PSNRs[epoch]:.2f} dB, "
                    f"Train SSIM={train_ssim[epoch]:.4f}, "
                    f"Val SSIM={val_ssim[epoch]:.4f}, "
                    f"Iter Train={iter_nums[epoch]}, "
                    f"Iter Val={iter_nums_val[epoch]}, "
                    f"Time Epoch={times_epoch[epoch]:.2f} sec\n"
                )
            f.write(f"Total training time: {time_total:.2f} seconds\n")
            f.write(f"Mean time per epoch: {mean_time_epoch:.2f} seconds\n")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_PSNRs": train_PSNRs,
            "val_PSNRs": val_PSNRs,
            "train_ssim": train_ssim,
            "val_ssim": val_ssim,
            "iter_nums": iter_nums,
            "iter_nums_val": iter_nums_val,
            "times_epoch": times_epoch,
            "time_total": time_total
        }
    
    def evaluate(self, 
                 test_loader, 
                 init_train=None, 
                 n_display=1, 
                 accelerated=False,
                 andersen_acceleration=False, 
                 m_andersen=5,
                 cycle_andersen=False,
                 PnP=False, 
                 pretrained_path=None):

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
        mean_time = []
        n_collected = 0

        # convergence plots
        psnr_per_iter_accumulator = None
        eps_per_iter_accumulator = None
        n_batches = 0

        self.accelerated = accelerated
        self.init_train = init_train

        self.andersen_acceleration = andersen_acceleration
        self.m_andersen = m_andersen
        self.cycle_andersen = cycle_andersen

        if self.accelerated and self.andersen_acceleration:
            print("Use of Andersen and Inertia. Inertia will be desabled")
            self.accelerated = False  # Disable inertia if Anderson acceleration is enabled, as they can interfere with each other

        with torch.no_grad():
            for batch_target, batch_input, batch_mask in test_loader:
                time_reconstruct = time.time()
                batch_input = batch_input.to(self.device).float()
                    
                B = batch_input.shape[0]

                if self.noise_type == 'gaussian':
                    batch_input = batch_input + self.sigma_noise * torch.randn_like(batch_input)
                    
                batch_mask = batch_mask["mask"].to(self.device).float()
                if self.problem == "MRI":
                    batch_input = batch_input * batch_mask  # For MRI, ensure input is consistent with mask
                else:
                    batch_input = torch.clamp(batch_input, 0, 1.0)  # Ensure input is in valid range
                batch_target = batch_target.to(self.device).float()              

                if PnP:
                    outputs, intermediates, stats = self.forward_PnP(batch_input, batch_mask)
                else:
                    print("Running standard DEQ forward for evaluation...")
                    outputs, intermediates, stats = self.forward(batch_input, batch_mask)
                time_reconstruct = time.time() - time_reconstruct
                mean_time.append(time_reconstruct)

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
                epsilons = stats["epsilons"]

                if eps_per_iter_accumulator is None:
                    eps_per_iter_accumulator = np.zeros(len(epsilons))

                eps_per_iter_accumulator[:len(epsilons)] += np.array(epsilons)

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

            test_mse = float(np.mean(test_mse))
            test_PSNR = float(np.mean(test_PSNR))
            test_SSIM = float(np.mean(test_SSIM))

            input_mse = float(np.mean(input_mse))
            input_PSNR = float(np.mean(input_PSNR))
            input_SSIM = float(np.mean(input_SSIM))

            mean_time = float(np.mean(mean_time))

            print(f"Test MSE: {test_mse:.6f}, Test PSNR: {test_PSNR:.2f} dB")
            print(f"Input MSE: {input_mse:.6f}, Input PSNR: {input_PSNR:.2f} dB")
            print(f"Test SSIM: {test_SSIM:.4f}, Input SSIM: {input_SSIM:.4f}")

            # ---------------------------------------
            # PSNR vs iteration
            # ---------------------------------------
            if psnr_per_iter_accumulator is not None:
                psnr_curve = psnr_per_iter_accumulator / n_batches

                plt.figure()
                plt.plot(psnr_curve)
                plt.xlabel("Iteration")
                plt.ylabel("PSNR (dB)")
                plt.title("PSNR evolution")
                plt.tight_layout()
                plt.savefig(os.path.join(self.path, "psnr_evolution.pdf"), dpi=300)
                plt.close()

            # ---------------------------------------
            # epsilon vs iteration
            # ---------------------------------------
            if eps_per_iter_accumulator is not None:
                eps_curve = eps_per_iter_accumulator / n_batches

                plt.figure()
                plt.plot(eps_curve)
                plt.xlabel("Iteration")
                plt.ylabel("Relative error")
                plt.title("Convergence metric")
                plt.tight_layout()
                plt.yscale("log")
                plt.savefig(os.path.join(self.path, "epsilon_evolution.pdf"), dpi=300)
                plt.close()

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
                    plt.savefig(os.path.join(self.path, f"test_reconstruction_{i}.pdf"))
                    plt.close()

                    # Only the reconstruction
                    plt.figure(figsize=(5,5))
                    ax = plt.subplot(1,1,1)
                    show_image(
                        ax,
                        image,
                        f"PSNR {psnr_recon:.2f}, SSIM {ssim_recon:.4f}"
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.path, f"test_reconstruction_only_{i}.pdf"), dpi=300)
                    plt.close()

                    # Only the ground truth
                    plt.figure(figsize=(5,5))
                    ax = plt.subplot(1,1,1)
                    show_image(
                        ax,
                        target,
                        "Ground Truth"
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.path, f"test_ground_truth_only_{i}.pdf"), dpi=300)
                    plt.close()

                    # Only the input image
                    plt.figure(figsize=(5,5))
                    ax = plt.subplot(1,1,1)
                    show_image(
                        ax,
                        input_image,
                        f"Zero-Filled\nPSNR {psnr_input_img:.2f}, SSIM {ssim_input_img:.4f}"
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.path, f"test_input_image_only_{i}.pdf"), dpi=300)
                    plt.close()

        times = stats["times"] if "times" in stats else None
        with open(os.path.join(self.path, "test_times.txt"), "w") as f:
            f.write(f"Time per iteration: {times}\n")

        return {
            "test_mse": test_mse,
            "test_PSNR": test_PSNR,
            "input_mse": input_mse,
            "input_PSNR": input_PSNR,
            "test_SSIM": test_SSIM,
            "input_SSIM": input_SSIM,
            "PSNR_list": psnr_curve,
            "mean_time": mean_time,
        }
