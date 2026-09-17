from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
import urllib.error
import contextlib
import socket
import io
import os
import numpy as np

def get_weights_url(model_name, file_name):
    return (
        "https://huggingface.co/deepinv/"
        + model_name
        + "/resolve/main/"
        + file_name
        + "?download=true"
    )

def load_state_dict_from_url(*args, **kwargs) -> dict:
    """
    A wrapper for :func:`torch.hub.load_state_dict_from_url` that respects the `DEEPINV_DOWNLOAD_VERBOSE`
    environment variable. If set to 0, stdout prints are suppressed.

    Network-level failures (HTTP errors, connection failures, timeouts) are
    re-raised as :class:`deepinv.utils.DownloadError` so they can be handled
    uniformly with other deepinv downloads.
    """
    # Read the environment variable. Default to "1" (True/Verbose) if not set.
    env_value = os.environ.get("DEEPINV_DOWNLOAD_VERBOSE", "1").lower()

    # Check if the user explicitly turned verbosity off
    is_silent = env_value in ("0", "false", "no", "f")

    # Choose the context manager based on the is_silent flag
    if is_silent:
        ctx = contextlib.redirect_stdout(io.StringIO())
        # Optional: Also force progress=False to hide the stderr progress bar
        kwargs["progress"] = False
    else:
        # nullcontext() does nothing, allowing stdout to print normally
        ctx = contextlib.nullcontext()

    try:
        with ctx:
            return torch.hub.load_state_dict_from_url(*args, **kwargs)
    except (
        urllib.error.URLError,
        ConnectionError,
        TimeoutError,
        socket.gaierror,
    ) as exc:
        url = args[0] if args else kwargs.get("url", "<unknown>")
        raise ValueError(
            f"Failed to download the file from {url}. Please check your internet connection or try again later."
        ) from exc

class Denoiser(torch.nn.Module):
    r"""
    Base class for denoiser models.

    Provides a template for defining denoiser models.

    While most denoisers :math:`\denoisername` are designed to handle Gaussian noise
    with variance :math:`\sigma^2`, this is not mandatory.

    .. note::

        A Denoiser can be converted into a :class:`Reconstructor <deepinv.models.Reconstructor>`
        by using the :class:`deepinv.models.ArtifactRemoval` class.

    The base class inherits from :class:`torch.nn.Module`.

    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, x: Tensor, sigma: float | Tensor, **kwargs) -> torch.Tensor:
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.
        The input `x` is expected to be with pixel values in `[0, 1]` range, up to random noise. The output is also expected to be in `[0, 1]` range.

        :param torch.Tensor x: noisy input, of shape `[B, C, H, W]`.
        :param torch.Tensor, float sigma: noise level. Can be a `float` or a :class:`torch.Tensor` of shape `[B]`.
            If a single `float` is provided, the same noise level is used for all samples in the batch.
            Otherwise, batch-wise noise levels are used.

        :returns: (:class:`torch.Tensor`) Denoised tensor.
        """
        raise NotImplementedError()

    @staticmethod
    def _handle_sigma(
        sigma: float | torch.Tensor | list[float],
        batch_size: int = None,
        ndim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwarg,
    ) -> torch.Tensor:
        r"""
        Convert various noise level types to the appropriate format for batch processing.
            If `sigma` is a single float or int, the same value will be used for each sample in the batch.
            If `sigma` is a tensor, it should be of shape `(batch_size,)` or a scalar.
            If `sigma` is a list, it should be of length `batch_size` or `1`.

        To be overridden by subclasses if necessary.

        :param float, torch.Tensor sigma: noise level.
        :param int batch_size: number of samples in the batch (optional).
        :param int ndim: number of dimensions of the input tensor (optional).
        :param torch.device device: device to which the tensor should be moved (optional).
        :param torch.dtype dtype: data type of the tensor (optional).
        :param args: additional positional arguments.
        :param kwarg: additional keyword arguments.

        :returns: noise levels for each sample in the batch adapted to the denoiser.
        """
        if isinstance(sigma, (float, int)):
            sigma = float(sigma)
        elif isinstance(sigma, torch.Tensor):
            sigma = sigma.squeeze().to(dtype=dtype, device=device)
        elif isinstance(sigma, list):
            sigma = torch.tensor(sigma, dtype=dtype, device=device).squeeze()
        elif isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma, dtype=dtype, device=device).squeeze()
        else:
            raise TypeError(
                f"Sigma must be a float, int, or torch.Tensor. Got {type(sigma)}."
            )

        # Will reshape to (batch_size,) if batch_size is not None
        if batch_size is not None:
            # duplicate sigma for each sample in the batch
            if isinstance(sigma, float):
                sigma = torch.tensor([sigma] * batch_size, dtype=dtype, device=device)
            elif sigma.ndim == 0:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) == 1:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) != batch_size:
                raise ValueError(
                    f"Sigma tensor size {sigma.size(0)} does not match batch size {batch_size}."
                )

        # Will reshape to (batch_size, 1, ..., 1) if ndim is not None
        if ndim is not None:
            if isinstance(sigma, float):
                sigma = torch.tensor(sigma, dtype=dtype, device=device).view(
                    1, *([1] * (ndim - 1))
                )
            elif sigma.ndim == 0:
                sigma = sigma.view(1, *([1] * (ndim - 1)))
            elif sigma.ndim == 1:
                sigma = sigma.view(-1, *([1] * (ndim - 1)))
            else:
                raise ValueError(
                    f"Sigma tensor has {sigma.ndim} dimensions, expected 0 or 1."
                )
        return sigma



class StudentGrad(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.model = denoiser

    def forward(self, x, sigma):
        return self.model(x, sigma)


class GSPnP_MRI(Denoiser):
    r"""
    Gradient Step module to use a denoiser architecture as a Gradient Step Denoiser.

    See :footcite:t:`hurault2021gradient`. Code from https://github.com/samuro95/GSPnP.

    :param torch.nn.Module denoiser: Denoiser model.
    :param float alpha: Relaxation parameter
    :param bool detach: If `True`, the denoiser output will be detached from the computation graph.
        Setting this to `False` allows one to compute the gradient of the denoiser output with respect to the input, it is necessary in training.
        Default is `True`.
    """

    def __init__(self, denoiser, alpha: float = 1.0, detach: bool = False):
        super().__init__()
        self.student_grad = StudentGrad(denoiser)
        self.alpha = alpha
        self.detach = detach

    def potential(
        self, x: Tensor, sigma: float | torch.Tensor, *args, **kwargs
    ) -> Tensor:
        
        if x.shape[1] == 2 and  x.ndim == 4:
            # x is complex, we want a 0 channel for the phase, so we add a channel of zeros
            MRI = True
            x = x[:, 0, ...].unsqueeze(1)
        else:
            MRI = False

        N = self.student_grad(x, sigma)
        potential = (
            0.5
            * self.alpha
            * torch.linalg.vector_norm(x - N, dim=tuple(range(1, x.ndim)), ord=2) ** 2
        )
        if MRI:
            zero_channel = torch.zeros_like(potential)
            potential = torch.cat((potential, zero_channel), dim=1)

        return potential

    def potential_grad(
        self, x: Tensor, sigma: float | torch.Tensor, *args, **kwargs
    ) -> Tensor:
        r"""
        Calculate :math:`\nabla g` the gradient of the regularizer :math:`g` at input :math:`x`.

        :param torch.Tensor x: Input image
        :param float sigma: Denoiser level :math:`\sigma` (std)
        """
        if x.shape[1] == 2 and  x.ndim == 4:
            # x is complex, we want a 0 channel for the phase, so we add a channel of zeros
            MRI = True
            x = x[:, 0, ...].unsqueeze(1)
        else:
            MRI = False

        with torch.enable_grad():
            x = x.to(torch.float32)
            x = x.requires_grad_()
            N = self.student_grad(x, sigma)
            JN = torch.autograd.grad(
                N, x, grad_outputs=x - N, create_graph=False if self.detach else True
            )[0]
        if self.detach:
            x = x.detach()
            JN = JN.detach()

        Dg = x - N - JN

        if MRI:
            zero_channel = torch.zeros_like(Dg)
            Dg = torch.cat((Dg, zero_channel), dim=1)

        return self.alpha * Dg

    def forward(self, x: Tensor, sigma: float | torch.Tensor) -> Tensor:
        r"""
        Denoising with Gradient Step Denoiser

        :param torch.Tensor x: Input image
        :param float sigma: Denoiser level (std)
        """
        Dg = self.potential_grad(x, sigma)

        if x.shape[1] == 2 and  x.ndim == 4:
            # x is complex, we want a 0 channel for the phase, so we add a channel of zeros
            x = x[:, 0, ...].unsqueeze(1)
            zero_channel = torch.zeros_like(x)
            x = torch.cat((x, zero_channel), dim=1)

        x_hat = x - Dg
        return x_hat


def GSDRUNet_MRI(
    alpha=1.0,
    in_channels=3,
    out_channels=3,
    nb=2,
    nc=(64, 128, 256, 512),
    act_mode="E",
    pretrained=None,
    device=torch.device("cpu"),
):
    """
    Gradient Step Denoiser with DRUNet architecture.

    Based on the GSPnP method from :footcite:t:`hurault2021gradient`.

    :param float alpha: Relaxation parameter
    :param int in_channels: Number of input channels
    :param int out_channels: Number of output channels
    :param int nb: Number of blocks in the DRUNet
    :param Sequence[int,int,int,int] nc: number of channels per convolutional layer in the DRUNet. The network has a fixed number of 4 scales with ``nb`` blocks per scale (default: ``[64,128,256,512]``).
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU "E" for ELU and "S" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelshuffle" for pixel
        shuffling, and "upconv" for nearest neighbour upsampling with additional convolution.
    :param bool download: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param str device: gpu or cpu.
    """
    from deepinv.models.drunet import DRUNet

    denoiser = DRUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        nb=nb,
        nc=nc,
        act_mode=act_mode,
        pretrained=None,
        device=device,
    )
    GSmodel = GSPnP_MRI(denoiser, alpha=alpha)
    if pretrained:
        if pretrained == "download":
            if in_channels == 3:
                file_name = "GSDRUNet_torch.ckpt"
            elif in_channels == 1:
                file_name = "GSDRUNet_grayscale_torch.ckpt"
            url = get_weights_url(model_name="gradientstep", file_name=file_name)
            ckpt = load_state_dict_from_url(
                url,
                map_location=lambda storage, loc: storage,
                file_name=file_name,
            )
        else:
            ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        GSmodel.load_state_dict(ckpt, strict=False)
        GSmodel.eval()
    return GSmodel