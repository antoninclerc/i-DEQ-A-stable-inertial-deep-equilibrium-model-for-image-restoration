import torch
import torch.nn as nn

from networks.ResUnet import UNetRes as Net
import deepinv as dinv

import torch
from torch.hub import load_state_dict_from_url

from deepinv.models.drunet import DRUNet
from deepinv.models.GSPnP import GSPnP

class Drunet(torch.nn.Module):# DRUNet model definition 
    def __init__(self, n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=False, pretrained=None):
        super(Drunet, self).__init__()
        self.model = Net(in_channels=n_channels+1, out_channels=n_channels, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.detach = False
        self.model.load_state_dict(torch.load(pretrained), strict=True)
        for k, v in self.model.named_parameters():
            v.requires_grad = False

    def forward(self, x, sigma):
        '''
        x : image with values in [0, 1]
        sigma : standard deviation of denoising in [0, 1]
        '''
        sigma = float(sigma)
        sigma_div_255 = torch.FloatTensor([sigma]).repeat(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.model(x)
    
    def potential(self, x, sigma):
        N = self.student_grad(x, sigma)
        return 0.5 * torch.norm((x - N).reshape(x.shape[0], -1), p=2, dim=-1) ** 2


def GSDRUNet(
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
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelsuffle" for pixel
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
    GSmodel = dinv.models.GSPnP.GSPnP(denoiser, alpha=alpha, detach=False)
    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage, weights_only=False)

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        GSmodel.load_state_dict(ckpt, strict=False)
        GSmodel.eval()
    return GSmodel