import torch
import torch.nn as nn

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=2):
        super(PixelShuffleUpsample, self).__init__()
        # Assuming the input has 'C' channels, and you want to upscale by a factor of 2,
        # the number of output channels from the convolution should be C * (scale_factor ** 2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    

class ConvTranspose3DUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(ConvTranspose3DUpsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels * upscale_factor**3,  # Upscale factor cubed for 3D
            kernel_size=3,
            stride=upscale_factor,
            padding=1,
            output_padding=1
        )
        self.conv_refine = nn.Conv3d(
            in_channels=out_channels * upscale_factor**3,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.conv_refine(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # initialize weight and bias with a mean of 1 and a small standard deviation
        self.weight = nn.Parameter(torch.randn(num_channels) * 0.05 + 0.1)
        self.bias = nn.Parameter(torch.randn(num_channels) * 0.01)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x