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