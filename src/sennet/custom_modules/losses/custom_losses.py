import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Apply sigmoid to input (predictions)
        input_sigmoid = torch.sigmoid(input).view(-1)
        target_flat = target.view(-1)
        
        intersection = (input_sigmoid * target_flat).sum()

        return 1 - (
            (2.0 * intersection + self.smooth)
            / (input_sigmoid.sum() + target_flat.sum() + self.smooth)
        )


class BCELoss(nn.Module):
    def forward(self, bce_loss):
        return bce_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, bce_loss):
        p_t = torch.exp(-bce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * bce_loss
        return focal_loss.mean()
