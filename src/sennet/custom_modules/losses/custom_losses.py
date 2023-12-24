import pytorch_toolbelt.losses as ptb_loss
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


class LovaszLoss(nn.Module):
    def __init__(self, per_image: bool = False, ignore_index=None):
        nn.Module.__init__(self)
        self.loss = ptb_loss.BinaryLovaszLoss(per_image=per_image, ignore_index=ignore_index)
        # self.loss = ptb_loss.LovaszLoss(per_image=per_image)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (b, z, h, w)
        # target: (b, z, h, w)
        # note: none of this really work
        num_channels = logits.shape[1]
        loss = 0
        for i in range(num_channels):
            loss += self.loss(logits[:, i, :, :], target[:, i, :, :].long())
        return loss / (num_channels + 1e-6)
        # num_channels = logits.shape[1]
        # loss = 0
        # for i in range(num_channels):
        #     loss += self.loss(
        #         torch.cat([
        #             torch.zeros_like(logits[:, i, :, :]).unsqueeze(1),
        #             torch.nn.functional.sigmoid(logits[:, i, :, :]).unsqueeze(1),
        #         ], dim=1),
        #         target[:, i, :, :]
        #     )
        # return loss / (num_channels + 1e-6)


class PtbLoss(nn.Module):
    def __init__(self, version: str, kwargs: dict[str, any]):
        nn.Module.__init__(self)
        loss_class = getattr(ptb_loss, version)
        self.loss = loss_class(**kwargs)

    def forward(self, logits, target):
        return self.loss(logits, target)
