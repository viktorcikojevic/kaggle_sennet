from sennet.custom_modules.metrics.surface_dice_metric_fast import create_table_neighbour_code_to_surface_area
import segmentation_models_pytorch.losses as smp_loss
import pytorch_toolbelt.losses as ptb_loss
from line_profiler_pycharm import profile
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Apply sigmoid to input (predictions)
        input_sigmoid = torch.sigmoid(input).reshape(-1)
        target_flat = target.reshape(-1)
        
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


class MccLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.loss = smp_loss.MCCLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (b, z, h, w)
        # target: (b, z, h, w)
        loss = self.loss(torch.nn.functional.sigmoid(logits), target)
        return loss


class PtbLoss(nn.Module):
    def __init__(self, version: str, kwargs: dict[str, any]):
        nn.Module.__init__(self)
        loss_class = getattr(ptb_loss, version)
        self.loss = loss_class(**kwargs)

    def forward(self, logits, target):
        return self.loss(logits, target)


class SurfaceDiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, device="cuda"):
        nn.Module.__init__(self)
        self.smooth = smooth
        self.area = create_table_neighbour_code_to_surface_area((1, 1, 1))
        self.area = torch.from_numpy(self.area).to(device)  # torch.float32
        self.unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)
        self.device = device
        self.bases = torch.tensor([
            [c == "1" for c in f"{i:08b}"[::-1]]
            for i in range(256)
        ]).float().to(device)
        self.n_bases = self.bases.shape[0]

    @profile
    def _forward_z_pair(self, pred, labels):
        # pred = (batch, z, y, x): logits
        # labels = (batch, z, y, x): labels (0 or 1 only)

        # unfold: group 8 corners together
        unfolded_pred = self.unfold(pred)
        unfolded_labels = self.unfold(labels.float())

        batch_size, n_corners, n_points = unfolded_pred.shape
        assert n_corners == 8, f"{n_corners=} == 8, are you working on 2 slices?"
        weights = torch.zeros((batch_size, self.n_bases, n_points), device=self.device)

        # greedy solve weights
        done = False
        copied_unfolded_pred = unfolded_pred.clone()
        for i in range(self.n_bases):
            done = (copied_unfolded_pred.abs() < 1e-5).all()
            if done:
                break

            nonzero_masks = copied_unfolded_pred > 0

            pred_cube_bytes = sum(nonzero_masks[:, k, :] << k for k in range(8))
            pred_cube_bases = self.bases[pred_cube_bytes, :].permute((0, 2, 1))
            subtraction_weights = torch.where(copied_unfolded_pred > 0, copied_unfolded_pred, torch.inf).min(1).values  # (batch, n_points)
            inf_mask = subtraction_weights.isinf()
            subtraction_weights[inf_mask] = 0
            # subtraction_weights = torch.where(subtraction_weights.isinf(), 0, subtraction_weights)

            for b in range(batch_size):
                w = subtraction_weights[b]
                weights[b][pred_cube_bytes[b], torch.arange(weights.shape[2], device=self.device)] += w
                copied_unfolded_pred[b] -= w[None, :] * pred_cube_bases[b]
        assert done, f"solver failed"

        # computing label areas
        label_cubes_byte = sum(unfolded_labels[:, k, :].to(torch.int32) << k for k in range(8))

        label_areas = torch.zeros((label_cubes_byte.shape[0], label_cubes_byte.shape[1]), dtype=torch.float32, device=self.device)
        for b in range(label_cubes_byte.shape[0]):
            label_areas[b, :] = self.area[label_cubes_byte[b, :]]

        # computing pred areas
        pred_areas = (weights.permute((0, 2, 1)) @ self.area.tile((batch_size, 1)).unsqueeze(-1)).squeeze(-1)

        numerator = torch.zeros(label_cubes_byte.shape[0], dtype=torch.float32, device=self.device)
        for b in range(label_cubes_byte.shape[0]):
            idx = torch.logical_and(pred_areas[b] > 0, label_areas[b] > 0)
            numerator[b] = label_areas[b][idx].sum() + pred_areas[b][idx].sum()
        denominator = label_areas.sum(1) + pred_areas.sum(1)

        # idx = (pred_areas > 0) & (label_areas > 0)
        # area_sum = label_areas + pred_areas
        # numerator2 = (area_sum * idx).sum(1)
        # denominator2 = area_sum.sum(1)
        # assert (numerator2 - numerator).abs().sum() < 1e-3
        # assert (denominator2 - denominator).abs().sum() < 1e-3

        # aaa_pred_areas_np = pred_areas.numpy().reshape((-1, 1))
        # aaa_label_areas_np = label_areas.numpy().reshape((-1, 1))
        # aaa_label_cubes_byte = label_cubes_byte.numpy().reshape((-1, 1))
        # aaa_max_weights_values = weights[0].max(0).values.numpy().reshape((-1, 1))
        # aaa_pred_cubes_byte_w = weights[0].max(0).indices.numpy().reshape((-1, 1))
        # aaa_pred_label_diff = (unfolded_pred - unfolded_labels).abs().sum()
        # residual = (self.bases.permute((1, 0)).tile((batch_size, 1, 1)) @ weights) - unfolded_pred
        # aaa_residual = residual.abs().sum()
        # aaa_area_diff = aaa_pred_areas_np - aaa_label_areas_np
        # aaa_bytes_diff = aaa_pred_cubes_byte_w - aaa_label_cubes_byte
        # aaa_area_diff_sum = (pred_areas - label_areas).abs().sum()
        return numerator, denominator
        # return numerator2, denominator2

    @profile
    def forward_sigmoid(self, pred_sigmoid, labels):
        batch_size, zs, ys, xs = pred_sigmoid.shape
        assert pred_sigmoid.shape == labels.shape
        assert zs >= 2, f"zs not >= 2, {pred_sigmoid.shape=}"
        num = 0
        denom = 0
        for z_start in range(zs-1):
            z_end = z_start + 2
            pair_num, pair_denom = self._forward_z_pair(pred_sigmoid[:, z_start: z_end, ...], labels[:, z_start: z_end, ...])
            num += pair_num
            denom += pair_denom
        dice = 1 - ((num + self.smooth) / (denom + self.smooth))
        return dice.mean()

    @profile
    def forward(self, pred, labels):
        pred_sigmoid = torch.sigmoid(pred)
        return self.forward_sigmoid(pred_sigmoid, labels)


if __name__ == "__main__":
    import time
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    _device = "cuda"
    # _device = "cpu"
    _batch_size = 1
    _zs = 2
    _ys = 3
    _xs = 3
    _pred = torch.rand((_batch_size, _zs, _ys, _xs)).to(_device)
    _labels = (torch.rand((_batch_size, _zs, _ys, _xs)) > 0.5).to(_device).float()
    _criterion = SurfaceDiceLoss(device=_device)

    _t0 = time.time()
    _loss = _criterion.forward_sigmoid(_pred, _labels)
    _loss = _loss.cpu()
    _loss_label = _criterion.forward_sigmoid(_labels, _labels)
    _loss_label = _loss_label.cpu()
    _t1 = time.time()
    print(f"loss={_loss}, loss_label={_loss_label}, t={_t1 - _t0}")

    # _raw_pred = torch.log(_pred / (1 - _pred + 1e-6))
    # _var = _raw_pred.clone().detach().requires_grad_(True)
    # _optim = torch.optim.SGD([_var], lr=1.0)
    # for _i in range(10000):
    #     _optim.zero_grad()
    #     _loss = _criterion(_var, _labels)
    #     _loss.backward()
    #     _optim.step()
    #     if _i % 100 == 0:
    #         print(f"[{_i=}]: {_loss=}")
