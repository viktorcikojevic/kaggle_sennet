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


class TopKPercentBCELoss(nn.Module):
    def __init__(self, k_percent=0.1):
        super(TopKPercentBCELoss, self).__init__()
        self.k_percent = k_percent

    def forward(self, bce_loss):
        # Flatten the BCE loss tensor
        bce_loss_flat = bce_loss.view(-1)

        # Calculate the number of elements to keep (top k percent)
        num_elements = int(self.k_percent * bce_loss_flat.size(0))

        # Sort the BCE loss and select the top k percent
        topk_loss, _ = torch.topk(bce_loss_flat, num_elements, sorted=False)

        # Calculate the mean loss of the top k percent pixels
        topk_mean_loss = topk_loss.mean()

        return topk_mean_loss
def _2x2x2_unfold(tensor: torch.Tensor) -> torch.Tensor:
    # tensor: (b, z, h, w)
    # returns: (b, 8, num_points), unfolded tensor for marching cube
    batch_size = tensor.shape[0]
    tall_unfolded = torch.nn.functional.unfold(tensor, kernel_size=(2, 2), padding=1)
    unfolded = tall_unfolded.unfold(dimension=1, size=8, step=4).permute((0, 3, 1, 2)).reshape((batch_size, 8, -1))
    return unfolded


# shamelessly copied over from Igor
class BoundaryDoULossV2(nn.Module):
    def __init__(self, n_classes=1, allowed_outlier_fraction=0.25):
        super(BoundaryDoULossV2, self).__init__()
        self.n_classes = n_classes
        self.allowed_outlier_fraction = allowed_outlier_fraction

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).half()
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
        )
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - h + 1,
                padding_out.shape[2] - w + 1,
            )
        ).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0).cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(
            alpha, 0.8
        )  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return loss

    def forward(self, inputs, target):
        inputs = inputs.sigmoid()

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])

        # Apply outlier fraction logic to BoundaryDoULoss
        output = inputs[:, 0]  # Assuming binary classification

        output = output.float()
        target = target.float()

        pos_mask = target.eq(1.0)
        neg_mask = ~pos_mask

        pt = output.sigmoid().clamp(1e-6, 1 - 1e-6)

        neg_loss = (
            -torch.pow(pt, 2) * torch.nn.functional.logsigmoid(-output) * neg_mask
        )

        if self.allowed_outlier_fraction < 1:
            neg_loss = neg_loss.flatten()
            M = neg_loss.numel()
            num_elements_to_keep = int(M * (1 - self.allowed_outlier_fraction))
            neg_loss, _ = torch.topk(
                neg_loss, k=num_elements_to_keep, largest=False, sorted=False
            )

        neg_loss_reduced = neg_loss.sum() / (neg_mask.sum() + 1e-6)
        loss_outlier = neg_loss_reduced

        return (loss + loss_outlier) / (self.n_classes + 1)


class SurfaceDiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, device="cuda", verbose: bool = False):
        nn.Module.__init__(self)
        self.verbose = verbose
        self.smooth = smooth
        self.area = create_table_neighbour_code_to_surface_area((1, 1, 1))
        self.area = torch.from_numpy(self.area).to(device)  # torch.float32
        self.device = device
        self.bases = torch.tensor([
            [c == "1" for c in f"{i:08b}"[::-1]]
            for i in range(256)
        ]).float().to(device)
        self.n_bases = self.bases.shape[0]

    @profile
    def _forward_z_pair(
            self,
            pred: torch.Tensor,
            labels: torch.Tensor,
            raw_pred: torch.Tensor | None = None,
    ):
        # pred = (batch, z, y, x): sigmoids
        # labels = (batch, z, y, x): labels (0 or 1 only)
        # raw_pred = (batch, z, y, x): logits: advantageous when doing 16-bit precisions

        # unfold: group 8 corners together
        if raw_pred is not None:
            unfolded_raw_pred = torch.nn.functional.unfold(raw_pred, kernel_size=(2, 2), padding=1)
        else:
            unfolded_raw_pred = None
        unfolded_pred = torch.nn.functional.unfold(pred, kernel_size=(2, 2), padding=1)
        unfolded_labels = torch.nn.functional.unfold(labels.float(), kernel_size=(2, 2), padding=1)
        unfolded_labels_int = unfolded_labels.to(torch.int32)

        batch_size, n_corners, n_points = unfolded_pred.shape
        if n_corners != 8:
            raise RuntimeError(f"{n_corners=} == 8, are you working on 2 slices?")
        pred_weights_in_label_cubes = torch.zeros((batch_size, n_points), device=self.device)

        label_cubes_byte = sum(unfolded_labels_int[:, k, :] << k for k in range(8))

        # these blocks below are the same, just trading increased time for less space requirement
        # 16 comes from: np.linalg.norm(np.ones(256) - np.zeros(256)) == 16
        # basis_weights = 1 - torch.norm(unfolded_pred[:, None, :, :] - self.bases[None, :, :, None], dim=2) / 16.0
        # pred_cube_bytes = torch.argmax(basis_weights, dim=1)
        # NOTE: if we wanna go more efficient we can start chunking the points with torch.chunk
        pred_cube_bytes = torch.zeros((batch_size, n_points), device=self.device, dtype=torch.int32)
        for i in range(self.n_bases):
            this_basis_weight = 1 - torch.norm(unfolded_pred - self.bases[None, i, :, None], dim=1) / 16.0
            larger_mask = this_basis_weight > pred_cube_bytes
            pred_cube_bytes = torch.where(larger_mask, i, pred_cube_bytes)

            is_label_cube_mask = i == label_cubes_byte
            pred_weights_in_label_cubes[is_label_cube_mask] = this_basis_weight[is_label_cube_mask]

        # computing areas
        pred_areas = self.area[pred_cube_bytes]
        label_areas = self.area[label_cubes_byte]

        intersection = (pred_weights_in_label_cubes * label_areas).sum(1)
        numerator = 2*intersection
        denominator = label_areas.sum(1) + pred_areas.sum(1)

        selected_volumetric_loss = torch.zeros((batch_size, ), device=self.device)
        volumetric_loss_mask = (label_cubes_byte == 0) | (label_cubes_byte == 255)
        for b in range(batch_size):
            if raw_pred is not None:
                # it's taking the mean over the 8 corner points
                selected_volumetric_loss[b] = torch.nn.functional.binary_cross_entropy_with_logits(
                    unfolded_raw_pred[b, :, volumetric_loss_mask[b]],
                    unfolded_labels[b, :, volumetric_loss_mask[b]],
                    reduction="mean",
                )
            else:
                # it's taking the mean over the 8 corner points
                selected_volumetric_loss[b] = torch.nn.functional.binary_cross_entropy(
                    unfolded_pred[b, :, volumetric_loss_mask[b]],
                    unfolded_labels[b, :, volumetric_loss_mask[b]],
                    reduction="mean",
                )
        return numerator, denominator, selected_volumetric_loss

    @profile
    def forward_sigmoid(
            self,
            pred_sigmoid: torch.Tensor,
            labels: torch.Tensor,
            pred: torch.Tensor | None
    ):
        batch_size, zs, ys, xs = pred_sigmoid.shape
        assert pred_sigmoid.shape == labels.shape
        assert zs >= 2, f"zs not >= 2, {pred_sigmoid.shape=}"
        num = 0
        denom = 0
        vol = 0
        # blank_slice = torch.zeros((batch_size, 1, ys, xs), dtype=pred_sigmoid.dtype, device=self.device)
        for z_start in range(-1, zs, 1):
            z_end = z_start + 2
            if z_start < 0:
                continue
                # pair_num, pair_denom, pair_vol = self._forward_z_pair(
                #     torch.cat([blank_slice, pred_sigmoid[:, 0: z_end, ...]], dim=1),
                #     torch.cat([blank_slice, labels[:, 0: z_end, ...]], dim=1),
                # )
            elif z_end <= zs:
                if pred is None:
                    pred_slice = None
                else:
                    pred_slice = pred[:, z_start: z_end, ...]
                pair_num, pair_denom, pair_vol = self._forward_z_pair(
                    pred_sigmoid[:, z_start: z_end, ...],
                    labels[:, z_start: z_end, ...],
                    raw_pred=pred_slice
                )
            else:
                continue
                # assert z_start == zs-1, f"too high {z_start=}, expected {zs-1}"
                # pair_num, pair_denom, pair_vol = self._forward_z_pair(
                #     torch.cat([pred_sigmoid[:, z_start: z_start + 1, ...], blank_slice], dim=1),
                #     torch.cat([labels[:, z_start: z_start + 1, ...], blank_slice], dim=1),
                # )
            # print("pred_sigmoid")
            # print(pred_sigmoid)
            if self.verbose:
                print(f"[{z_start}:{z_end}]: {pair_num=}, {pair_denom=}")
            num += pair_num
            denom += pair_denom
            vol += pair_vol
        dice = 1 - ((num + self.smooth) / (denom + self.smooth))
        return dice.mean() + vol.mean()

    @profile
    def forward(self, pred, labels):
        pred_sigmoid = torch.sigmoid(pred)
        return self.forward_sigmoid(pred_sigmoid, labels, pred)


if __name__ == "__main__":
    import time
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    _device = "cuda"
    # _device = "cpu"
    _batch_size = 2
    # _zs = 3
    # _ys = 3
    # _xs = 3
    _zs = 2
    _ys = 512
    _xs = 512
    _pred = torch.rand((_batch_size, _zs, _ys, _xs)).to(_device).float()
    # _pred = torch.ones((_batch_size, _zs, _ys, _xs)).to(_device).float()
    # _pred = torch.full((_batch_size, _zs, _ys, _xs), 0.00001).to(_device).float()
    _labels = (torch.rand((_batch_size, _zs, _ys, _xs)) > 0.5).to(_device).float()
    # _labels = torch.zeros((_batch_size, _zs, _ys, _xs)).to(_device).float()
    # _labels[0, 1, 1, 1] = True
    # _labels[:] = False
    _criterion = SurfaceDiceLoss(verbose=False, device=_device)

    _t0 = time.time()
    _pred = _pred.requires_grad_(True)
    # _loss = _criterion.forward_sigmoid(_pred, _labels)
    _loss = _criterion(torch.log(_pred / (1 - _pred + 1e-8)), _labels)
    _t1 = time.time()
    _loss.backward()
    _loss = _loss.cpu()
    print((_pred.grad * 1e6).int())
    print(f"loss={_loss}, t={_t1 - _t0}")

    # _loss_label = _criterion.forward_sigmoid(_labels, _labels)
    # _loss_label = _loss_label.cpu()
    # print(f"loss={_loss_label}")

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
