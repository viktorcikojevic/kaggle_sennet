# https://www.kaggle.com/code/junkoda/fast-surface-dice-computation

import numpy as np
import pandas as pd
import sys

import time
import torch
import torch.nn as nn
from PIL import Image
from line_profiler_pycharm import profile


from sennet.custom_modules.metrics.surface_dice_metric import create_table_neighbour_code_to_surface_area

# PyTorch version dependence on index data type
torch_ver_major = int(torch.__version__.split('.')[0])
dtype_index = torch.int32 if torch_ver_major >= 2 else torch.long

device = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')


def rle_decode(mask_rle: str, shape: tuple) -> np.array:
    """
    Decode rle string
    https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode/script
    https://www.kaggle.com/stainsby/fast-tested-rle

    Args:
      mask_rle: run length (rle) as string
      shape: (height, width) of the mask

    Returns:
      array[uint8], 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def compute_area(y: list, unfold: nn.Unfold, area: torch.Tensor) -> torch.Tensor:
    """
    Args:
      y (list[Tensor]): A pair of consecutive slices of mask
      unfold: nn.Unfold(kernel_size=(2, 2), padding=1)
      area (Tensor): surface area for 256 patterns (256, )

    Returns:
      Surface area of surface in 2x2x2 cube
    """
    # Two layers of segmentation masks
    yy = torch.stack(y, dim=0).to(torch.float16).unsqueeze(0)
    # (batch_size=1, nch=2, H, W) 
    # bit (0/1) but unfold requires float

    # unfold slides through the volume like a convolution
    # 2x2 kernel returns 8 values (2 channels * 2x2)
    cubes_float = unfold(yy).squeeze(0)  # (8, n_cubes)

    # Each of the 8 values are either 0 or 1
    # Convert those 8 bits to one uint8
    cubes_byte = torch.zeros(cubes_float.size(1), dtype=dtype_index, device=device)
    # indices are required to be int32 or long for area[cube_byte] below, not uint8
    # Can be int32 for torch 2.0.0, int32 raise IndexError in torch 1.13.1.
    
    for k in range(8):
        cubes_byte += cubes_float[k, :].to(dtype_index) << k

    # Use area lookup table: pattern index -> area [float]
    cubes_area = area[cubes_byte]

    return cubes_area


def compute_surface_dice_score(submit: pd.DataFrame, label: pd.DataFrame) -> float:
    """
    Compute surface Dice score for one 3D volume

    submit (pd.DataFrame): submission file with id and rle
    label (pd.DataFrame): ground truth id, rle, and also image height, width
    """
    # submit and label must contain exact same id in same order
    assert (submit['id'] == label['id']).all()
    assert len(label) > 0

    # All height, width must be the same
    len(label['height'].unique()) == 1
    len(label['width'].unique()) == 1

    # Surface area lookup table: Tensor[float32] (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    r = label.iloc[0]
    h, w = r['height'], r['width']
    n_slices = len(label)

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0     # numerator of surface Dice
    denom = 0   # denominator
    for i in range(n_slices + 1):
        # Load one slice
        if i < n_slices:
            r = label.iloc[i]
            y1 = rle_decode(r['rle'], (h, w))
            y1 = torch.from_numpy(y1).to(device)

            r = submit.iloc[i]
            y1_pred = rle_decode(r['rle'], (h, w))
            y1_pred = torch.from_numpy(y1_pred).to(device)
        else:
            # Padding after the last slice
            y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

        # Compute the surface area between two slices (n_cubes,)
        area_pred = compute_area([y0_pred, y1_pred], unfold, area)
        area_true = compute_area([y0, y1], unfold, area)

        # True positive cube indices
        idx = torch.logical_and(area_pred > 0, area_true > 0)

        # Surface dice numerator and denominator
        num += area_pred[idx].sum() + area_true[idx].sum()
        denom += area_pred.sum() + area_true.sum()

        # Next slice
        y0 = y1
        y0_pred = y1_pred

    dice = num / denom.clamp(min=1e-8)
    return dice.item()


def compute_surface_dice_score_from_thresholded_mmap(
        thresholded_chunks: list[np.ndarray],
        label: np.ndarray,
) -> float:
    """
    Compute surface Dice score for one 3D volume

    note: this is basically a shameless copy of the function below, I'm just too lazy to make it right
    """
    # submit and label must contain exact same id in same order
    total_chunk_channels = sum(p.shape[0] for p in thresholded_chunks)
    assert total_chunk_channels == label.shape[0], f"{total_chunk_channels=} != {label.shape[0]=}"

    # Surface area lookup table: Tensor[float32] (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    h = label.shape[1]
    w = label.shape[2]
    n_slices = label.shape[0]

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0  # numerator of surface Dice
    denom = 0  # denominator
    i = 0

    # y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)
    for chunk in thresholded_chunks:
        for c in range(chunk.shape[0]):
            if i < n_slices:
                y1 = torch.from_numpy(label[i, :, :].copy()).to(device)
                y1_pred = torch.from_numpy(chunk[c, :, :]).to(device)
            else:
                y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

            # Compute the surface area between two slices (n_cubes,)
            area_pred = compute_area([y0_pred, y1_pred], unfold, area)
            area_true = compute_area([y0, y1], unfold, area)

            # True positive cube indices
            idx = torch.logical_and(area_pred > 0, area_true > 0)

            # Surface dice numerator and denominator
            num += area_pred[idx].sum() + area_true[idx].sum()
            denom += area_pred.sum() + area_true.sum()

            # Next slice
            y0 = y1
            y0_pred = y1_pred
            i += 1
    dice = num / denom.clamp(min=1e-8)
    return dice.item()


@profile
def compute_surface_dice_score_from_mmap(
        mean_prob_chunks: list[np.ndarray],
        label: np.ndarray,
        threshold: float,
) -> float:
    """
    Compute surface Dice score for one 3D volume

    submit (pd.DataFrame): submission file with id and rle
    label (pd.DataFrame): ground truth id, rle, and also image height, width
    """
    # submit and label must contain exact same id in same order
    assert sum(p.shape[0] for p in mean_prob_chunks) == label.shape[0]

    # Surface area lookup table: Tensor[float32] (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    h = label.shape[1]
    w = label.shape[2]
    n_slices = label.shape[0]

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0  # numerator of surface Dice
    denom = 0  # denominator
    i = 0

    # y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)
    for chunk in mean_prob_chunks:
        for c in range(chunk.shape[0]):
            if i < n_slices:
                y1 = torch.from_numpy(label[i, :, :].copy()).to(device)
                y1_pred = torch.from_numpy(chunk[c, :, :] > threshold).to(device)
            else:
                y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

            # Compute the surface area between two slices (n_cubes,)
            area_pred = compute_area([y0_pred, y1_pred], unfold, area)
            area_true = compute_area([y0, y1], unfold, area)

            # True positive cube indices
            idx = torch.logical_and(area_pred > 0, area_true > 0)

            # Surface dice numerator and denominator
            num += area_pred[idx].sum() + area_true[idx].sum()
            denom += area_pred.sum() + area_true.sum()

            # Next slice
            y0 = y1
            y0_pred = y1_pred
            i += 1
    dice = num / denom.clamp(min=1e-8)
    return dice.item()


if __name__ == "__main__":
    from pathlib import Path
    from sennet.core.mmap_arrays import read_mmap_array
    import time

    _root_path = Path("/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_dense/")
    _mmap_paths = sorted([p for p in _root_path.glob("chunk_*")])
    _mmaps = [read_mmap_array(p / "mean_prob") for p in _mmap_paths]
    _label = read_mmap_array("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_3_dense/label", mode="r")
    print([m.shape for m in _mmaps])

    _t0 = time.time()
    _dice = compute_surface_dice_score_from_mmap(
        [m.data for m in _mmaps],
        _label.data,
        0.2,
    )
    _t1 = time.time()
    print(f"{_t1 - _t0}, {_dice}")

    _t0 = time.time()
    _dice = compute_surface_dice_score_from_mmap(
        [m.data for m in _mmaps],
        _label.data,
        0.8,
    )
    _t1 = time.time()
    print(f"{_t1 - _t0}, {_dice}")
