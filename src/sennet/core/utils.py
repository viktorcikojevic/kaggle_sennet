import torch
from typing import Tuple


DEPTH_ALONG_CHANNEL = 0
DEPTH_ALONG_HEIGHT = 1
DEPTH_ALONG_WIDTH = 2


def resize_3d_image(img: torch.Tensor, new_whd: Tuple[int, int, int]):
    out_w, out_h, out_d = new_whd
    batch_size, _c, in_d, in_h, in_w = img.shape
    mesh_z, mesh_y, mesh_x = torch.meshgrid([
        torch.linspace(-1.0, 1.0, out_d),
        torch.linspace(-1.0, 1.0, out_h),
        torch.linspace(-1.0, 1.0, out_w),
    ], indexing="ij")
    grid = torch.stack((mesh_x, mesh_y, mesh_z), 3).tile((batch_size, 1, 1, 1, 1)).to(img.device)
    out = torch.nn.functional.grid_sample(
        img,
        grid,
        mode="bilinear",
        align_corners=True
    )
    return out
