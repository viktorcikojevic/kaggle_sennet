from sennet.custom_modules.models.unet3d.model import UNet3D, UNet2D
from sennet.custom_modules.models.base_model import Base3DSegmentor, SegmentorOutput
from sennet.custom_modules.models import medical_net_resnet3d as resnet3ds
from sennet.environments.constants import PRETRAINED_DIR
from typing import Union, Optional, Tuple
from pathlib import Path
import torch


def resize_3d_image(img: torch.Tensor, new_whd: Tuple[int, int, int]):
    out_w, out_h, out_d = new_whd
    batch_size, _c, in_d, in_h, in_w = img.shape
    mesh_z, mesh_y, mesh_x = torch.meshgrid([
        torch.linspace(-1.0, 1.0, out_d),
        torch.linspace(-1.0, 1.0, out_h),
        torch.linspace(-1.0, 1.0, out_w),
    ])
    grid = torch.stack((mesh_x, mesh_y, mesh_z), 3).tile((batch_size, 1, 1, 1, 1)).to(img.device)
    out = torch.nn.functional.grid_sample(
        img,
        grid,
        mode="bilinear",
        align_corners=True
    )
    return out


class WrappedUNet3D(Base3DSegmentor):
    def __init__(self, pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.model = UNet3D(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            load_res = self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"{self.__class__.__name__}: {load_res = }")

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        model_out = self.model(img)
        return SegmentorOutput(
            pred=model_out[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )


class WrappedUNet2D(Base3DSegmentor):
    def __init__(self, pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.model = UNet2D(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            load_res = self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"{self.__class__.__name__}: {load_res = }")

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[2] == 1, f"{self.__class__.__name__} works in 2D images only, expected to have z=1, got {img.shape=}"
        model_out = self.model(img[:, :, 0, :, :])   # (b, c, h, w)
        return SegmentorOutput(
            pred=model_out.reshape((model_out.shape[0], model_out.shape[1], 1, model_out.shape[2], model_out.shape[3]))[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=1,
        )


class WrappedMedicalNetResnet3D(Base3DSegmentor):
    def __init__(self, version: str = "resnet18", pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        constructor = getattr(resnet3ds, self.version)
        self.model = constructor(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ckpt["state_dict"], prefix="module.")
            load_res = self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"{self.__class__.__name__}: {load_res = }")

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        # note: resnets give downsampled results: we need to sample them back up
        batch_size, _c, in_d, in_h, in_w = img.shape
        down_sampled_model_out = self.model(img)
        model_out = resize_3d_image(down_sampled_model_out, (in_w, in_h, in_d))
        return SegmentorOutput(
            pred=model_out[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )


if __name__ == "__main__":
    _device = "cuda"
    _model = WrappedMedicalNetResnet3D(
        "resnet18",
        "medical_nets/resnet_18.pth",
        num_seg_classes=1,
        shortcut_type="A",
    ).to(_device)
    _data = torch.randn((2, 1, 16, 512, 512)).to(_device)
    _out = _model.predict(_data)
    print(":D")
