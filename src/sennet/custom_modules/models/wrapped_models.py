from sennet.custom_modules.models.unet3d import model as unet_model
from sennet.custom_modules.models.base_model import Base3DSegmentor, SegmentorOutput
from sennet.custom_modules.models import medical_net_resnet3d as resnet3ds
from sennet.environments.constants import PRETRAINED_DIR
import segmentation_models_pytorch as smp
from line_profiler_pycharm import profile
from typing import Union, Optional
from pathlib import Path
import torch


class WrappedUNet3D(Base3DSegmentor):
    def __init__(self, version: str = "UNet3D", pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        constructor = getattr(unet_model, self.version)
        self.model = constructor(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            load_res = self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"{self.__class__.__name__}: {load_res = }")

    def get_name(self):
        return self.version

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        model_out = self.model(img)
        return SegmentorOutput(
            pred=model_out[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )


class WrappedUNet2D(Base3DSegmentor):
    """
    this outputs only the first channel's seg only
    """
    def __init__(self, pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.model = unet_model.UNet2D(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            load_res = self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"{self.__class__.__name__}: {load_res = }")

    def get_name(self):
        return f"{self.version}(2d)"

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[2] == 1, f"{self.__class__.__name__} works in 2D images only, expected to have z=1, got {img.shape=}"
        model_out = self.model(img[:, :, 0, :, :])   # (b, c, h, w)
        return SegmentorOutput(
            pred=model_out.reshape((model_out.shape[0], model_out.shape[1], 1, model_out.shape[2], model_out.shape[3]))[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=1,
        )


class WrappedUNet2P5D(Base3DSegmentor):
    def __init__(self, pretrained: Optional[Union[str, Path]] = None, **kw):
        Base3DSegmentor.__init__(self)
        self.model = unet_model.UNet2D(**kw)
        self.pretrained = pretrained
        if self.pretrained is not None:
            ckpt = torch.load(PRETRAINED_DIR / self.pretrained)
            load_res = self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"{self.__class__.__name__}: {load_res = }")

    def get_name(self):
        return f"{self.version}(2.5d)"

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[1] == 1, f"{self.__class__.__name__} works in 1 channel images only (for not), expected to have c=1, got {img.shape=}"
        model_out = self.model(img.reshape((img.shape[0], img.shape[1]*img.shape[2], img.shape[3], img.shape[4])))   # (b, c, h, w)
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=img.shape[2],
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

    def get_name(self):
        return self.version

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        # note: resnets give downsampled results: we need to sample them back up
        model_out = self.model(img)
        return SegmentorOutput(
            pred=model_out[:, 0, :, :, :],
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )


class SMPModel(Base3DSegmentor):
    def __init__(self, version: str, replace_batch_norm_with_layer_norm: bool = False, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        self.kw = kw
        self.replace_batch_norm_with_layer_norm = replace_batch_norm_with_layer_norm
        constructor = getattr(smp, self.version)
        self.model = constructor(**kw)
        if self.replace_batch_norm_with_layer_norm:
            self.replace_bn_with_ln(self.model)

    def __repr__(self):
        return str(self.model)

    def replace_bn_with_ln(self, module: torch.nn.Module):
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                setattr(module, child_name, torch.nn.GroupNorm(1, child.num_features))
                print(f"replaced: {child_name} with GroupNorm(1, {child.num_features})")
            else:
                self.replace_bn_with_ln(child)

    def get_name(self) -> str:
        return f"SMP_{self.version}_{self.kw['encoder_name']}_{self.kw['encoder_weights']}"

    @profile
    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[1] == 1, f"{self.__class__.__name__} works in 1 channel images only (for now), expected to have c=1, got {img.shape=}"
        # model_out = self.model(img[:, 0, :, :, :])
        model_out = self.model(img.squeeze(1))
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )


if __name__ == "__main__":
    _device = "cuda"
    # _model = WrappedMedicalNetResnet3D(
    #     "resnet18",
    #     "medical_nets/resnet_18.pth",
    #     num_seg_classes=1,
    #     shortcut_type="A",
    # ).to(_device)
    # _model = WrappedUNet3D(
    #     "UNet3D",
    #     None,
    #     in_channels=1,
    #     out_channels=1,
    #     num_groups=8,
    #     f_maps=32,
    #     final_sigmoid=False,
    #     is_segmentation=False,
    #     is3d=True,
    # ).to(_device)
    # _model = WrappedMedicalNetResnet3D(
    #     "resnet200",
    #     None,
    #     num_seg_classes=1,
    # ).to(_device)
    # smp.UnetPlusPlus
    _c = 3
    _model = SMPModel(
        "Unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        replace_batch_norm_with_layer_norm=True,
        in_channels=_c,
        classes=_c,
    ).to(_device).train()
    _data = torch.randn((2, 1, _c, 512, 512)).to(_device)
    _out = _model.predict(_data)
    print(_model)
    print(f"{_out.pred.shape=}, {_out.pred.max()=}, {_out.pred.min()=}")
    print(":D")
