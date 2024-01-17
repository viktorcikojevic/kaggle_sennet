from sennet.custom_modules.models.unet3d import model as unet_model
from sennet.custom_modules.models.base_model import Base3DSegmentor, SegmentorOutput
from sennet.custom_modules.models import medical_net_resnet3d as resnet3ds
from sennet.custom_modules.models import layers
from sennet.environments.constants import PRETRAINED_DIR
import segmentation_models_pytorch as smp
from line_profiler_pycharm import profile
from typing import Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


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
        if 'replace_batch_norm_with_layer_norm' in kw:
            replace_batch_norm_with_layer_norm = kw.pop('replace_batch_norm_with_layer_norm')
        else:
            replace_batch_norm_with_layer_norm = False
        constructor = getattr(smp, self.version)
        self.model = constructor(**kw)
        if replace_batch_norm_with_layer_norm:
            self.replace_bn_with_ln(self.model)

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
        
    def replace_bn_with_ln(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # Get the number of channels in the BatchNorm layer
                num_channels = child.num_features
                # Replace with LayerNorm2d
                setattr(module, child_name, layers.LayerNorm2d(num_channels))
            else:
                # Recursively apply to child modules
                self.replace_bn_with_ln(child)
        

class SegformerModel(Base3DSegmentor):
    def __init__(self, **kw):
        Base3DSegmentor.__init__(self)
        self.kw = kw
        self.upsampler = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        segformer_config =  SegformerConfig(**kw)
        self.segformer = SegformerForSemanticSegmentation(segformer_config)

    def get_name(self) -> str:
        # concatenate all depths in self.kw['depths']
        depths = ""
        for d in self.kw['depths']:
            depths += str(d)
        return f"Segformer_{depths}"

    def predict(self, img: torch.Tensor) -> SegmentorOutput:

        x = self.upsampler(img[:, 0, :, :, :])
        model_out = self.segformer(x).logits

        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )




class SMPModelUpsampleBy2(Base3DSegmentor):
    def __init__(self, version: str, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        if 'freeze_bn_layers' in kw:
            freeze_bn_layers = kw.pop('freeze_bn_layers') if kw['freeze_bn_layers'] is not None else False
        else: 
            freeze_bn_layers = False
        self.freeze_bn_layers = freeze_bn_layers 
        self.kw = kw
        self.upsampler = layers.PixelShuffleUpsample(in_channels=1, upscale_factor=2)
        constructor = getattr(smp, self.version)
        self.model = constructor(**kw)
        self.downscale_layer = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        
        if self.freeze_bn_layers:
            self.freeze_bn(self.model)

    def freeze_bn(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                child.eval()
            else:
                self.freeze_bn(child)

    def get_name(self) -> str:
        return f"SMP_{self.version}_{self.kw['encoder_name']}_{self.kw['encoder_weights']}"

    @profile
    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[1] == 1, f"{self.__class__.__name__} works in 1 channel images only (for now), expected to have c=1, got {img.shape=}"
        img_upsampled = self.upsampler(img[:, 0, :, :, :])
        model_out = self.model(img_upsampled)
        model_out = self.downscale_layer(model_out)
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )

class SMPModelUpsampleBy4(Base3DSegmentor):
    def __init__(self, version: str, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        self.kw = kw
        self.upsampler = layers.PixelShuffleUpsample(in_channels=1, upscale_factor=4)
        self.freeze_bn_layers = kw.pop('freeze_bn_layers') if 'freeze_bn_layers' in kw else False
        constructor = getattr(smp, self.version)
        self.model = constructor(**kw)
        self.downscale_layer_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downscale_layer_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if self.freeze_bn_layers:
            self.freezing_parameters = self.get_list_of_bn_parameters()
        else:
            self.freezing_parameters = []

    def get_list_of_bn_parameters(self):
        bn_params = []
        for name, param in self.model.named_parameters():
            if any(part.startswith('bn') for part in name.split('.')):
                bn_params.append(name)
        return bn_params

    def get_name(self) -> str:
        return f"SMP_{self.version}_{self.kw['encoder_name']}_{self.kw['encoder_weights']}"

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[1] == 1, f"{self.__class__.__name__} works in 1 channel images only (for now), expected to have c=1, got {img.shape=}"
        img_upsampled = self.upsampler(img[:, 0, :, :, :])
        model_out = self.model(img_upsampled)
        model_out = self.downscale_layer_1(model_out)
        model_out = self.downscale_layer_2(model_out)
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=img.shape[2],
        )



class SMPModelUpsampleBy2With3DEncoding(Base3DSegmentor):
    def __init__(self, version: str, **kw):
        Base3DSegmentor.__init__(self)
        self.version = version
        self.kw = kw
        self.freeze_2d_model = kw.pop('freeze_2d_model') if 'freeze_2d_model' in kw else False
        pretrained = kw.pop('pretrained') if 'pretrained' in kw else None
        kw['in_channels'] = 1 
        kw['classes'] = 1 
        num_3d_layers = kw.pop('num_3d_layers') if 'num_3d_layers' in kw else 1
        self.model_2d = SMPModelUpsampleBy2(version, **kw)
        
        if pretrained: 
            self.load_pretrained_2d_model(pretrained)
        
        # Define 3D convolutions and batch norm layers
        self.conv3d_layers = nn.ModuleList()
        for _ in range(num_3d_layers):
            conv3d = nn.Conv3d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=3, 
                padding=1,
                stride=1
            )
            # initialize the weights to be small in the beginning
            nn.init.normal_(conv3d.weight, mean=0.0, std=0.0002)
            nn.init.zeros_(conv3d.bias)
            self.conv3d_layers.append(conv3d)


    def get_name(self) -> str:
        return f"SMPModelUpsampleBy2With3DEncoding_{self.version}_{self.kw['encoder_name']}_{self.kw['encoder_weights']}"

    def load_pretrained_2d_model(self, pretrained) -> None:
        
        model_state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
        
        if any(k.startswith("ema_") for k in model_state_dict.keys()):
            model_state_dict = OrderedDict([
                    (k, v)
                    for k, v in model_state_dict.items()
                    if k.startswith("ema_model.")
                ])
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="ema_model.module.")
        else:
            model_state_dict = OrderedDict([
                    (k, v)
                    for k, v in model_state_dict.items()
                    if k.startswith("model.")
                ])
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")
        
        load_status = self.model_2d.load_state_dict(model_state_dict)
        print(load_status)
        
    def forward_3d(self, x):
        for layer in self.conv3d_layers:
            x = F.relu(layer(x)) + x
        return x

    @profile
    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        
        B, _, C, H, W = img.shape
        # reshape to B*C, 1, H, W
        img = img.reshape(B*C, 1, 1, H, W)
        if self.freeze_2d_model:
            with torch.no_grad():
                model_out = self.model_2d.predict(img).pred
        else:
            model_out = self.model_2d.predict(img).pred
        
        # reshape to B, C, H, W
        model_out = model_out.reshape(B, C, H, W).unsqueeze(1)
    
        # apply 3d convolutions
        model_out = self.forward_3d(model_out).squeeze(1)
        
        
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=C,
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
