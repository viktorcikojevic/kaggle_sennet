import math
from functools import partial
# from mmseg.models.builder import BACKBONES
# from mmengine.model.base_module import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class ResNet(BaseModule):
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 flattening_strategy="mean"):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.flattening_strategy = flattening_strategy
        assert self.flattening_strategy in ("mean", "reshape")

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _3d_to_2d(self, x):
        if self.flattening_strategy == "mean":
            return torch.mean(x, dim=2)
        elif self.flattening_strategy == "reshape":
            return x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        else:
            raise NotImplementedError(self.flattening_strategy)

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        outs = []
        x = self.layer1(x)
        outs.append(self._3d_to_2d(x))
        x = self.layer2(x)
        outs.append(self._3d_to_2d(x))
        x = self.layer3(x)
        outs.append(self._3d_to_2d(x))
        x = self.layer4(x)
        outs.append(self._3d_to_2d(x))

        return outs
        #
        # x = self.avgpool(x)
        #
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        # return x


# @BACKBONES.register_module()
class Resnet3D10(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [1, 1, 1, 1],
            get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D18(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [2, 2, 2, 2],
            get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D34(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            block_inplanes=get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D50(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [3, 4, 6, 3],
            get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D101(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [3, 4, 23, 3],
            get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D152(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [3, 8, 36, 3],
            get_inplanes(),
            **kwargs,
        )


# @BACKBONES.register_module()
class Resnet3D200(ResNet):
    def __init__(self, **kwargs):
        ResNet.__init__(
            self,
            BasicBlock,
            [3, 24, 36, 3],
            get_inplanes(),
            **kwargs,
        )
