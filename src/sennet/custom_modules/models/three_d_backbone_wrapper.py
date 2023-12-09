# from mmengine.model.base_module import BaseModule
# from mmseg.models.builder import BACKBONES as MMSEG_BACKBONES
# from mmdet.registry import MODELS as MMDET_BACKBONES


# @MMSEG_BACKBONES.register_module()
# @MMDET_BACKBONES.register_module()
class ThreeDBackboneWrapper(BaseModule):
    def __init__(
            self,
            backbone,
            in_channels: int = 3,
            num_feature_maps: int = 4,
            registry: str = "mmseg"
    ):
        super().__init__()
        if registry == "mmseg":
            self.backbone = MMSEG_BACKBONES.build(backbone)
        elif registry == "mmdet":
            self.backbone = MMDET_BACKBONES.build(backbone)
        else:
            raise NotImplementedError(registry)
        self.in_channels = in_channels
        self.num_feature_maps = num_feature_maps

    def forward(self, x):
        num_chans = x.shape[1]
        features = [[] for _ in range(self.num_feature_maps)]

        for i in range(0, num_chans-self.in_channels, 1):
            outs = self.backbone(x[:, i:i+self.in_channels, :, :])
            assert len(outs) == self.num_feature_maps
            for j, o in enumerate(outs):
                features[j].append(o)

        return tuple([sum(fis) / len(fis) for fis in features])
