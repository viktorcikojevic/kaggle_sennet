from typing import *
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmseg.registry import TRANSFORMS as MMSEG_TRANSFORMS
# from line_profiler_pycharm import profile
import albumentations as alb
from mmengine.utils import is_str
import inspect

import copy


@MMSEG_TRANSFORMS.register_module()
@avoid_cache_randomness
class Albu(BaseTransform):
    def __init__(self, transforms: List[dict]) -> None:
        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = alb.Compose([self.albu_builder(t) for t in self.transforms])

    def albu_builder(self, cfg: dict) -> alb:
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()
        obj_type = args.pop("type")
        if is_str(obj_type):
            if alb is None:
                raise RuntimeError("alb is not installed")
            obj_cls = getattr(alb, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform)
                for transform in args["transforms"]
            ]

        return obj_cls(**args)

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        alb_results = self.aug(
            image=results["img"],
            mask=results["gt_seg_map"],
        )
        results["img"] = alb_results["image"]
        results["gt_seg_map"] = alb_results["mask"]
        if results is None:
            return None
        results["img_shape"] = results["img"].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str
