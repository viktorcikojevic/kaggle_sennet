import albumentations as A
import cv2
import numpy as np


class VanillaAugmentation:

    def __init__(
            self,
    ):
        self._transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.25),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.25
            ),
        ])

    def transform(self, data):
        img = data["img"]
        gt_seg_map = data["gt_seg_map"]

        # Get them to (h, w, c)
        img = np.transpose(img, (1, 2, 0))
        gt_seg_map = np.transpose(gt_seg_map, (1, 2, 0))

        out = self._transform(image=img, mask=gt_seg_map)
        img_augmented = out["image"]
        gt_seg_map_augmented = out["mask"]

        # Get images back to (c, h, w)
        data["img"] = np.transpose(img_augmented, (2, 0, 1))
        data["gt_seg_map"] = np.transpose(gt_seg_map_augmented, (2, 0, 1))

        return data
