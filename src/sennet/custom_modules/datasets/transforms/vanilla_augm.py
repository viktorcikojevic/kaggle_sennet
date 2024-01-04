import albumentations as A
import cv2
import numpy as np


class VanillaAugmentation:

    def __init__(
            self,
            p: float = 0.25
    ):
        self._transform = A.Compose([
            A.Rotate(limit=(-180, 180), p=p),
            A.VerticalFlip(p=p),
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p,
            ),
            # A.RandomGamma(p=p),
            # A.OneOf([
                # A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                # A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1.0),
            # ], p=0.5),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=p),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    interpolation=cv2.INTER_AREA,
                    # border_mode=cv2.BORDER_CONSTANT,
                    # value=0,
                    # mask_value=0,
                    p=0.25
                ),
            ])
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
