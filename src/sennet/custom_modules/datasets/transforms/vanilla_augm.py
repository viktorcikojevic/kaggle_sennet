import albumentations as A
import cv2
import numpy as np


class VanillaAugmentation:

    def __init__(
            self,
            p: float = 0.25
    ):
        self._transform = A.Compose([
            # A.Rotate(limit=(-180, 180), p=p),
            A.RandomRotate90(p=p),
            A.VerticalFlip(p=p),
            A.HorizontalFlip(p=p),
            # A.RandomGamma(p=p),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.05,
                p=p,
            ),
            # A.OneOf([
            #     A.GaussNoise(var_limit=(0.005, 0.005), always_apply=True),
            #     A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, always_apply=True),
            # ], p=p),
            A.OneOf([
                A.GaussianBlur(always_apply=True),
                A.MotionBlur(always_apply=True),
            ], p=p),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=p, always_apply=True),
                # A.ElasticTransform(
                #     alpha=1,
                #     sigma=50,
                #     alpha_affine=50,
                #     interpolation=cv2.INTER_AREA,
                #     always_apply=True,
                # ),
            ], p=p)
        ])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._transform})"

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


if __name__ == "__main__":
    _a = VanillaAugmentation()
    print(_a)
