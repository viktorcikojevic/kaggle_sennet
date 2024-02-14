import albumentations as A
import cv2
import numpy as np


class IgorAugmentation:

    def __init__(
            self,
    ):
        self._transform = A.Compose([
              A.ShiftScaleRotate(scale_limit=0.2),
              A.HorizontalFlip(p=0.5),
              A.VerticalFlip(p=0.5),
              A.RandomRotate90(p=0.5),
              A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.RandomBrightness(),
                    A.RandomGamma(),
              ], p=1.0,),
        ], p=1.0,)

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
    _a = IgorAugmentation()
    print(_a)
