import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
import albumentations as alb


class ForegroundSegmentationDataset(Dataset):
    def __init__(
            self,
            img_dir: Union[str, Path],
            label_dir: Optional[Union[str, Path]] = None,
            stride: int = 4,
            aug: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.img_paths = {p.stem.split("_")[0]: p for p in self.img_dir.glob("*")}
        self.stride = stride
        self.aug = aug

        if self.label_dir is None:
            self.label_paths = None
            self.keys = sorted(list(self.img_paths.keys()))
        else:
            self.label_paths = {p.stem.split("_")[0]: p for p in self.label_dir.glob("*")}
            self.keys = sorted(list(set(self.img_paths.keys()).intersection(set(self.label_paths.keys()))))

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.aug:
            self.transform = alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.MedianBlur(blur_limit=3, p=0.1),
                alb.RandomBrightnessContrast(),
                # alb.RandomResizedCrop(384, 384, scale=(0.5, 1.0), p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        img_path = self.img_paths[k]
        img = cv2.imread(str(img_path), 0)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = cv2.resize(img, dsize=None, fx=1/self.stride, fy=1/self.stride, interpolation=cv2.INTER_AREA)

        if self.label_dir is not None:
            label_path = self.label_paths[k]
            seg = cv2.imread(str(label_path), 0) > 0
        else:
            seg = None

        if self.aug:
            img = cv2.resize(img, (seg.shape[1], seg.shape[0]))
            out = self.transform(image=img, mask=seg)
            img = out["image"]
            seg = out["mask"]

        # img = self.clahe.apply(img)
        img = img.astype(np.float32)
        img -= 127.0
        img /= 60.0
        data = {
            "img": torch.from_numpy(img).unsqueeze(0),
            "img_path": str(img_path),
            "img_w": img_w,
            "img_h": img_h,
        }

        if self.label_dir is not None:
            data["gt_seg_map"] = torch.from_numpy(seg).unsqueeze(0)
        return data


if __name__ == "__main__":
    _ds = ForegroundSegmentationDataset(
        "/home/clay/research/kaggle/sennet/data/blood-vessel-segmentation/train/kidney_1_dense/images",
        "/home/clay/research/kaggle/sennet/data/labeled_masks/kidney_1_dense",
    )
    _data = _ds[10]
    _dl = DataLoader(
        _ds,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )
    for _batch in tqdm(_dl, total=len(_dl)):
        pass
