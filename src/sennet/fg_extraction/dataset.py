import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm


class ForegroundSegmentationDataset(Dataset):
    def __init__(
            self,
            img_dir: Union[str, Path],
            label_dir: Optional[Union[str, Path]] = None
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.img_paths = {p.stem: p for p in self.img_dir.glob("*")}
        if self.label_dir is None:
            self.keys = sorted(list(self.img_paths.keys()))
            self.label_paths = None
        else:
            self.keys = sorted(list(set(self.img_paths.keys()).intersection(set(self.label_paths.keys()))))
            self.label_paths = {p.stem: p for p in self.label_dir.glob("*")}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        img_path = self.img_paths[k]
        img = cv2.imread(str(img_path), 0).astype(np.float32)
        img -= 127.0
        img /= 60.0
        data = {
            "img": torch.from_numpy(img).unsqueeze(0),
            "img_path": str(img_path),
        }
        if self.label_dir is not None:
            label_path = self.label_paths[k]
            seg = cv2.imread(str(label_path), 0) > 0
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
