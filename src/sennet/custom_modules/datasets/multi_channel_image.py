from sennet.environments.constants import PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from typing import *
import numpy as np


def generate_crop_bboxes(
        crop_size: int,
        n_take_channels: int,
        xy_stride: int,
        z_stride: int,
        shape: Tuple[int, int, int],
        mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    half_crop_size = int(crop_size / 2)
    n_half_take_channels = int(n_take_channels / 2)
    bboxes = []
    for c in range(n_half_take_channels, shape[0] - n_half_take_channels, z_stride):
        if c + n_take_channels >= shape[0]:
            continue
        for i in range(half_crop_size, shape[1] - half_crop_size, xy_stride):
            if i + crop_size >= shape[1]:
                continue
            for j in range(half_crop_size, shape[2] - half_crop_size, xy_stride):
                if j + crop_size >= shape[2]:
                    continue
                if (mask is not None) and (not np.any(mask[c:c+n_take_channels, i:i+crop_size, j:j+crop_size])):
                    continue
                bboxes.append([
                    c, j, i,
                    c + n_take_channels, j + crop_size, i + crop_size,
                ])
    return np.array(bboxes)


class MultiChannelDataset:
    DUMMY_COLOR = (0, 0, 0)
    METAINFO = dict(
        classes=('background', 'blood_vessel'),
        palette=[DUMMY_COLOR, (0, 255, 0)]  # this is due to reduce_zero_label shifting every indices down by one
        # classes=("text",),
        # palette=[[0, 255, 0]]  # this is due to reduce_zero_label shifting every indices down by one
    )

    def __init__(
            self,
            folder: str,
            crop_size: int,
            n_take_channels: int,
            reduce_zero_label=True,
            assert_label_exists: bool = False,
            substride: float = 0.25,
            channel_start: int = 0,
            channel_end: Optional[int] = None,
            sample_with_mask: bool = False,
    ) -> None:
        self.folder = PROCESSED_DATA_DIR / folder
        print(f"reading from the following folder: {self.folder}")
        self.crop_size = crop_size
        self.half_crop_size = int(self.crop_size / 2)
        self.n_take_channels = n_take_channels
        self.n_half_take_channels = int(self.n_take_channels / 2)
        self.substride = substride
        self.xy_stride = max(1, int(self.substride * crop_size))
        self.z_stride = max(1, int(self.substride * n_take_channels))
        self.assert_label_exists = assert_label_exists
        self.channel_start = channel_start
        self.channel_end = channel_end
        self.reduce_zero_label = reduce_zero_label
        self.sample_with_mask = sample_with_mask
        self.image_paths = {}
        self._load_data_list()

    def __len__(self):
        return self.bboxes.shape[0]

    def __getitem__(self, i):
        return dict(
            bbox=self.bboxes[i],
            **self.general_metadata
        )

    def _load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        image_dir = self.folder / "image"
        mask_dir = self.folder / "mask"
        label_dir = self.folder / "label"
        image_paths = [Path(p) for p in Path(self.folder / "image_paths").read_text().split("\n")]
        assert image_dir.is_dir(), f"{image_dir=} doesn't exist"
        assert mask_dir.is_dir(), f"{mask_dir=} doesn't exist"
        if self.assert_label_exists:
            assert label_dir.is_dir(), f"{label_dir=} doesn't exist but {self.assert_label_exists=}"
        # not really sure if loading all non_zero indices is the best idea, also with all these channels
        mask = read_mmap_array(mask_dir, mode="r")
        self.general_metadata = dict(
            folder=str(self.folder.absolute().resolve()),
            image_dir=str(image_dir.absolute().resolve()),
            seg_dir=str(label_dir.absolute().resolve()),
            img_h=mask.shape[1],
            img_w=mask.shape[2],
            img_c=mask.shape[0],
            reduce_zero_label=self.reduce_zero_label,
            seg_fields=[],
        )
        self.image_paths = [f"{p.parent.parent.stem}_{p.stem}" for p in image_paths]
        print("generating indices")
        self.bboxes = generate_crop_bboxes(
            crop_size=self.crop_size,
            n_take_channels=self.n_take_channels,
            xy_stride=self.xy_stride,
            z_stride=self.z_stride,
            shape=(mask.shape[0], mask.shape[1], mask.shape[2]),
            mask=mask.data if self.sample_with_mask else None
        )
        print(f"{self.folder}: bboxes={len(self.bboxes)}")
        print("done generating indices")


if __name__ == "__main__":
    _ds = MultiChannelDataset(
        "kidney_1_dense",
        512,
        20,
        substride=0.5,
    )
