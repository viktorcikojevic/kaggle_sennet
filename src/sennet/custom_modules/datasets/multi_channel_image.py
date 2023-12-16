from sennet.environments.constants import PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from typing import *
import numpy as np
from tqdm import tqdm


# @MMSEG_DATASETS.register_module()
# class MultiChannelDataset(BaseSegDataset):
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
        mask_data = mask.data[::self.z_stride, ::self.xy_stride, ::self.xy_stride]
        c_takes, i_takes, j_takes = np.nonzero(mask_data)
        c_takes = (c_takes * self.z_stride).astype(int)
        i_takes = (i_takes * self.xy_stride).astype(int)
        j_takes = (j_takes * self.xy_stride).astype(int)
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
        c_mins = c_takes - self.n_half_take_channels
        c_maxes = c_mins + self.n_take_channels
        i_mins = i_takes - self.half_crop_size
        i_maxes = i_mins + self.crop_size
        j_mins = j_takes - self.half_crop_size
        j_maxes = j_mins + self.crop_size
        channel_end = mask.shape[0] if self.channel_end is None else self.channel_end
        take_masks = (
                (self.channel_start < c_mins) & (c_maxes < channel_end)
                & (0 < i_mins) & (i_maxes < mask.shape[1])
                & (0 < j_mins) & (j_maxes < mask.shape[2])
        )
        take_indices = np.nonzero(take_masks)[0]
        print(f"taking {len(take_indices)}/{len(take_masks)} ({len(take_indices)/(len(take_masks) + 1e-6)*100:.2f})% for {self.folder}")
        print("generating indices")
        if self.sample_with_mask:
            self.bboxes = np.array([
                [
                    c_mins[i], j_mins[i], i_mins[i],
                    c_maxes[i], j_maxes[i], i_maxes[i],
                ]
                for i in tqdm(take_indices, desc="indexing", total=len(take_indices))
            ])
        else:
            self.bboxes = np.array([
                [
                    c, j, i,
                    c+self.n_take_channels, j+self.crop_size, i+self.crop_size,
                ]
                for c in range(self.n_half_take_channels, mask.shape[0]-self.n_half_take_channels, self.z_stride)
                for i in range(self.half_crop_size, mask.shape[1]-self.half_crop_size, self.xy_stride)
                for j in range(self.half_crop_size, mask.shape[2]-self.half_crop_size, self.xy_stride)
                if (
                    (c+self.n_take_channels < mask.shape[0])
                    and (i+self.crop_size < mask.shape[1])
                    and (j+self.crop_size < mask.shape[2])
                )
            ])
        print("done generating indices")


if __name__ == "__main__":
    _ds = MultiChannelDataset(
        "kidney_1_dense",
        512,
        20,
        substride=0.5,
    )
