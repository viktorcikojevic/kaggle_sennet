from src.sennet.environments.constants import PROCESSED_DATA_DIR
from src.sennet.core.mmap_arrays import read_mmap_array
from mmseg.registry import DATASETS as MMSEG_DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from typing import *
import numpy as np
from tqdm import tqdm


@MMSEG_DATASETS.register_module()
class MultiChannelDataset(BaseSegDataset):
    DUMMY_COLOR = (0, 0, 0)
    METAINFO = dict(
        classes=('background', 'blood_vessel'),
        palette=[DUMMY_COLOR, (0, 255, 0)]  # this is due to reduce_zero_label shifting every indices down by one
        # classes=("text",),
        # palette=[[0, 255, 0]]  # this is due to reduce_zero_label shifting every indices down by one
    )

    def __init__(self,
                 folders: List[str],
                 crop_size: int,
                 n_take_channels: int,
                 reduce_zero_label=True,
                 assert_label_exists: bool = False,
                 stride: int = 4,
                 **kwargs) -> None:
        self.folders = [PROCESSED_DATA_DIR / folder for folder in folders]
        self.crop_size = crop_size
        self.half_crop_size = int(self.crop_size / 2)
        self.n_take_channels = n_take_channels
        self.n_half_take_channels = int(self.n_take_channels / 2)
        self.stride = stride
        self.assert_label_exists = assert_label_exists
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".png",
            reduce_zero_label=reduce_zero_label,
            serialize_data=False,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        metadata = []
        for folder in self.folders:
            image_dir = folder / "image"
            mask_dir = folder / "mask"
            label_dir = folder / "label"
            assert image_dir.is_dir(), f"{image_dir=} doesn't exist"
            assert mask_dir.is_dir(), f"{mask_dir=} doesn't exist"
            if self.assert_label_exists:
                assert label_dir.is_dir(), f"{label_dir=} doesn't exist but {self.assert_label_exists=}"
            # not really sure if loading all non_zero indices is the best idea, also with all these channels
            mask = read_mmap_array(mask_dir)
            mask_data = mask.data[:, ::self.stride, ::self.stride]
            c_takes, i_takes, j_takes = np.nonzero(mask_data)
            i_takes *= self.stride
            j_takes *= self.stride
            md = dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            c_mins = c_takes - self.n_half_take_channels
            c_maxes = c_takes + self.n_half_take_channels
            i_mins = i_takes - self.half_crop_size
            i_maxes = i_takes + self.half_crop_size
            j_mins = j_takes - self.half_crop_size
            j_maxes = j_takes + self.half_crop_size
            take_masks = (
                    (0 < c_mins) & (c_maxes < mask.shape[0])
                    & (0 < i_mins) & (i_maxes < mask.shape[1])
                    & (0 < j_mins) & (j_maxes < mask.shape[2])
            )
            take_indices = np.nonzero(take_masks)[0]
            items = [
                dict(
                    bbox=[
                        c_mins[i], c_maxes[i],
                        i_mins[i], i_maxes[i],
                        j_mins[i], j_maxes[i],
                    ],
                    **md,
                )
                for i in tqdm(take_indices, desc="indexing", total=len(take_indices))
            ]
            metadata += items
        return metadata


if __name__ == "__main__":
    _ds = MultiChannelDataset(
        ["kidney_1_dense"],
        512,
        20,
        stride=10,
    )
    _m = _ds.load_data_list()
