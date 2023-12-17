import numpy as np

from sennet.custom_modules.datasets.multi_channel_image import MultiChannelDataset
from sennet.custom_modules.datasets.transforms.loading import LoadMultiChannelImageAndAnnotationsFromFile
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from line_profiler_pycharm import profile


class ThreeDSegmentationDataset(Dataset):
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

            crop_size_range: Optional[Tuple[int, int]] = None,
            output_crop_size: Optional[int] = None,
            to_float32: bool = False,
            channels_jitter: int = 0,
            p_channel_jitter: float = 0.0,
            load_ann: bool = True,
            seg_fill_val: int = 255,
            crop_location_noise: int = 0,

            transforms: Optional[List] = None,
    ):
        Dataset.__init__(self)

        if output_crop_size is None:
            output_crop_size = crop_size

        self.dataset = MultiChannelDataset(
            folder=folder,
            crop_size=crop_size,
            n_take_channels=n_take_channels,
            reduce_zero_label=reduce_zero_label,
            assert_label_exists=assert_label_exists,
            substride=substride,
            channel_start=channel_start,
            channel_end=channel_end,
            sample_with_mask=sample_with_mask,
        )
        self.loader = LoadMultiChannelImageAndAnnotationsFromFile(
            crop_size_range=crop_size_range,
            output_crop_size=output_crop_size,
            to_float32=to_float32,
            channels_jitter=channels_jitter,
            p_channel_jitter=p_channel_jitter,
            load_ann=load_ann,
            seg_fill_val=seg_fill_val,
            crop_location_noise=crop_location_noise,
        )

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = []

    def __len__(self):
        return len(self.dataset)

    @profile
    def __getitem__(self, i: int):
        data = self.dataset[i]
        data = self.loader.transform(data)
        for t in self.transforms:
            data = t.transform(data)

        # unsqueeze(0) are there to create a channel dimension
        if "gt_seg_map" in data:
            data["gt_seg_map"] = torch.from_numpy(data["gt_seg_map"]).unsqueeze(0)

        data["img"] = torch.from_numpy(data["img"].astype(np.float32))
        data["img"] -= 127.0
        data["img"] /= 60.0
        data["img"] = data["img"].unsqueeze(0)
        data["bbox"] = np.array(data["bbox"])
        return data
        # return data["img"]


if __name__ == "__main__":
    _ds = ThreeDSegmentationDataset(
        # ["kidney_1_dense", "kidney_3_sparse"],
        "kidney_1_dense",
        64,
        64,
        output_crop_size=64,
        substride=1.0,
        load_ann=True,
        sample_with_mask=True,
    )
    print(f"{len(_ds) = }")
    _dl = DataLoader(
        _ds,
        num_workers=0,
        batch_size=10,
        shuffle=True,
        drop_last=True,
    )
    _item = _ds[100]
    for _batch in tqdm(_dl, total=len(_dl)):
        pass
    # for _ in range(10):
    #     for _i in tqdm(_ds, total=len(_ds)):
    #         pass

    # _batch = next(iter(_dl))
    print(":D")
