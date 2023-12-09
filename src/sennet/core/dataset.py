from sennet.custom_modules.datasets.multi_channel_image import MultiChannelDataset
from sennet.custom_modules.datasets.transforms.loading import LoadMultiChannelImageAndAnnotationsFromFile
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch


class ThreeDSegmentationDataset(Dataset):
    def __init__(
            self,
            folders: List[str],
            crop_size: int,
            n_take_channels: int,
            reduce_zero_label=True,
            assert_label_exists: bool = False,
            stride: int = 4,
            channel_start: int = 0,
            channel_end: Optional[int] = None,

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

        if crop_size_range is None:
            crop_size_range = (crop_size, crop_size)
        if output_crop_size is None:
            output_crop_size = crop_size

        self.dataset = MultiChannelDataset(
            folders=folders,
            crop_size=crop_size,
            n_take_channels=n_take_channels,
            reduce_zero_label=reduce_zero_label,
            assert_label_exists=assert_label_exists,
            stride=stride,
            channel_start=channel_start,
            channel_end=channel_end,
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
        self.data_list = self.dataset.load_data_list()

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i: int):
        data = self.data_list[i]
        data = self.loader.transform(data)
        for t in self.transforms:
            data = t.transform(data)

        data["img"] = torch.from_numpy(data["img"]).permute((2, 0, 1))
        if "gt_seg_map" in data:
            data["gt_seg_map"] = torch.from_numpy(data["gt_seg_map"]).permute((2, 0, 1))
        return data


if __name__ == "__main__":
    _ds = ThreeDSegmentationDataset(
        ["kidney_1_dense"],
        512,
        20,
        stride=20,
    )
    _dl = DataLoader(
        _ds,
        batch_size=10,
        shuffle=True,
    )
    _item = _ds[100]
    for _batch in _dl:
        print(":D")
