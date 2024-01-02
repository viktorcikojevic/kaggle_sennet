import numpy as np
from sennet.custom_modules.datasets.transforms.normalisation import Normalise
from sennet.custom_modules.datasets.multi_channel_image import MultiChannelDataset
from sennet.custom_modules.datasets.transforms.loading import LoadMultiChannelImageAndAnnotationsFromFile
from sennet.custom_modules.datasets import transforms as augmentations
from typing import List, Optional, Tuple, Dict, Any
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
            n_appereant_channels: int,
            reduce_zero_label=True,
            assert_label_exists: bool = False,
            substride: float = 0.25,
            channel_start: int = 0,
            channel_end: Optional[int] = None,
            sample_with_mask: bool = False,
            add_depth_along_channel: bool = True,
            add_depth_along_width: bool = False,
            add_depth_along_height: bool = False,

            crop_size_range: Optional[Tuple[int, int]] = None,
            output_crop_size: Optional[int] = None,
            to_float32: bool = False,
            load_ann: bool = True,
            seg_fill_val: int = 255,
            crop_location_noise: int = 0,
            p_crop_location_noise: float = 0.0,
            p_crop_size_noise: float = 0.0,
            p_crop_size_keep_ar: float = 0.0,

            augmenter_class: Optional[str] = None,
            augmenter_kwargs: Optional[Dict[str, Any]] = None,

            transforms: Optional[List] = None,
            normalisation_kwargs: Optional[Dict] = None,
    ):
        Dataset.__init__(self)

        if output_crop_size is None:
            output_crop_size = crop_size

        self.n_take_channels = n_take_channels
        self.n_appereant_channels = n_appereant_channels

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
            add_depth_along_channel=add_depth_along_channel,
            add_depth_along_width=add_depth_along_width,
            add_depth_along_height=add_depth_along_height,
        )
        self.loader = LoadMultiChannelImageAndAnnotationsFromFile(
            crop_size_range=crop_size_range,
            output_crop_size=output_crop_size,
            to_float32=to_float32,
            load_ann=load_ann,
            seg_fill_val=seg_fill_val,
            crop_location_noise=crop_location_noise,
            p_crop_location_noise=p_crop_location_noise,
            p_crop_size_noise=p_crop_size_noise,
            p_crop_size_keep_ar=p_crop_size_keep_ar,
        )

        self.augmenter_class = augmenter_class
        if self.augmenter_class is not None:
            augmenter_constructor = getattr(augmentations, self.augmenter_class)
            self.augmenter = augmenter_constructor(**augmenter_kwargs)
        else:
            self.augmenter = None

        self.transforms = transforms
        self.normalisation_kwargs = normalisation_kwargs
        if self.transforms is None:
            self.transforms = []
        if self.augmenter is not None:
            self.transforms.append(self.augmenter)
        if self.normalisation_kwargs is not None:
            self.transforms.append(Normalise(**self.normalisation_kwargs))

    def __len__(self):
        return len(self.dataset)

    @profile
    def __getitem__(self, i: int):
        data = self.dataset[i]
        data = self.loader.transform(data)
        for t in self.transforms:
            try:
                data = t.transform(data)
            except Exception as e:
                print(f"can't run {t}: {repr(e)}")
                raise
        
        if self.n_appereant_channels < self.n_take_channels:
            # take a subset of the channels
            channel_start = np.random.randint(0, self.n_take_channels - self.n_appereant_channels)
            channel_end = channel_start + self.n_appereant_channels
            take_channels = np.arange(channel_start, channel_end)
            data["img"] = data["img"][take_channels]
            if "gt_seg_map" in data:
                data["gt_seg_map"] = data["gt_seg_map"][take_channels]
        
        # unsqueeze(0) are there to create a channel dimension
        if "gt_seg_map" in data:
            data["gt_seg_map"] = torch.from_numpy(data["gt_seg_map"]).unsqueeze(0)

        data["img"] = torch.from_numpy(data["img"].astype(np.float32))
        data["img"] = data["img"].unsqueeze(0)
        data["bbox"] = np.array(data["bbox"])
        return data
        # return data["img"]


if __name__ == "__main__":
    _ds = ThreeDSegmentationDataset(
        "kidney_1_dense",
        # "kidney_3_dense",
        512,
        12,
        crop_location_noise=100,
        crop_size_range=(400, 600),
        output_crop_size=512,
        substride=1.0,
        load_ann=True,
        sample_with_mask=True,
        add_depth_along_channel=False,
        add_depth_along_height=False,
        add_depth_along_width=True,
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
