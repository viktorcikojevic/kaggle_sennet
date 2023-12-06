from mmseg.registry import DATASETS as MMSEG_DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from vesuvius.core.images import get_image_to_patches_bbox
from typing import *
from pathlib import Path
import numpy as np
import json
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
                 reduce_zero_label=True,
                 apparent_length: int = None,
                 allow_missing_files: bool = False,
                 **kwargs) -> None:
        self.apparent_length = apparent_length
        self.allow_missing_files = allow_missing_files
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
        metadata_path = Path(self.data_root) / "metadata.json"
        assert metadata_path.exists(), \
            f"{self.__class__.__name__} expects the metadata file at {metadata_path} to exist"
        metadata = json.loads(metadata_path.read_text())
        for md in metadata:
            assert "full_width" in md
            assert "full_height" in md
            assert "num_channels" in md
            assert "available_scales" in md
            for s, info in md["available_scales"].items():
                assert "width" in info
                assert "height" in info
                for k, v in info.items():
                    if k.endswith("_path") or k.endswith("_root"):
                        resolved_path = (Path(self.data_root) / v).resolve().absolute()
                        info[k] = str(resolved_path)
                        if self.allow_missing_files and not resolved_path.exists():
                            print(f"warning: [{s}][{k}]: {resolved_path}, doesn't exist")
                        else:
                            assert resolved_path.exists(), f"[{s}][{k}]: {resolved_path}, doesn't exist"
            md["label_map"] = self.label_map
            md["reduce_zero_label"] = self.reduce_zero_label
            md["seg_fields"] = []
        num_repeats = None
        if self.apparent_length is not None:
            num_repeats = int(np.ceil(self.apparent_length / len(metadata)))
            if num_repeats > 1:
                metadata = metadata * num_repeats
        print(f"{num_repeats = }, {len(metadata) = }")
        return metadata


def stride_mask_indices(mask_indices: np.ndarray, stride: int) -> np.ndarray:
    min_x = np.min(mask_indices[:, 0])
    min_y = np.min(mask_indices[:, 1])
    take_mask = ((mask_indices[:, 0] - min_x) % stride == 0) & ((mask_indices[:, 1] - min_y) % stride == 0)
    return mask_indices[take_mask, :]


@MMSEG_DATASETS.register_module()
class PerPixelMultiChannelDataset(MultiChannelDataset):
    DUMMY_COLOR = MultiChannelDataset.DUMMY_COLOR
    METAINFO = MultiChannelDataset.METAINFO

    def __init__(self,
                 crop_size: int,
                 img_suffix=".tif",
                 seg_map_suffix=".png",
                 reduce_zero_label=True,
                 stride: int = 4,
                 target_scale: float = 1.0,
                 additional_data_root: str = "",  # for training the aggregator model
                 use_box: bool = False,
                 sample_around_seg: bool = False,
                 prior_masks: Optional[List[np.ndarray]] = None,
                 **kwargs) -> None:
        self.stride = stride
        self.crop_size = crop_size
        self.target_scale = target_scale
        self.half_crop_size = int(self.crop_size / 2)
        self.additional_data_root = additional_data_root
        self.additional_data = None
        self.prior_masks = prior_masks
        self.use_box = use_box
        self.sample_around_seg = sample_around_seg
        assert not (self.sample_around_seg and self.use_box), \
            f"only one of {self.sample_around_seg=} or {self.use_box=} should be True"
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            apparent_length=None,
            **kwargs)

    def _load_additional_data(self):
        if not self.additional_data_root:
            return
        metadata_path = Path(self.additional_data_root) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        self.additional_data = {}
        for md in metadata:
            assert "name" in md
            assert "num_channels" in md
            assert "available_scales" in md
            self.additional_data[md["name"]] = md

    def _get_ss_mask_indices(self, idx, img_meta, scale_meta):
        if self.prior_masks is not None:
            prior_mask = self.prior_masks[idx]
            assert prior_mask.shape == (scale_meta["height"], scale_meta["width"]), \
                f'{prior_mask.shape=} != ({scale_meta["height"]=}, {scale_meta["width"]=})'
            y_indices, x_indices = np.nonzero(self.prior_masks[idx])
            prior_mask_indices = np.stack([x_indices, y_indices], axis=1)
            ss_mask_indices = stride_mask_indices(prior_mask_indices, self.stride)
            print(f"{prior_mask_indices.shape = } -> {ss_mask_indices.shape = }")
        elif self.sample_around_seg:
            seg_path = scale_meta["full_seg_npy_path"]
            seg_map = np.memmap(
                seg_path,
                mode="r",
                dtype=np.uint8,
                shape=(scale_meta["height"], scale_meta["width"])
            )
            y_indices, x_indices = np.nonzero(seg_map)
            seg_mask_indices = np.stack([x_indices, y_indices], axis=1)
            ss_mask_indices = stride_mask_indices(seg_mask_indices, self.stride)
            print(f"{seg_mask_indices.shape = } -> {ss_mask_indices.shape = }")
        elif self.use_box:
            assert "boxes" in img_meta, \
                f"{self.__class__.__name__} instructed to read from box but 'boxes' is not written into img_meta"
            normalised_boxes = img_meta["boxes"]
            boxes = [
                (
                    int(b_lx * scale_meta["width"]),
                    int(b_ly * scale_meta["height"]),
                    int(b_ux * scale_meta["width"]),
                    int(b_uy * scale_meta["height"]),
                )
                for b_lx, b_ly, b_ux, b_uy
                in normalised_boxes
            ]
            ss_mask_indices = np.array([
                [x, y]
                for b_lx, b_ly, b_ux, b_uy in boxes
                for x in range(b_lx, b_ux, self.stride)
                for y in range(b_ly, b_uy, self.stride)
            ])
            print(f"{len(boxes) = } -> {ss_mask_indices.shape = }")
        else:
            mask_indices = np.load(scale_meta["full_mask_indices_path"])
            ss_mask_indices = stride_mask_indices(mask_indices, self.stride)
            print(f"{mask_indices.shape = } -> {ss_mask_indices.shape = }")
        return ss_mask_indices

    def load_data_list(self) -> List[dict]:
        self._load_additional_data()
        img_dataset_list = MultiChannelDataset.load_data_list(self)
        per_pixel_dataset_list = []
        if self.prior_masks is not None:
            assert len(img_dataset_list) == len(self.prior_masks), \
                f"{len(img_dataset_list)=} != {len(self.prior_masks)=}"
        for i, img_meta in enumerate(img_dataset_list):
            scale_str = f"{self.target_scale:.2f}"
            scale_meta = img_meta["available_scales"][scale_str]
            ss_mask_indices = self._get_ss_mask_indices(i, img_meta, scale_meta)
            for x, y in tqdm(ss_mask_indices):
                bbox = [
                    x-self.half_crop_size,
                    y-self.half_crop_size,
                    x+self.half_crop_size,
                    y+self.half_crop_size,
                ]
                if (
                    0 <= bbox[0] < scale_meta["width"]
                    and 0 <= bbox[2] < scale_meta["width"]
                    and 0 <= bbox[1] < scale_meta["height"]
                    and 0 <= bbox[3] < scale_meta["height"]
                ):
                    if self.additional_data is not None:
                        name = img_meta["name"]
                        additional_scale_meta = self.additional_data[name]["available_scales"][scale_str]
                        assert name in self.additional_data, f"additional_data given but does not have name: {name}"
                        assert scale_meta["width"] == additional_scale_meta["width"], f"additional data at scale {scale_str} doesn't have the same width as that of the original image: {scale_meta['width']=} != {additional_scale_meta['width']=}"
                        assert scale_meta["height"] == additional_scale_meta["height"], f"additional data at scale {scale_str} doesn't have the same height as that of the original image: {scale_meta['height']=} != {additional_scale_meta['height']=}"
                        scale_meta["num_additional_channels"] = self.additional_data[name]["num_channels"]
                        scale_meta["additional_channels_npy_path"] = str(Path(self.additional_data_root) / additional_scale_meta["npy_path"])
                    else:
                        scale_meta["num_additional_channels"] = 0
                        scale_meta["additional_channels_npy_path"] = ""
                    bbox_img_meta = {
                        "img_idx": i,
                        "pixel_x": x,
                        "pixel_y": y,
                        "scaled_width": scale_meta["width"],
                        "scaled_height": scale_meta["height"],
                        "target_scale": self.target_scale,
                        "crop_bbox": bbox,
                        **img_meta
                    }
                    per_pixel_dataset_list.append(bbox_img_meta)
        print(f"{len(per_pixel_dataset_list) = }")
        return per_pixel_dataset_list
