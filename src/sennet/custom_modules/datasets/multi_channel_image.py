from sennet.core.utils import DEPTH_ALONG_CHANNEL, DEPTH_ALONG_WIDTH, DEPTH_ALONG_HEIGHT
from sennet.environments.constants import PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from typing import *
import numpy as np


def generate_crop_bboxes(
        crop_size: int,
        n_take_channels: int,
        substride: float,
        shape: Tuple[int, int, int],
        mask: Optional[np.ndarray] = None,
        depth_mode: int = DEPTH_ALONG_CHANNEL,
) -> np.ndarray:
    # bbox is always [channel_lb, x_lb, y_lb, channel_ub, x_ub, y_ub]
    bboxes = []

    crop_stride = max(1, int(substride * crop_size))
    channel_stride = max(1, int(substride * n_take_channels))

    c_stride = channel_stride if depth_mode == DEPTH_ALONG_CHANNEL else crop_stride
    c_take_range = n_take_channels if depth_mode == DEPTH_ALONG_CHANNEL else crop_size

    y_stride = channel_stride if depth_mode == DEPTH_ALONG_HEIGHT else crop_stride
    y_take_range = n_take_channels if depth_mode == DEPTH_ALONG_HEIGHT else crop_size

    x_stride = channel_stride if depth_mode == DEPTH_ALONG_WIDTH else crop_stride
    x_take_range = n_take_channels if depth_mode == DEPTH_ALONG_WIDTH else crop_size

    for c in range(0, shape[0] - c_take_range, c_stride):
        for i in range(0, shape[1] - y_take_range, y_stride):
            for j in range(0, shape[2] - x_take_range, x_stride):
                lc = c
                lx = j
                ly = i
                uc = c + c_take_range
                ux = j + x_take_range
                uy = i + y_take_range
                box = [lc, lx, ly, uc, ux, uy]
                if (mask is not None) and (not np.any(mask[lc:uc, ly:uy, lx:ux])):
                    continue
                bboxes.append(box)
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
            add_depth_along_channel: bool = True,
            add_depth_along_width: bool = False,
            add_depth_along_height: bool = False,
    ) -> None:
        self.folder = PROCESSED_DATA_DIR / folder
        print(f"reading from the following folder: {self.folder}")
        self.crop_size = crop_size
        self.n_take_channels = n_take_channels
        self.substride = substride
        self.assert_label_exists = assert_label_exists
        self.channel_start = channel_start
        self.channel_end = channel_end
        self.reduce_zero_label = reduce_zero_label
        self.sample_with_mask = sample_with_mask
        self.image_paths = {}
        self.add_depth_along_channel = add_depth_along_channel
        self.add_depth_along_width = add_depth_along_width
        self.add_depth_along_height = add_depth_along_height
        self._load_data_list()

    def __len__(self):
        return self.bboxes.shape[0]

    def __getitem__(self, i):
        return dict(
            bbox=self.bboxes[i],
            bbox_type=self.bbox_types[i],
            **self.general_metadata,
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
        self.bboxes = np.zeros((0, 6), dtype=int)
        self.bbox_types = np.zeros(0, dtype=int)
        for (flag, bbox_type, msg) in (
                (self.add_depth_along_channel, DEPTH_ALONG_CHANNEL, "channel"),
                (self.add_depth_along_width, DEPTH_ALONG_WIDTH, "width"),
                (self.add_depth_along_height, DEPTH_ALONG_HEIGHT, "height"),
        ):
            if not flag:
                continue
            new_bboxes = generate_crop_bboxes(
                crop_size=self.crop_size,
                n_take_channels=self.n_take_channels,
                substride=self.substride,
                shape=(mask.shape[0], mask.shape[1], mask.shape[2]),
                mask=mask.data if self.sample_with_mask else None,
                depth_mode=bbox_type,
            )
            new_bbox_types = np.full(len(new_bboxes), bbox_type)
            print(f"adding depth along {msg}: {new_bboxes.shape[0]}")
            if new_bboxes.shape[0] > 0:
                self.bboxes = np.concatenate((self.bboxes, new_bboxes), axis=0)
                self.bbox_types = np.concatenate((self.bbox_types, new_bbox_types))
        print(f"{self.folder}: bboxes={len(self.bboxes)}")
        print("done generating indices")


if __name__ == "__main__":
    _ds = MultiChannelDataset(
        "kidney_1_dense",
        512,
        20,
        substride=0.5,
        sample_with_mask=True,
        add_depth_along_channel=True,
        add_depth_along_height=True,
        add_depth_along_width=True,
    )
