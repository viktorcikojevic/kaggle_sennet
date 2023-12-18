from typing import *
import numpy as np
from sennet.core.mmap_arrays import read_mmap_array, MmapArray
from sennet.core.utils import DEPTH_ALONG_CHANNEL, DEPTH_ALONG_HEIGHT, DEPTH_ALONG_WIDTH
from pathlib import Path
import cv2
from line_profiler_pycharm import profile


class LoadMultiChannelImageAndAnnotationsFromFile:
    def __init__(
            self,
            crop_size_range: Optional[Tuple[int, int]],
            output_crop_size: int,
            to_float32: bool = False,
            channels_jitter: int = 0,
            p_channel_jitter: float = 0.0,
            load_ann: bool = True,
            seg_fill_val: int = 255,
            crop_location_noise: int = 0
    ):
        if crop_size_range is not None:
            assert crop_size_range[0] <= crop_size_range[1], f"{crop_size_range=}"
        self.crop_size_range = crop_size_range
        self.output_crop_size = output_crop_size
        self.to_float32 = to_float32
        self.channels_jitter = channels_jitter
        self.p_channel_jitter = p_channel_jitter
        self.load_ann = load_ann
        self.seg_fill_val = seg_fill_val
        self.crop_location_noise = crop_location_noise
        self.loaded_image_mmaps: Dict[str, MmapArray] = {}
        self.loaded_seg_mmaps: Dict[str, MmapArray] = {}

    @profile
    def _get_take_channels(self, results: Dict) -> Tuple[bool, Union[Tuple[int, int], np.ndarray]]:
        # skip some channels and take others (like not taking consecutive channels)?
        channel_start = results["bbox"][0]
        channel_end = results["bbox"][3]
        num_channels = results["img_c"]
        if self.p_channel_jitter > 1e-3 and np.random.binomial(n=1, p=self.p_channel_jitter):
            # NOTE: I don't think I need this, the scanning machine is pretty precise and won't have this noise
            min_jitter_val = -min(channel_start, self.channels_jitter)
            max_jitter_val = min(num_channels-channel_end, self.channels_jitter)
            channel_jitter = np.random.randint(min_jitter_val, max_jitter_val)
            take_channels = np.arange(start=channel_start, stop=channel_end, step=1)
            take_channels += channel_jitter
            return True, take_channels.astype(int)
        else:
            return False, (channel_start, channel_end)

    # @profile
    def _get_pixel_bbox(self, results: Dict) -> Tuple[int, int, int, int]:
        # TODO(Sumo): you'll need to account for the fact that different bbox type has
        #  different location of "x" and "y" wrt their bboxes
        lx = results["bbox"][1]
        ly = results["bbox"][2]
        ux = results["bbox"][4]
        uy = results["bbox"][5]
        if self.crop_location_noise == 0:
            return lx, ly, ux, uy
        mid_x_noise = np.random.randint(-self.crop_location_noise, self.crop_location_noise+1)
        mid_y_noise = np.random.randint(-self.crop_location_noise, self.crop_location_noise+1)
        mid_x = int(0.5*lx + 0.5*ux) + mid_x_noise
        mid_y = int(0.5*ly + 0.5*uy) + mid_y_noise
        if self.crop_size_range is not None:
            new_crop_size_x = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
            new_crop_size_y = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
        else:
            new_crop_size_x = ux - lx
            new_crop_size_y = uy - ly
        lx = int(np.clip(mid_x - 0.5*new_crop_size_x, 0, results["img_w"] - new_crop_size_x))
        ly = int(np.clip(mid_y - 0.5*new_crop_size_y, 0, results["img_h"] - new_crop_size_y))
        ux = int(lx + new_crop_size_x)
        uy = int(ly + new_crop_size_y)
        return lx, ly, ux, uy

    @profile
    def transform(self, results: Dict) -> Optional[Dict]:
        crop_bbox = self._get_pixel_bbox(results)
        is_channels_augmented, take_channels = self._get_take_channels(results)
        img, seg = self._read_image_and_seg(
            results,
            crop_bbox,
            take_channels,
            is_channels_augmented,
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        if seg is not None:
            results["gt_seg_map"] = seg
        return results

    @profile
    def _read_image_and_seg(
            self,
            results,
            crop_bbox,
            take_channels,
            is_channels_augmented: bool,
    ):
        img_path = results["image_dir"]
        if img_path not in self.loaded_image_mmaps:
            self.loaded_image_mmaps[img_path] = read_mmap_array(Path(img_path), mode="r")
        image_mmap = self.loaded_image_mmaps[img_path]
        img = self._get_3d_slice(image_mmap.data, crop_bbox, take_channels, is_channels_augmented)
        if self.load_ann:
            seg_path = results["seg_dir"]
            if seg_path not in self.loaded_seg_mmaps:
                self.loaded_seg_mmaps[seg_path] = read_mmap_array(Path(seg_path), mode="r")
            seg_mmap = self.loaded_seg_mmaps[seg_path]
            seg_map = self._get_3d_slice(seg_mmap.data, crop_bbox, take_channels, is_channels_augmented)
        else:
            seg_map = None

        # molding crops so they look the same for the model
        bbox_type = results["bbox_type"]
        if bbox_type == DEPTH_ALONG_CHANNEL:
            # we're stacking along channel by default
            img = np.ascontiguousarray(img)
            if seg_map is not None:
                seg_map = np.ascontiguousarray(seg_map)
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            # (c, d, c) -> (d, c, c)
            img = np.ascontiguousarray(np.stack([
                img[:, c, :]
                for c in range(img.shape[1])
            ], axis=0))
            if seg_map is not None:
                seg_map = np.ascontiguousarray(np.stack([
                    seg_map[:, c, :]
                    for c in range(seg_map.shape[1])
                ], axis=0))
        elif bbox_type == DEPTH_ALONG_WIDTH:
            # (c, c, d) -> (d, c, c)
            img = np.ascontiguousarray(np.stack([
                img[:, :, c]
                for c in range(img.shape[2])
            ], axis=0))
            if seg_map is not None:
                seg_map = np.ascontiguousarray(np.stack([
                    seg_map[:, :, c]
                    for c in range(seg_map.shape[2])
                ], axis=0))
        else:
            print(f"unknown {bbox_type=}")
        return img, seg_map

    @profile
    def _get_3d_slice(
            self,
            full_array: np.ndarray,
            crop_bbox,
            take_channels,
            is_channels_augmented: bool = False,
    ) -> np.ndarray:
        # TODO(Sumo): revive when crop bbox jitter is back (or just move 3d augmentations out really)
        lx, ly, ux, uy = crop_bbox
        take_img = full_array[take_channels[0]: take_channels[1], ly: uy, lx: ux]
        expected_img_shape = take_channels[1] - take_channels[0], uy-ly, ux-lx
        assert take_img.shape == expected_img_shape, f"{take_img.shape=} != {expected_img_shape=}"
        return take_img
        # if ux-lx == self.output_crop_size and uy-ly == self.output_crop_size:
        #     if is_channels_augmented:
        #         return full_array[take_channels, ly: uy, lx: ux]
        #     else:
        #         return full_array[take_channels[0]: take_channels[1], ly: uy, lx: ux]
        # resized_arrays = [
        #     self._resize_to_output_size(full_array[c, ly: uy, lx: ux])
        #     for c in (take_channels if is_channels_augmented else range(take_channels[0], take_channels[1]))
        # ]
        # stacked_array = np.stack(resized_arrays, axis=0)
        # return stacked_array

    def _resize_to_output_size(self, img: np.ndarray):
        if img.shape[0] == img.shape[1] == self.output_crop_size:
            return img
        return cv2.resize(
            img,
            dsize=(self.output_crop_size, self.output_crop_size),
            interpolation=cv2.INTER_AREA
        )

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__}("
                    f"to_float32={self.to_float32}, "
                    f"channels_jitter={self.channels_jitter}, "
                    f"p_channel_jitter={self.p_channel_jitter})")
        return repr_str
