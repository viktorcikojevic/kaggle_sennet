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

    # @profile
    def _get_pixel_bbox(self, results: Dict) -> Tuple[int, int, int, int, int, int]:
        lc = results["bbox"][0]
        lx = results["bbox"][1]
        ly = results["bbox"][2]
        uc = results["bbox"][3]
        ux = results["bbox"][4]
        uy = results["bbox"][5]
        if self.crop_location_noise == 0:
            return lc, lx, ly, uc, ux, uy

        mid_x_noise = np.random.randint(-self.crop_location_noise, self.crop_location_noise+1)
        mid_y_noise = np.random.randint(-self.crop_location_noise, self.crop_location_noise+1)
        mid_c_noise = np.random.randint(-self.crop_location_noise, self.crop_location_noise+1)
        mid_x = int(0.5*lx + 0.5*ux) + mid_x_noise
        mid_y = int(0.5*ly + 0.5*uy) + mid_y_noise
        mid_c = int(0.5*lc + 0.5*uc) + mid_c_noise

        new_crop_size_c = uc - lc
        new_crop_size_x = ux - lx
        new_crop_size_y = uy - ly
        if self.crop_size_range is not None:
            bbox_type = results["bbox_type"]
            if bbox_type == DEPTH_ALONG_CHANNEL:
                new_crop_size_x = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
                new_crop_size_y = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
            elif bbox_type == DEPTH_ALONG_HEIGHT:
                new_crop_size_c = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
                new_crop_size_x = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
            elif bbox_type == DEPTH_ALONG_WIDTH:
                new_crop_size_c = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
                new_crop_size_y = np.random.randint(self.crop_size_range[0], self.crop_size_range[1])
            else:
                raise RuntimeError(f"unknown {bbox_type=}")
        lc = int(np.clip(mid_c - 0.5*new_crop_size_c, 0, results["img_c"] - new_crop_size_c))
        lx = int(np.clip(mid_x - 0.5*new_crop_size_x, 0, results["img_w"] - new_crop_size_x))
        ly = int(np.clip(mid_y - 0.5*new_crop_size_y, 0, results["img_h"] - new_crop_size_y))
        uc = int(lc + new_crop_size_c)
        ux = int(lx + new_crop_size_x)
        uy = int(ly + new_crop_size_y)
        return lc, lx, ly, uc, ux, uy

    @profile
    def transform(self, results: Dict) -> Optional[Dict]:
        crop_bbox = self._get_pixel_bbox(results)
        img, seg = self._read_image_and_seg(
            results,
            crop_bbox,
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        if seg is not None:
            results["gt_seg_map"] = seg
        return results

    @profile
    def _read_image_and_seg(
            self,
            results: Dict[str, Any],
            crop_bbox: Tuple[int, int, int, int, int, int],
    ):
        img_path = results["image_dir"]
        if img_path not in self.loaded_image_mmaps:
            self.loaded_image_mmaps[img_path] = read_mmap_array(Path(img_path), mode="r")
        image_mmap = self.loaded_image_mmaps[img_path]
        img = self._get_3d_slice(results, image_mmap.data, crop_bbox)
        if self.load_ann:
            seg_path = results["seg_dir"]
            if seg_path not in self.loaded_seg_mmaps:
                self.loaded_seg_mmaps[seg_path] = read_mmap_array(Path(seg_path), mode="r")
            seg_mmap = self.loaded_seg_mmaps[seg_path]
            seg_map = self._get_3d_slice(results, seg_mmap.data, crop_bbox)
        else:
            seg_map = None
        return img, seg_map

    @profile
    def _get_3d_slice(
            self,
            results: Dict[str, Any],
            full_array: np.ndarray,
            crop_bbox: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        # crops from the original 3d array
        # molding crops so they look the same for the model
        # optionally take the fast past if applicable
        lc, lx, ly, uc, ux, uy = crop_bbox

        bbox_type = results["bbox_type"]
        if bbox_type == DEPTH_ALONG_CHANNEL:
            # we're stacking along channel by default
            n_take_channels = uc-lc
            is_fast_path = ux-lx == self.output_crop_size and uy-ly == self.output_crop_size
            if is_fast_path:
                take_img = np.ascontiguousarray(full_array[lc: uc, ly: uy, lx: ux])
            else:
                resized_arrays = [self._resize_to_output_size(full_array[c, ly: uy, lx: ux]) for c in range(lc, uc)]
                take_img = np.ascontiguousarray(np.stack(resized_arrays, axis=0))
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            # (c, d, c) -> (d, c, c)
            n_take_channels = uy-ly
            is_fast_path = ux-lx == self.output_crop_size and uc-lc == self.output_crop_size
            if is_fast_path:
                take_img = np.ascontiguousarray(np.stack([
                    full_array[lc: uc, y, lx: ux]
                    for y in range(ly, uy)
                ], axis=0))
            else:
                resized_arrays = [self._resize_to_output_size(full_array[lc: uc, y, lx: ux]) for y in range(ly, uy)]
                take_img = np.ascontiguousarray(np.stack(resized_arrays, axis=0))
        elif bbox_type == DEPTH_ALONG_WIDTH:
            # (c, c, d) -> (d, c, c)
            n_take_channels = ux-lx
            is_fast_path = uc-lc == self.output_crop_size and uy-ly == self.output_crop_size
            if is_fast_path:
                take_img = np.ascontiguousarray(np.stack([
                    full_array[lc: uc, ly: uy, x]
                    for x in range(lx, ux)
                ], axis=0))
            else:
                resized_arrays = [self._resize_to_output_size(full_array[lc: uc, ly: uy, x]) for x in range(lx, ux)]
                take_img = np.ascontiguousarray(np.stack(resized_arrays, axis=0))
        else:
            raise RuntimeError(f"unknown {bbox_type=}")

        expected_img_shape = n_take_channels, self.output_crop_size, self.output_crop_size
        if take_img.shape != expected_img_shape:
            raise RuntimeError(f"{take_img.shape=} != {expected_img_shape=}")
        return take_img

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
