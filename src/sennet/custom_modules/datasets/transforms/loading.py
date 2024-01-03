import numpy as np
from sennet.core.mmap_arrays import read_mmap_array, MmapArray
from sennet.core.utils import DEPTH_ALONG_CHANNEL, DEPTH_ALONG_HEIGHT, DEPTH_ALONG_WIDTH
from pathlib import Path
import cv2
from line_profiler_pycharm import profile
from copy import deepcopy


class LoadMultiChannelImageAndAnnotationsFromFile:
    def __init__(
            self,
            crop_size_range: tuple[float, float] | None,
            output_crop_size: int,
            to_float32: bool = False,
            load_ann: bool = True,
            seg_fill_val: int = 255,
            crop_location_noise: float = 0.0,
            p_crop_location_noise: float = 0.0,
            p_crop_size_noise: float = 0.0,
            p_crop_size_keep_ar: float = 0.5,
    ):
        if crop_size_range is not None:
            assert crop_size_range[0] <= crop_size_range[1], f"{crop_size_range=}"
        self.crop_size_range = crop_size_range
        self.output_crop_size = output_crop_size
        self.to_float32 = to_float32
        self.load_ann = load_ann
        self.seg_fill_val = seg_fill_val
        self.crop_location_noise = crop_location_noise
        self.p_crop_location_noise = p_crop_location_noise
        self.p_crop_size_noise = p_crop_size_noise
        self.p_crop_size_keep_ar = p_crop_size_keep_ar
        self.loaded_image_mmaps: dict[str, MmapArray] = {}
        self.loaded_seg_mmaps: dict[str, MmapArray] = {}

    # @profile
    @profile
    def _get_pixel_bbox(self, results: dict) -> tuple[int, int, int, int, int, int]:
        lc = results["bbox"][0]
        lx = results["bbox"][1]
        ly = results["bbox"][2]
        uc = results["bbox"][3]
        ux = results["bbox"][4]
        uy = results["bbox"][5]

        # _original_box = deepcopy(results["bbox"])

        should_randomise_crop_location = self.crop_location_noise > 0 and np.random.binomial(p=self.p_crop_location_noise, n=1) > 0.5
        should_randomise_crop_size = self.crop_size_range is not None and np.random.binomial(p=self.p_crop_size_noise, n=1) > 0.5
        if not (should_randomise_crop_location or should_randomise_crop_size):
            return lc, lx, ly, uc, ux, uy

        # "new" cuz they'll be permuted during aug later
        new_crop_size_c = uc - lc
        new_crop_size_x = ux - lx
        new_crop_size_y = uy - ly
        bbox_type = results["bbox_type"]
        if bbox_type == DEPTH_ALONG_CHANNEL:
            crop_size = new_crop_size_x
            min_size = int(min(results["img_w"], results["img_h"]))
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            crop_size = new_crop_size_c
            min_size = int(min(results["img_w"], results["img_c"]))
        elif bbox_type == DEPTH_ALONG_WIDTH:
            crop_size = new_crop_size_c
            min_size = int(min(results["img_h"], results["img_c"]))
        else:
            raise RuntimeError(f"unknown {bbox_type=}")

        if should_randomise_crop_location:
            crop_location_noise = int(self.crop_location_noise * crop_size)
            mid_x_noise = np.random.randint(-crop_location_noise, crop_location_noise+1)
            mid_y_noise = np.random.randint(-crop_location_noise, crop_location_noise+1)
            mid_c_noise = np.random.randint(-crop_location_noise, crop_location_noise+1)
        else:
            mid_x_noise = 0
            mid_y_noise = 0
            mid_c_noise = 0
        mid_x = 0.5*lx + 0.5*ux
        mid_y = 0.5*ly + 0.5*uy
        mid_c = 0.5*lc + 0.5*uc

        should_keep_ar = np.random.binomial(p=self.p_crop_size_keep_ar, n=1) > 0.5
        crop_size_lb = int(self.crop_size_range[0] * crop_size)
        crop_size_ub = min(int(self.crop_size_range[1] * crop_size), min_size)
        new_crop_size = np.random.randint(crop_size_lb, crop_size_ub)
        if should_randomise_crop_size:
            if bbox_type == DEPTH_ALONG_CHANNEL:
                mid_x += mid_x_noise
                mid_y += mid_y_noise
                if should_keep_ar:
                    new_crop_size_x = new_crop_size
                    new_crop_size_y = new_crop_size
                else:
                    new_crop_size_x = np.random.randint(crop_size_lb, crop_size_ub)
                    new_crop_size_y = np.random.randint(crop_size_lb, crop_size_ub)
            elif bbox_type == DEPTH_ALONG_HEIGHT:
                mid_c += mid_c_noise
                mid_x += mid_x_noise
                if should_keep_ar:
                    new_crop_size_c = new_crop_size
                    new_crop_size_x = new_crop_size
                else:
                    new_crop_size_c = np.random.randint(crop_size_lb, crop_size_ub)
                    new_crop_size_x = np.random.randint(crop_size_lb, crop_size_ub)
            elif bbox_type == DEPTH_ALONG_WIDTH:
                mid_c += mid_c_noise
                mid_y += mid_y_noise
                if should_keep_ar:
                    new_crop_size_c = new_crop_size
                    new_crop_size_y = new_crop_size
                else:
                    new_crop_size_c = np.random.randint(crop_size_lb, crop_size_ub)
                    new_crop_size_y = np.random.randint(crop_size_lb, crop_size_ub)
            else:
                raise RuntimeError(f"unknown {bbox_type=}")
        lc = int(np.clip(mid_c - 0.5*new_crop_size_c, 0, results["img_c"] - new_crop_size_c))
        lx = int(np.clip(mid_x - 0.5*new_crop_size_x, 0, results["img_w"] - new_crop_size_x))
        ly = int(np.clip(mid_y - 0.5*new_crop_size_y, 0, results["img_h"] - new_crop_size_y))
        uc = int(lc + new_crop_size_c)
        ux = int(lx + new_crop_size_x)
        uy = int(ly + new_crop_size_y)

        _new_box = lc, lx, ly, uc, ux, uy
        # print(f"augmented bbox from: {_original_box} -> {_new_box}, {bbox_type=}")
        return _new_box

    @profile
    def transform(self, results: dict) -> dict | None:
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
            results: dict[str, any],
            crop_bbox: tuple[int, int, int, int, int, int],
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
            results: dict[str, any],
            full_array: np.ndarray,
            crop_bbox: tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        # crops from the original 3d array
        # molding crops, so they look the same for the model
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
                # take_img = np.ascontiguousarray(np.stack([
                #     full_array[lc: uc, y, lx: ux]
                #     for y in range(ly, uy)
                # ], axis=0))
                # equivalent with above, checked, trust me
                take_img = np.ascontiguousarray(full_array[lc: uc, ly: uy, lx: ux].transpose((1, 0, 2)))
            else:
                resized_arrays = [self._resize_to_output_size(full_array[lc: uc, y, lx: ux]) for y in range(ly, uy)]
                take_img = np.ascontiguousarray(np.stack(resized_arrays, axis=0))
        elif bbox_type == DEPTH_ALONG_WIDTH:
            # (c, c, d) -> (d, c, c)
            n_take_channels = ux-lx
            is_fast_path = uc-lc == self.output_crop_size and uy-ly == self.output_crop_size
            if is_fast_path:
                # take_img = np.ascontiguousarray(np.stack([
                #     full_array[lc: uc, ly: uy, x]
                #     for x in range(lx, ux)
                # ], axis=0))
                # equivalent with above, checked, trust me
                take_img = np.ascontiguousarray(full_array[lc: uc, ly: uy, lx: ux].transpose((2, 0, 1)))
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
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"crop_location_noise={self.crop_location_noise})"
            f"crop_size_range={self.crop_size_range})"
        )
        return repr_str
