import numpy as np
from sennet.core.mmap_arrays import read_mmap_array, MmapArray
from sennet.core.utils import DEPTH_ALONG_CHANNEL, DEPTH_ALONG_HEIGHT, DEPTH_ALONG_WIDTH
from pathlib import Path
import cv2
from line_profiler_pycharm import profile
from scipy.spatial.transform import Rotation
from dataclasses import dataclass


@dataclass
class BboxInfo:
    bbox: tuple[int, int, int, int, int, int]
    crop_size_x: int
    crop_size_y: int
    n_take_channels: int


def slice_3d_array(
        rot: np.ndarray,
        center_xyz: tuple[float, float, float] | np.ndarray,
        bbox_size_xyz: tuple[int, int, int] | np.ndarray,
        image: np.ndarray,
) -> np.ndarray:
    """

    :param rot: 3x3 rotation matrix
    :param center_xyz: center of the bbox along xyz
    :param bbox_size_xyz: bbox size along xyz
    :param image: (c, h, w): image to slice
    :return: sliced image
    """
    (
        pixels_z,
        pixels_y,
        pixels_x,
    ) = np.meshgrid(
        (
            np.linspace(-0.5 * bbox_size_xyz[2], 0.5 * bbox_size_xyz[2], num=bbox_size_xyz[2])
            if bbox_size_xyz[2] > 1 else
            np.zeros(bbox_size_xyz[2])
        ),
        np.linspace(-0.5 * bbox_size_xyz[1], 0.5 * bbox_size_xyz[1], num=bbox_size_xyz[1]),
        np.linspace(-0.5 * bbox_size_xyz[0], 0.5 * bbox_size_xyz[0], num=bbox_size_xyz[0]),
        indexing="ij",
    )

    n_rotated_pixels_x = rot[0, 0] * pixels_x + rot[0, 1] * pixels_y + rot[0, 2] * pixels_z
    n_rotated_pixels_y = rot[1, 0] * pixels_x + rot[1, 1] * pixels_y + rot[1, 2] * pixels_z
    n_rotated_pixels_z = rot[2, 0] * pixels_x + rot[2, 1] * pixels_y + rot[2, 2] * pixels_z
    rotated_pixels_x = n_rotated_pixels_x + center_xyz[0]
    rotated_pixels_y = n_rotated_pixels_y + center_xyz[1]
    rotated_pixels_z = n_rotated_pixels_z + center_xyz[2]

    sliced_image = image[
        np.clip(np.round(rotated_pixels_z).astype(int), 0, image.shape[0]),
        np.clip(np.round(rotated_pixels_y).astype(int), 0, image.shape[1]),
        np.clip(np.round(rotated_pixels_x).astype(int), 0, image.shape[2]),
    ]
    _expected_out_size = bbox_size_xyz[2], bbox_size_xyz[1], bbox_size_xyz[0]
    assert sliced_image.shape == _expected_out_size, f"{sliced_image.shape=} != {_expected_out_size=}"
    return sliced_image


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
            p_random_3d_rotation: float = 0.0,
            rot_magnitude_normal_deg: float = 0.0,
            rot_magnitude_plane_deg: float = 0.0,
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
        self.p_random_3d_rotation = p_random_3d_rotation
        self.rot_magnitude_normal_rad = rot_magnitude_normal_deg / 180.0 * np.pi
        self.rot_magnitude_plane_rad = rot_magnitude_plane_deg / 180.0 * np.pi
        self.loaded_image_mmaps: dict[str, MmapArray] = {}
        self.loaded_seg_mmaps: dict[str, MmapArray] = {}

    # @profile
    @profile
    def _get_pixel_bbox(self, results: dict) -> BboxInfo:
        lc = results["bbox"][0]
        lx = results["bbox"][1]
        ly = results["bbox"][2]
        uc = results["bbox"][3]
        ux = results["bbox"][4]
        uy = results["bbox"][5]

        # _original_box = deepcopy(results["bbox"])
        # "new" cuz they'll be permuted during aug later
        new_crop_size_c = uc - lc
        new_crop_size_x = ux - lx
        new_crop_size_y = uy - ly
        bbox_type = results["bbox_type"]
        if bbox_type == DEPTH_ALONG_CHANNEL:
            crop_size = new_crop_size_x
            n_take_channels = new_crop_size_c
            min_size = int(min(results["img_w"], results["img_h"]))
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            crop_size = new_crop_size_c
            n_take_channels = new_crop_size_y
            min_size = int(min(results["img_w"], results["img_c"]))
        elif bbox_type == DEPTH_ALONG_WIDTH:
            crop_size = new_crop_size_c
            n_take_channels = new_crop_size_x
            min_size = int(min(results["img_h"], results["img_c"]))
        else:
            raise RuntimeError(f"unknown {bbox_type=}")

        should_randomise_crop_location = self.crop_location_noise > 0 and np.random.binomial(p=self.p_crop_location_noise, n=1) > 0.5
        should_randomise_crop_size = self.crop_size_range is not None and np.random.binomial(p=self.p_crop_size_noise, n=1) > 0.5
        if not (should_randomise_crop_location or should_randomise_crop_size):
            return BboxInfo(
                bbox=(lc, lx, ly, uc, ux, uy),
                crop_size_x=crop_size,
                crop_size_y=crop_size,
                n_take_channels=n_take_channels,
            )

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

        bbox_info = BboxInfo(
            bbox=(0, 0, 0, 0, 0, 0),
            crop_size_x=crop_size,
            crop_size_y=crop_size,
            n_take_channels=n_take_channels,
        )
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
                bbox_info.crop_size_x = new_crop_size_x
                bbox_info.crop_size_y = new_crop_size_y
                bbox_info.n_take_channels = new_crop_size_c
            elif bbox_type == DEPTH_ALONG_HEIGHT:
                mid_c += mid_c_noise
                mid_x += mid_x_noise
                if should_keep_ar:
                    new_crop_size_c = new_crop_size
                    new_crop_size_x = new_crop_size
                else:
                    new_crop_size_c = np.random.randint(crop_size_lb, crop_size_ub)
                    new_crop_size_x = np.random.randint(crop_size_lb, crop_size_ub)
                bbox_info.crop_size_x = new_crop_size_c
                bbox_info.crop_size_y = new_crop_size_x
                bbox_info.n_take_channels = new_crop_size_y
            elif bbox_type == DEPTH_ALONG_WIDTH:
                mid_c += mid_c_noise
                mid_y += mid_y_noise
                if should_keep_ar:
                    new_crop_size_c = new_crop_size
                    new_crop_size_y = new_crop_size
                else:
                    new_crop_size_c = np.random.randint(crop_size_lb, crop_size_ub)
                    new_crop_size_y = np.random.randint(crop_size_lb, crop_size_ub)
                bbox_info.crop_size_x = new_crop_size_c
                bbox_info.crop_size_y = new_crop_size_y
                bbox_info.n_take_channels = new_crop_size_x
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
        bbox_info.bbox = _new_box
        return bbox_info

    @profile
    def transform(self, results: dict) -> dict | None:
        crop_bbox = self._get_pixel_bbox(results)
        should_do_3d_aug = bool(np.random.binomial(p=self.p_random_3d_rotation, n=1))
        image_mmap, seg_mmap = self._load_image_and_seg(results)
        if should_do_3d_aug:
            img, seg = self._get_rotated_3d_slice(results, crop_bbox, image_mmap, seg_mmap)
        else:
            img, seg = self._read_image_and_seg(results, crop_bbox.bbox, image_mmap, seg_mmap)
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        if seg is not None:
            results["gt_seg_map"] = seg
        return results

    def _load_image_and_seg(self, results: dict) -> tuple[MmapArray, MmapArray | None]:
        img_path = results["image_dir"]
        if img_path not in self.loaded_image_mmaps:
            self.loaded_image_mmaps[img_path] = read_mmap_array(Path(img_path), mode="r")
        if self.load_ann:
            seg_path = results["seg_dir"]
            if seg_path not in self.loaded_seg_mmaps:
                self.loaded_seg_mmaps[seg_path] = read_mmap_array(Path(seg_path), mode="r")
            seg_mmap = self.loaded_seg_mmaps[seg_path]
        else:
            seg_mmap = None
        return self.loaded_image_mmaps[img_path], seg_mmap

    @profile
    def _get_rotated_3d_slice(
            self,
            results: dict[str, any],
            crop_bbox: BboxInfo,
            image_mmap: MmapArray,
            seg_mmap: MmapArray | None,
    ):
        rot_bound = np.array([self.rot_magnitude_plane_rad, self.rot_magnitude_plane_rad, self.rot_magnitude_normal_rad])
        rot_vec = np.random.uniform(-rot_bound, rot_bound)
        rot = Rotation.from_rotvec(rot_vec).as_matrix()
        bbox_type = results["bbox_type"]
        if bbox_type == DEPTH_ALONG_CHANNEL:
            prefix_rot = np.eye(3)
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            prefix_rot = Rotation.from_rotvec([np.pi/2, 0, 0]).as_matrix()
        elif bbox_type == DEPTH_ALONG_WIDTH:
            prefix_rot = Rotation.from_rotvec([np.pi/2, 0, 0]).as_matrix()
        else:
            raise RuntimeError(f"unknown {bbox_type=}")
        combined_rot = prefix_rot @ rot

        # note: override the bbox to have its normals pointing to z again
        bbox_size_xyz = (
            crop_bbox.crop_size_x,
            crop_bbox.crop_size_y,
            crop_bbox.n_take_channels,
        )
        center_xyz = (
            crop_bbox.bbox[0] + 0.5 * bbox_size_xyz[0],
            crop_bbox.bbox[1] + 0.5 * bbox_size_xyz[1],
            crop_bbox.bbox[2] + 0.5 * bbox_size_xyz[2],
        )
        img_path = results["image_dir"]
        if img_path not in self.loaded_image_mmaps:
            self.loaded_image_mmaps[img_path] = read_mmap_array(Path(img_path), mode="r")
        img = slice_3d_array(
            rot=combined_rot,
            center_xyz=center_xyz,
            bbox_size_xyz=bbox_size_xyz,
            image=image_mmap.data,
        )
        if self.load_ann:
            assert seg_mmap is not None, f"seg_mmap given as None when self.load_ann is True"
            seg_map = slice_3d_array(
                rot=combined_rot,
                center_xyz=center_xyz,
                bbox_size_xyz=bbox_size_xyz,
                image=seg_mmap.data,
            )
        else:
            seg_map = None
        is_fast_path = crop_bbox.crop_size_x == crop_bbox.crop_size_y == self.output_crop_size
        if is_fast_path:
            img = np.ascontiguousarray(img)
            seg_map = np.ascontiguousarray(seg_map)
        else:
            resized_img_arrays = [self._resize_to_output_size(img[c, ...]) for c in range(img.shape[0])]
            img = np.ascontiguousarray(np.stack(resized_img_arrays, axis=0))

            resized_seg_arrays = [self._resize_to_output_size(seg_map[c, ...]) for c in range(seg_map.shape[0])]
            seg_map = np.ascontiguousarray(np.stack(resized_seg_arrays, axis=0))
        return img, seg_map

    @profile
    def _read_image_and_seg(
            self,
            results: dict[str, any],
            crop_bbox: tuple[int, int, int, int, int, int],
            image_mmap: MmapArray,
            seg_mmap: MmapArray | None,
    ):
        img = self._get_3d_slice(results, image_mmap.data, crop_bbox)
        if self.load_ann:
            assert seg_mmap is not None, f"seg_mmap given as None when self.load_ann is True"
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
