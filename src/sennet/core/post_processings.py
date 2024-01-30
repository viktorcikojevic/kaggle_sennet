from sennet.core.mmap_arrays import create_mmap_array
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cc3d


def filter_out_small_blobs(
        thresholded_pred: np.ndarray,
        out_path: str | Path | None,
        dust_threshold: int = 1000,
        connectivity: int = 26,
) -> np.ndarray | None:
    print(f"computing cc3d object: {thresholded_pred.shape}")
    print(f"computing dust")
    labels_out = cc3d.dust(
        thresholded_pred,
        threshold=dust_threshold,
        connectivity=connectivity,
        in_place=True,
    )
    if out_path is None:
        print("returning in memory filtered array")
        filtered_pred = labels_out > 0
        return filtered_pred
    else:
        print("dumping output mmap")
        out_mmap = create_mmap_array(
            Path(out_path),
            list(thresholded_pred.shape),
            bool,
        )
        out_mmap.data[:] = False
        out_mmap.data[labels_out > 0] = True
        print("flushing out mmap")
        out_mmap.data.flush()


def intersect_3d_bbox(
        bbox0: list[int],
        bbox1: list[int]
) -> None | list[int]:
    x0_min, y0_min, z0_min, x0_max, y0_max, z0_max = bbox0
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = bbox1

    inter_x_min = max(x0_min, x1_min)
    inter_y_min = max(y0_min, y1_min)
    inter_z_min = max(z0_min, z1_min)

    inter_x_max = min(x0_max, x1_max)
    inter_y_max = min(y0_max, y1_max)
    inter_z_max = min(z0_max, z1_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max and inter_z_min < inter_z_max:
        intersection_bbox = [int(inter_x_min), int(inter_y_min), int(inter_z_min), int(inter_x_max), int(inter_y_max), int(inter_z_max)]
        return intersection_bbox
    else:
        return None


def cc3d_bbox_to_bbox(bbox):
    return [bbox[4], bbox[2], bbox[0], bbox[5], bbox[3], bbox[1]]


def largest_k_closest_to_center(
        thresholded_pred: np.ndarray,
        connectivity: int = 26,
        largest_k: int = 100,
        disable_tqdm: bool = False,
        out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        out = np.zeros_like(thresholded_pred, dtype=bool)
    else:
        assert out.shape == thresholded_pred.shape, f"{out.shape=} != {thresholded_pred.shape=}"

    # NOTE: don't use this with incomplete kidneys (so no k3d or half kidneys) since it makes assumptions that the
    #  largest vessels largest k clusters
    largest_k_out, label_n = cc3d.largest_k(
        thresholded_pred,
        k=largest_k,
        connectivity=connectivity,
        return_N=True,
    )
    stats = cc3d.statistics(largest_k_out, no_slice_conversion=True)

    # these are sorted ascending via voxel count (so the 100th cluster is the largest)
    bboxes = stats["bounding_boxes"][1:]
    # voxel_counts = stats["voxel_counts"][1:]  # it's there, though we don't need it yet
    # centroids = stats["centroids"][1:]  # it's there, though we don't need it yet

    if len(bboxes) == 0:
        return out

    largest_bbox = cc3d_bbox_to_bbox(bboxes[-1])
    keeps = []
    for b in tqdm(bboxes[:-1], disable=disable_tqdm):
        bbox = cc3d_bbox_to_bbox(b)
        intersection = intersect_3d_bbox(largest_bbox, bbox)
        if intersection is None:
            keeps.append(False)
        else:
            keeps.append(True)
    keeps.append(True)

    for label, image in tqdm(cc3d.each(largest_k_out, binary=True, in_place=True), disable=disable_tqdm):
        # label starts from 1
        i = label-1
        if not keeps[i]:
            continue
        out[image] = True
    return out
