import json

from sennet.core.mmap_arrays import create_mmap_array, MmapArray
from sennet.environments.constants import PROCESSED_DATA_DIR, FG_MASK_DIR
# from sennet.core.foreground_extraction import get_foreground_mask
from typing import Optional
from pathlib import Path
import numpy as np
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ignore-mask", action="store_true", required=False)
    parser.add_argument("--label-as-mask", action="store_true", required=False)
    args, _ = parser.parse_known_args()
    path = Path(args.path)
    output_dir = (PROCESSED_DATA_DIR / path.name)
    ignore_mask = args.ignore_mask
    label_as_mask = args.label_as_mask
    assert path.is_dir(), f"{path=} doesn't exist"
    images_dir = path / "images"
    labels_dir = path / "labels"
    output_dir.mkdir(exist_ok=True, parents=True)

    if images_dir.is_dir():
        image_paths = sorted(list(images_dir.glob("*.tif")))
        (output_dir / "image_paths").write_text("\n".join([str(p) for p in image_paths]))
        print(f"found {len(image_paths)} images under {images_dir}")
        mmap_array: Optional[MmapArray] = None
        mask_mmap_array: Optional[MmapArray] = None
        fg_mask_dir = FG_MASK_DIR / images_dir.parent.name
        fg_mask_paths = sorted(list(fg_mask_dir.glob("*")))
        if ignore_mask:
            pass
        elif label_as_mask:
            pass
        elif len(fg_mask_paths) > 0:
            assert len(fg_mask_paths) == len(image_paths), f"{len(fg_mask_paths)=} != {len(image_paths)=}"
        print(f"{fg_mask_dir=}, {len(fg_mask_paths)=}")
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path), 0)
            if i == 0:
                # (c, h, w) to streamline conversion to torch
                shape = [len(image_paths), image.shape[0], image.shape[1]]
                mmap_array = create_mmap_array(output_dir / "image", shape, np.uint8)
                mask_mmap_array = create_mmap_array(output_dir / "mask", shape, bool)
            mmap_array.data[i, :, :] = image
            # fg_mask = get_foreground_mask(image)
            # # if fg_mask.sum() == 0:
            # #     print(f"0 mask found on {i}")
            # # if fg_mask.mean() > 0.9:
            # #     print(f"big mask found on {i}")
            # mask_mmap_array.data[i, :, :] = fg_mask
            if ignore_mask:
                pass
            elif label_as_mask:
                pass
            elif len(fg_mask_paths) > 0:
                fg_mask_path = fg_mask_paths[i]
                mask = cv2.imread(str(fg_mask_path), 0)
                resized_mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0])) > 0
                mask_mmap_array.data[i, :, :] = resized_mask
            else:
                mask_mmap_array.data[i, :, :] = 1
            print(f"done images: {i+1}/{len(image_paths)}: {image_path}")

        print(f"computing stats")
        ps = [1, 0.001]
        percentiles = ps + [100 - p for p in ps[::-1]]
        channel_margin = 0.2
        channel_lb = int(channel_margin * mmap_array.data.shape[0])
        channel_ub = int((1 - channel_margin) * mmap_array.data.shape[0])
        global_percentile = np.percentile(mmap_array.data[channel_lb: channel_ub, :], percentiles)
        global_mean = np.mean(mmap_array.data[channel_lb: channel_ub, :])
        global_std = np.std(mmap_array.data[channel_lb: channel_ub, :])
        Path(output_dir / "stats.json").write_text(json.dumps(
            {
                "mean": float(global_mean),
                "std": float(global_std),
                "percentiles": {
                    p: float(g)
                    for p, g in zip(percentiles, global_percentile)
                },
            },
            indent=4,
        ))

        print(f"flushing images")
        if mmap_array is not None:
            mmap_array.data.flush()
        if mask_mmap_array is not None:
            mask_mmap_array.data.flush()
        print(f"done images")
    else:
        print(f"{images_dir=} doesn't exist, skipping")

    if labels_dir.is_dir():
        label_paths = sorted(list(labels_dir.glob("*.tif")))
        print(f"found {len(label_paths)} label_paths under {labels_dir}")
        mmap_array: Optional[MmapArray] = None
        for i, label_path in enumerate(label_paths):
            label = cv2.imread(str(label_path), 0)
            if i == 0:
                # (c, h, w) to streamline conversion to torch
                shape = [len(label_paths), label.shape[0], label.shape[1]]
                mmap_array = create_mmap_array(output_dir / "label", shape, np.uint8)
            mmap_array.data[i, :, :] = (label > 0).astype(np.uint8)
        print(f"flushing labels")
        if mmap_array is not None:
            mmap_array.data.flush()
        if label_as_mask:
            mask_mmap_array = create_mmap_array(output_dir / "mask", list(mmap_array.data.shape), bool)
            mask_mmap_array.data[:] = False
            mask_mmap_array.data[:] = mmap_array.data > 0
            mask_mmap_array.data.flush()
        print(f"done labels")
    else:
        print(f"{labels_dir=} doesn't exist, skipping")


if __name__ == "__main__":
    main()
