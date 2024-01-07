from src.sennet.environments.constants import PROCESSED_DATA_DIR
from src.sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from tqdm import tqdm
from line_profiler_pycharm import profile
import numpy as np
import argparse


@profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, type=str)
    args, _ = parser.parse_known_args()

    data_dir = PROCESSED_DATA_DIR / args.folder
    image_mmap = read_mmap_array(data_dir / "image")
    mask_mmap = read_mmap_array(data_dir / "mask")
    label_mmap = read_mmap_array(data_dir / "label")
    stride = 2
    image = image_mmap.data[::stride, ::stride, ::stride]
    mask = mask_mmap.data[::stride, ::stride, ::stride] > 0
    label = label_mmap.data[::stride, ::stride, ::stride] > 0
    print(f"{mask.sum() / mask.shape[0] / mask.shape[1] / mask.shape[2]}")
    image_out_path = data_dir / "cloud.ply"
    label_out_path = data_dir / "label.ply"
    n_cloud_points = mask.sum()
    n_label_points = (mask & (label > 0)).sum()
    Path(image_out_path).write_text(
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_cloud_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar intensity\n"
        "end_header\n"
    )
    Path(label_out_path).write_text(
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_label_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar intensity\n"
        "end_header\n"
    )
    scaling = 1e-3
    xs, ys = np.meshgrid(
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )
    xs = xs.astype(np.float32).ravel() * scaling
    ys = ys.astype(np.float32).ravel() * scaling
    num_points = xs.shape[0]
    print(f"xs: {np.min(xs)}: {np.max(xs)}")
    print(f"ys: {np.min(ys)}: {np.max(ys)}")
    print(f"zs: 0: {image.shape[0] * scaling}")
    xs = np.frombuffer(xs.tobytes(), dtype=np.uint8).reshape((-1, 4))
    ys = np.frombuffer(ys.tobytes(), dtype=np.uint8).reshape((-1, 4))
    assert xs.shape[0] == num_points, f"{xs.shape[0]=} != {num_points=}"
    n_written_points = 0
    with open(label_out_path, "ab") as label_f:
        with open(image_out_path, "ab") as image_f:
            for c in tqdm(range(image.shape[0]), position=0):
                zs = np.full(num_points, c, dtype=np.float32) * scaling
                zs = np.frombuffer(zs.tobytes(), dtype=np.uint8).reshape((-1, 4))
                intensities = image[c, :, :].ravel()
                mask_channel = mask[c, :, :].ravel()
                merged_array = np.ascontiguousarray(
                    np.concatenate((
                        (xs[mask_channel, ...] * stride).astype(np.float32),
                        (ys[mask_channel, ...] * stride).astype(np.float32),
                        (zs[mask_channel, ...] * stride).astype(np.float32),
                        intensities[:, None][mask_channel, ...].astype(np.uint8)
                    ), axis=1)
                )
                n_written_points += merged_array.shape[0]
                ba = merged_array.tobytes()
                image_f.write(ba)

                label_intensities = label[c, :, :].ravel()
                mask_channel = mask[c, :, :].ravel() & (label_intensities > 0)
                label_merged_array = np.ascontiguousarray(
                    np.concatenate((
                        (xs[mask_channel, ...] * stride).astype(np.float32),
                        (ys[mask_channel, ...] * stride).astype(np.float32),
                        (zs[mask_channel, ...] * stride).astype(np.float32),
                        np.full((mask_channel.sum(), 1), 255, dtype=np.uint8)
                    ), axis=1)
                )
                label_ba = label_merged_array.tobytes()
                label_f.write(label_ba)
    print(f"promised: {n_cloud_points} wrote {n_written_points} points")


if __name__ == "__main__":
    main()
