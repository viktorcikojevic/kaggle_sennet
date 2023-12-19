from src.sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from tqdm import tqdm
from line_profiler_pycharm import profile
import numpy as np
import argparse


@profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    args, _ = parser.parse_known_args()
    data_dir = Path(args.path)

    stride = 1
    chunk_dirs = sorted([d / "thresholded_prob" for d in data_dir.glob("chunk_*") if d.is_dir()])
    pred_mmaps = [read_mmap_array(d).data[::stride, ::stride, ::stride] for d in chunk_dirs]
    total_points = 0
    for pm in pred_mmaps:
        total_points += pm.sum()

    ply_out_path = data_dir / "pred.ply"
    Path(ply_out_path).write_text(
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar intensity\n"
        "end_header\n"
    )

    scaling = 1e-3
    xs, ys = np.meshgrid(
        np.arange(pred_mmaps[0].shape[1]),
        np.arange(pred_mmaps[0].shape[2]),
        indexing="ij",
    )
    xs = xs.astype(np.single).ravel() * scaling
    ys = ys.astype(np.single).ravel() * scaling
    total_channels = sum([pm.shape[0] for pm in pred_mmaps])
    num_points = xs.shape[0]
    print(f"xs: {np.min(xs)}: {np.max(xs)}")
    print(f"ys: {np.min(ys)}: {np.max(ys)}")
    print(f"zs: 0: {total_channels * scaling}")
    xs = np.frombuffer(xs.tobytes(), dtype=np.uint8).reshape((-1, 4))
    ys = np.frombuffer(ys.tobytes(), dtype=np.uint8).reshape((-1, 4))
    assert xs.shape[0] == num_points, f"{xs.shape[0]=} != {num_points=}"
    n_written_points = 0
    with open(ply_out_path, "ab") as pred_f:
        c = 0
        for pm in tqdm(pred_mmaps, position=0):
            for i in tqdm(range(pm.shape[0]), position=1, leave=False):
                zs = np.full(num_points, c, dtype=np.single) * scaling
                zs = np.frombuffer(zs.tobytes(), dtype=np.uint8).reshape((-1, 4))
                intensities = (pm[i, :, :] * 127).astype(np.uint8).ravel()
                mask_channel = (pm[i, :, :] > 0).ravel()
                merged_array = np.ascontiguousarray(
                    np.concatenate((
                        xs[mask_channel, ...] * stride,
                        ys[mask_channel, ...] * stride,
                        zs[mask_channel, ...] * stride,
                        intensities[:, None][mask_channel, ...]
                    ), axis=1)
                )
                n_written_points += merged_array.shape[0]
                ba = merged_array.tobytes()
                pred_f.write(ba)
                c += 1
    print(f"promised: {total_points} wrote {n_written_points} points")


if __name__ == "__main__":
    main()
