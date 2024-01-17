from src.sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from tqdm import tqdm
from line_profiler_pycharm import profile
import numpy as np
import argparse
import cc3d
import randomcolor
import torch


@torch.no_grad()
@profile
def main():
    rand_color = randomcolor.RandomColor()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--stride", type=int, required=False, default=1)
    parser.add_argument("--dir-name", type=str, required=False, default=None)
    args, _ = parser.parse_known_args()
    data_dir = Path(args.path)
    stride = args.stride
    dir_name = args.dir_name

    if not dir_name:
        dir_name = "thresholded_prob"

    chunk_dirs = sorted([d / dir_name for d in data_dir.glob("chunk_*") if d.is_dir()])
    pred = np.ascontiguousarray(read_mmap_array(chunk_dirs[0]).data[::stride, ::stride, ::stride])
    # pred = np.ascontiguousarray(read_mmap_array(chunk_dirs[0]).data[0:300, ::stride, ::stride])
    total_points = pred.sum()

    ply_out_path = data_dir / f"pred_colored_{stride}_{dir_name}.ply"
    Path(ply_out_path).write_text(
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float red\n"
        "property float green\n"
        "property float blue\n"
        "end_header\n"
    )

    scaling = 1e-3
    # xs, ys = np.meshgrid(
    #     np.arange(pred.shape[1]),
    #     np.arange(pred.shape[2]),
    #     indexing="ij",
    # )
    # xs = xs.astype(np.single).ravel() * scaling
    # ys = ys.astype(np.single).ravel() * scaling
    # total_channels = pred.shape[0]
    # num_points = xs.shape[0]
    # print(f"xs: {np.min(xs)}: {np.max(xs)}")
    # print(f"ys: {np.min(ys)}: {np.max(ys)}")
    # print(f"zs: 0: {total_channels * scaling}")
    # xs = np.frombuffer(xs.tobytes(), dtype=np.uint8).reshape((-1, 4))
    # ys = np.frombuffer(ys.tobytes(), dtype=np.uint8).reshape((-1, 4))
    # assert xs.shape[0] == num_points, f"{xs.shape[0]=} != {num_points=}"
    # n_written_points = 0

    labels_out, n_labels = cc3d.connected_components(pred, return_N=True)
    device = "cuda"
    image_tensor = torch.zeros(pred.shape, dtype=torch.bool, device=device)
    with open(ply_out_path, "ab") as pred_f:
        for label, image in tqdm(cc3d.each(labels_out, binary=True, in_place=True), total=n_labels):
            color_r, color_g, color_b = [int(x) for x in rand_color.generate(format_="rgb")[0].replace("rgb(", "").replace(")", "").split(", ")]
            # image_tensor[:] = image
            image_tensor.copy_(torch.from_numpy(image).to(device), non_blocking=True)

            zs = torch.zeros((0, ), device=device, dtype=torch.int)
            ys = torch.zeros((0, ), device=device, dtype=torch.int)
            xs = torch.zeros((0, ), device=device, dtype=torch.int)

            for c in range(image_tensor.shape[0]):
                z = c
                slice_x, slice_y = torch.nonzero(image_tensor[c, ...], as_tuple=True)
                xs = torch.cat([xs, slice_x])
                ys = torch.cat([ys, slice_y])
                zs = torch.cat([zs, torch.full(slice_x.shape, z, device=device, dtype=torch.int)])

            xs_arr = xs.cpu().numpy()
            ys_arr = ys.cpu().numpy()
            zs_arr = zs.cpu().numpy()
            merged_array = np.ascontiguousarray(
                np.stack((
                    xs_arr * stride * scaling,
                    ys_arr * stride * scaling,
                    zs_arr * stride * scaling,
                    np.full(xs.shape, color_r / 255.0),
                    np.full(xs.shape, color_g / 255.0),
                    np.full(xs.shape, color_b / 255.0),
                ), axis=1)
            ).astype(np.float32)
            ba = merged_array.tobytes()
            pred_f.write(ba)
    print("all done")


if __name__ == "__main__":
    main()
