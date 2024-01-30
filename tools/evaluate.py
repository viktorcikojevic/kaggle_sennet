import cv2
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score, create_table_neighbour_code_to_surface_area
from sennet.core.mmap_arrays import read_mmap_array
from sennet.environments.constants import DATA_DIR, PROCESSED_DATA_DIR, DATA_DUMPS_DIR
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from line_profiler_pycharm import profile


@profile
def main():
    labels = []
    for p in PROCESSED_DATA_DIR.rglob("rle.csv"):
        print(f"building label: {p}")
        label = pd.read_csv(p)
        labels.append(label)
    label = pd.concat(labels)

    paths = [
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_1_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_1_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_sparse/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_3_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_merged/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_2/submission.csv",
        "/opt/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_2/submission.csv",
    ]
    vis = True
    # vis = True
    return_colored_cloud = True

    for path in paths:
        path = Path(path)
        if not path.is_file():
            print(f"{path}: not found")
            continue
        df = pd.read_csv(path)
        # label = pd.read_csv("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_1_dense/rle.csv")
        filtered_label = label.loc[label["id"].isin(df["id"])].copy().sort_values("id").reset_index()
        filtered_label["width"] = df["width"]
        filtered_label["height"] = df["height"]

        score = compute_surface_dice_score(
            submit=df,
            label=filtered_label,
            return_colored_cloud=return_colored_cloud,
        )
        if return_colored_cloud:
            score, cloud = score
        else:
            cloud = None
        print(f"{path}: {score = }")
        if return_colored_cloud:
            cloud = cloud.cpu().numpy()
            out_cloud = np.zeros((cloud.shape[0], 6), dtype=np.float32)
            ply_out_path = path.parent / f"eval.ply"
            Path(ply_out_path).write_text(
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {out_cloud.shape[0]}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float red\n"
                "property float green\n"
                "property float blue\n"
                "end_header\n"
            )
            scaling = 1e-3
            area_denominator = create_table_neighbour_code_to_surface_area((1, 1, 1)).max()
            out_cloud[:, :3] = cloud[:, :3] * scaling
            out_cloud[cloud[:, 4] == 0, 3:] = cloud[cloud[:, 4] == 0, 3][:, None] / area_denominator * np.array([0.0, 1.0, 0.0])[None, :]  # tp
            out_cloud[cloud[:, 4] == 1, 3:] = cloud[cloud[:, 4] == 1, 3][:, None] / area_denominator * np.array([1.0, 0.0, 0.0])[None, :]  # fp
            out_cloud[cloud[:, 4] == 2, 3:] = cloud[cloud[:, 4] == 2, 3][:, None] / area_denominator * np.array([0.5, 0.5, 1.0])[None, :]  # fn
            with open(ply_out_path, "ab") as pred_f:
                ba = out_cloud.tobytes()
                pred_f.write(ba)

    if vis:
        for path in paths:
            path = Path(path)
            if not path.is_file():
                print(f"{path}: not found")
                continue
            dir_name = path.parent.name
            pred = read_mmap_array(path.parent / "chunk_00" / "thresholded_prob")
            image = read_mmap_array(PROCESSED_DATA_DIR / dir_name / "image")
            label = read_mmap_array(PROCESSED_DATA_DIR / dir_name / "label")
            out_dir = DATA_DUMPS_DIR / "evaluated" / dir_name
            out_dir.mkdir(exist_ok=True, parents=True)

            rendered_vis = np.zeros((pred.data.shape[0], pred.data.shape[1], pred.data.shape[2], 3), dtype=np.uint8)
            print("copying pred out")
            copied_pred = np.ascontiguousarray(pred.data.copy()).astype(bool)
            print("copying labels out")
            copied_label = np.ascontiguousarray(label.data.copy()).astype(bool)
            print("computing TP")
            rendered_vis[copied_pred & copied_label, :] = (0, 255, 0)  # green
            print("computing FP")
            rendered_vis[copied_pred & ~copied_label, :] = (0, 0, 255)  # red
            print("computing FN")
            rendered_vis[~copied_pred & copied_label, :] = (255, 0, 0)  # blue

            print("saving images")
            for c in tqdm(range(rendered_vis.shape[0])):
                channel = np.ascontiguousarray(image.data[c, ...].copy())
                channel = np.stack((channel, channel, channel), axis=2)
                save_img = cv2.addWeighted(channel, 1.0, rendered_vis[c, ...], 0.5, 0.0)
                cv2.imwrite(str(out_dir / f"{str(c).zfill(3)}.png"), save_img)
            print("done")


if __name__ == "__main__":
    main()
