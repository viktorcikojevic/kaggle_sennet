import cv2
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.core.mmap_arrays import read_mmap_array
from sennet.environments.constants import DATA_DIR, PROCESSED_DATA_DIR, DATA_DUMPS_DIR
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from line_profiler_pycharm import profile


@profile
def main():
    label = pd.read_csv(DATA_DIR / "train_rles.csv")

    paths = [
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_1_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_1_dense/submission.csv",
        "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_3_dense/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_2/submission.csv",
        # "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled_cc3d/kidney_2/submission.csv",
    ]

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
        )
        print(f"{path}: {score = }")

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
            save_img = cv2.addWeighted(channel, 0.5, rendered_vis[c, ...], 0.5, 0.0)
            cv2.imwrite(str(out_dir / f"{str(c).zfill(3)}.png"), save_img)
        print("done")


if __name__ == "__main__":
    main()
