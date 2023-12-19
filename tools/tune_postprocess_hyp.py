from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.environments.constants import DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from sennet.core.rles import rle_encode
import pandas as pd
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
import argparse
import optuna
import json


def objective(
        trial: optuna.Trial,
        chunk_dirs: List[Union[str, Path]],
        label: pd.DataFrame,
) -> float:
    threshold: float = trial.suggest_float("threshold", 0.01, 0.9)

    chunk_dirs = [Path(d) for d in chunk_dirs]
    data = {"id": [], "rle": [], "height": [], "width": []}
    for folder in chunk_dirs:
        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])
        # out_dir = POST_PROCESS_HYP_WORK_DIR / folder
        # (out_dir / "image_names").write_text()
        image_names = (folder / "image_names").read_text().split("\n")

        i = 0
        for cd in tqdm(chunk_dir_names, position=0):
            chunk_pred = read_mmap_array(folder / cd / "mean_prob", mode="r")
            for c in range(chunk_pred.shape[0]):
                thresholded_channel = chunk_pred.data[c, :, :] > threshold
                rle = rle_encode(thresholded_channel)
                if rle == "":
                    rle = "1 0"
                image_name = image_names[i]
                i += 1
                data["id"].append(image_name)
                data["rle"].append(rle)
                data["height"].append(int(thresholded_channel.shape[0]))
                data["width"].append(int(thresholded_channel.shape[1]))

    df = pd.DataFrame(data).sort_values("id")
    filtered_label = label.loc[label["id"].isin(df["id"])].copy().sort_values("id").reset_index()
    filtered_label["width"] = df["width"]
    filtered_label["height"] = df["height"]
    score = compute_surface_dice_score(
        submit=df,
        label=filtered_label,
    )
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    # parser.add_argument("--data-root", required=True)
    args, _ = parser.parse_known_args()
    pred_dir = Path(args.pred_dir)
    # data_root = Path(args.data_root)

    chunk_dirs = sorted([str(p) for p in pred_dir.glob("*")])
    print(f"optimising post process hyp for: {chunk_dirs}")
    print(json.dumps(chunk_dirs, indent=4))

    label = pd.read_csv(DATA_DIR / "train_rles.csv")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, chunk_dirs, label), n_trials=10000)


if __name__ == "__main__":
    main()
