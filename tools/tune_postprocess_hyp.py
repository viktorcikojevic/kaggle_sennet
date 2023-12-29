from sennet.custom_modules.metrics.surface_dice_metric_fast import (
    compute_surface_dice_score_from_thresholded_mmap,
)
from sennet.core.post_processings import filter_out_small_blobs
from sennet.environments.constants import PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from typing import List, Union
import numpy as np
import argparse
import optuna
import json


def objective(
        trial: optuna.Trial,
        chunk_dirs: List[Union[str, Path]],
        labels: dict[str, np.ndarray],
) -> float:
    threshold: float = trial.suggest_float("threshold", 0.01, 0.2)
    dust_threshold: int = trial.suggest_int("dust_threshold", 1, 10000000, log=True)
    do_dust: bool = trial.suggest_categorical("do_dust", [False, True])

    chunk_dirs = [Path(d) for d in chunk_dirs]
    mean_dice = 0.0
    dice_scores = {}
    for folder in chunk_dirs:
        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])

        mean_prob_chunks = [read_mmap_array(folder / cd / "mean_prob", mode="r").data for cd in chunk_dir_names]
        thresholded_pred = np.concatenate([
            m > threshold
            for m in mean_prob_chunks
        ], axis=0)
        if do_dust:
            filtered_pred = filter_out_small_blobs(
                thresholded_pred,
                out_path=None,
                dust_threshold=dust_threshold,
                connectivity=26,
            )
            dice_score = compute_surface_dice_score_from_thresholded_mmap(
                thresholded_chunks=[filtered_pred],
                label=labels[folder.name],
            )
        else:
            dice_score = compute_surface_dice_score_from_thresholded_mmap(
                thresholded_chunks=[thresholded_pred],
                label=labels[folder.name],
            )

        dice_scores[folder.name] = dice_score
        mean_dice += dice_score / len(chunk_dirs)
    print("---")
    print(json.dumps(dice_scores, indent=4))
    print("---")
    return mean_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    args, _ = parser.parse_known_args()
    pred_dir = Path(args.pred_dir)

    chunk_dirs = sorted([str(p) for p in pred_dir.glob("*")])

    labels = {
        p.name: read_mmap_array(p / "label", mode="r").data
        for p in PROCESSED_DATA_DIR.glob("*")
        if p.is_dir() and (p / "label").is_dir()
    }
    print(f"chunk dirs")
    print(json.dumps(chunk_dirs, indent=4))
    print("found labels:")
    print(json.dumps({k: v.shape for k, v in labels.items()}))

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, chunk_dirs, labels), n_trials=10000)


if __name__ == "__main__":
    main()
