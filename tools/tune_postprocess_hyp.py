from sennet.custom_modules.metrics.surface_dice_metric_fast import (
    compute_surface_dice_score_from_thresholded_mmap,
)
from sennet.core.post_processings import filter_out_small_blobs
from sennet.environments.constants import PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
import numpy as np
import argparse
import optuna
import json


def objective(
        trial: optuna.Trial,
        chunk_dirs: List[Union[str, Path]],
        labels: dict[str, np.ndarray],
        percentile_range: np.ndarray,
        cached_percentiles: dict[str, np.ndarray],
        percentile_low: float,
        percentile_high: float,
) -> float:
    percentile_threshold: float | None = trial.suggest_float("percentile_threshold", percentile_low, percentile_high)
    # do_dust: bool = trial.suggest_categorical("do_dust", [False, True])
    do_dust: bool = False
    if do_dust:
        dust_threshold: int = trial.suggest_int("dust_threshold", 1, 10000000, log=True)
    else:
        dust_threshold = 0

    chunk_dirs = [Path(d) for d in chunk_dirs]
    mean_dice = 0.0
    dice_scores = {}
    threshold = None
    for folder in chunk_dirs:
        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])
        assert len(chunk_dir_names) == 1, f"many chunks is now deprecated, got: {len(chunk_dir_names)}"
        mean_prob_chunks = [np.ascontiguousarray(read_mmap_array(folder / cd / "mean_prob", mode="r").data) for cd in chunk_dir_names]

        threshold = np.interp(percentile_threshold, percentile_range, cached_percentiles[folder.name])
        # threshold = np.percentile(mean_prob_chunks[0], percentile_threshold)
        if threshold < 1e-5:
            print(f"{threshold=}: pruned")
            return 0.0
        # print(f"thresholds: {np.percentile(mean_prob_chunks[0].data, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])}")
        thresholded_pred = mean_prob_chunks[0] > threshold
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
    print(f"{percentile_threshold=}, threshold={float(threshold):.5f}")
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

    percentile_low = 90.0
    percentile_high = 100.0
    percentile_range = np.linspace(percentile_low, percentile_high, num=1000)
    cached_percentiles = {}
    for folder in tqdm(chunk_dirs):
        folder = Path(folder)
        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])
        assert len(chunk_dir_names) == 1, f"many chunks is now deprecated, got: {len(chunk_dir_names)}"
        mean_prob_chunks = [np.ascontiguousarray(read_mmap_array(folder / cd / "mean_prob", mode="r").data) for cd in chunk_dir_names]
        cached_percentiles[folder.name] = np.percentile(mean_prob_chunks[0], percentile_range)
        print(f"cached percentiles for {folder.name}: {cached_percentiles[folder.name].min()} -> {cached_percentiles[folder.name].max()}")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(
        t,
        chunk_dirs,
        labels,
        percentile_range=percentile_range,
        cached_percentiles=cached_percentiles,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    ), n_trials=10000)


if __name__ == "__main__":
    main()
