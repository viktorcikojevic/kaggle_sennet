from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score_from_mmap
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
    threshold: float = trial.suggest_float("threshold", 0.01, 0.9)

    chunk_dirs = [Path(d) for d in chunk_dirs]
    mean_dice = 0.0
    for folder in chunk_dirs:
        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])

        mean_prob_chunks = [read_mmap_array(folder / cd / "mean_prob", mode="r").data for cd in chunk_dir_names]
        dice_score = compute_surface_dice_score_from_mmap(
            mean_prob_chunks=mean_prob_chunks,
            label=labels[folder.name],
            threshold=threshold,
        )
        mean_dice += dice_score / len(chunk_dirs)
    return mean_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    args, _ = parser.parse_known_args()
    pred_dir = Path(args.pred_dir)

    chunk_dirs = sorted([str(p) for p in pred_dir.glob("*")])
    print(f"optimising post process hyp for: {chunk_dirs}")
    print(json.dumps(chunk_dirs, indent=4))

    labels = {
        p.name: read_mmap_array(p / "label", mode="r").data
        for p in PROCESSED_DATA_DIR.glob("*")
        if p.is_dir() and (p / "label").is_dir()
    }
    print(json.dumps({k: v.shape for k, v in labels.items()}))

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, chunk_dirs, labels), n_trials=10000)


if __name__ == "__main__":
    main()
