from sennet.core.submission_utils import evaluate_chunked_inference_in_memory
from sennet.environments.constants import DATA_DUMPS_DIR, PROCESSED_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import json


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

    thresholds = (
        np.linspace(0.0001, 0.001, num=10).tolist()
        + np.linspace(0.001, 0.01, num=10).tolist()
        + np.linspace(0.01, 0.1, num=10).tolist()
        # + np.linspace(0.1, 1.0, num=10).tolist()
    )
    chunk_dirs = [Path(d) for d in chunk_dirs]

    for folder in tqdm(chunk_dirs):
        surface_dices = []
        f1_scores = []
        precisions = []
        recalls = []

        chunk_dir_names = sorted([c.name for c in folder.glob("chunk*") if c.is_dir()])
        assert len(chunk_dir_names) == 1, f"many chunks is now deprecated, got: {len(chunk_dir_names)}"
        mean_prob_chunks = [np.ascontiguousarray(read_mmap_array(folder / cd / "mean_prob", mode="r").data) for cd in chunk_dir_names]
        mean_prob_chunk = np.ascontiguousarray(mean_prob_chunks[0].data)
        label = np.ascontiguousarray(labels[folder.name].data)

        metrics = evaluate_chunked_inference_in_memory(
            mean_prob_chunk=mean_prob_chunk,
            label=label,
            thresholds=thresholds,
            device="cuda",
        )
        precisions += metrics.precisions
        recalls += metrics.recalls
        f1_scores += metrics.f1_scores
        surface_dices += metrics.surface_dices

        for item, name in (
                (surface_dices, "surface_dices"),
                (f1_scores, "f1_scores"),
                (precisions, "precisions"),
                (recalls, "recalls"),
        ):
            out_path = DATA_DUMPS_DIR / "plots" / f"{folder.name}_{name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(20, 10))
            plt.title(f"{folder.name}_{thresholds[np.argmax(item)]}")
            plt.plot(thresholds, item, label=name)
            plt.scatter(thresholds, item)
            for t, i in zip(thresholds, item):
                plt.annotate(f"{t:.5f}", (t, i))
            plt.legend()
            plt.savefig(out_path)


if __name__ == "__main__":
    main()
