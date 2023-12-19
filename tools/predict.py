from sennet.core.submission_utils import load_model_from_dir, build_data_loader, generate_submission_df_from_one_chunked_inference
from sennet.core.submission import generate_submission_df, ParallelizationSettings
from sennet.core.mmap_arrays import read_mmap_array, create_mmap_array
from sennet.environments.constants import PROCESSED_DATA_DIR, CONFIG_DIR, MODEL_OUT_DIR
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import json


def ensemble_predictions(
        pred_paths: List[Union[str, Path]],
        out_dir: Union[str, Path],
        threshold: float,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    pred_paths = [Path(p) for p in pred_paths]
    if len(pred_paths) == 0:
        return
    (out_dir / "image_names").write_text((pred_paths[0] / "image_names").read_text())
    chunk_dir_names = sorted([c.name for c in pred_paths[0].glob("chunk*") if c.is_dir()])
    for cd in tqdm(chunk_dir_names, position=0):
        chunk_out_dir = out_dir / cd
        chunk_out_dir.mkdir(exist_ok=True, parents=True)

        chunk_preds = [read_mmap_array(p / cd / "mean_prob", mode="r") for p in pred_paths]
        chunk_mean_prob = create_mmap_array(chunk_out_dir / "mean_prob", chunk_preds[0].shape, chunk_preds[0].dtype)
        chunk_thresholded_prob = create_mmap_array(chunk_out_dir / "thresholded_prob", chunk_preds[0].shape, bool)

        for cp in chunk_preds:
            chunk_mean_prob.data += (cp.data / len(chunk_preds))

        chunk_thresholded_prob.data[:] = chunk_mean_prob.data > threshold
        chunk_mean_prob.data.flush()
        chunk_thresholded_prob.data.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--folders", required=False, type=str)
    parser.add_argument("--run-all", required=False, action="store_true", default=False)

    submission_cfg_path = CONFIG_DIR / "submission.yaml"
    with open(submission_cfg_path, "rb") as f:
        submission_cfg = yaml.load(f, yaml.FullLoader)

    args, _ = parser.parse_known_args()
    out_dir = Path(args.out_dir)
    run_all = args.run_all

    if run_all:
        folders_override = sorted([d.relative_to(PROCESSED_DATA_DIR).name for d in PROCESSED_DATA_DIR.glob("*") if d.is_dir()])
        print(f"run_all given, folders overridden to: {folders_override}")
    elif args.folders is not None:
        folders_override = None if args.folders is None else args.folders.split(",")
        print(f"folders given, folders overridden to: {folders_override}")
    else:
        folders_override = None
        print(f"folders not given, running on each model's val set")

    folders_to_dirs = {}
    for model_name in submission_cfg["predictors"]["models"]:
        model_dir = MODEL_OUT_DIR / model_name
        cfg, model = load_model_from_dir(model_dir)

        if folders_override is None:
            folders = cfg["val_folders"]
            print(f"[{model_name}]: folder override not given, running on model's original val: {folders}")
        else:
            folders = folders_override

        for i, folder in enumerate(folders):
            data_out_dir = out_dir / model_name / folder
            data_out_dir.mkdir(exist_ok=True, parents=True)

            if folder not in folders_to_dirs:
                folders_to_dirs[folder] = []
            folders_to_dirs[folder].append(str(data_out_dir.absolute().resolve()))

            print(f"[{model_name}:{folder}]: [{i}/{len(folders)}] -> {data_out_dir}")
            data_loader = build_data_loader(folder, submission_cfg["predictors"]["substride"], cfg)
            generate_submission_df(
                model=model,
                data_loader=data_loader,
                threshold=submission_cfg["predictors"]["threshold"],
                parallelization_settings=ParallelizationSettings(
                    run_as_single_process=False,
                    n_chunks=submission_cfg["predictors"]["n_chunks"],
                    finalise_one_by_one=True,
                ),
                out_dir=data_out_dir,
                device="cuda",
                save_sub=True,
            )

    print(json.dumps(folders_to_dirs, indent=4))
    ensembled_dir = out_dir / "ensembled"
    ensembled_dir.mkdir(exist_ok=True, parents=True)
    print("ensembling predictions")
    for folder, pred_paths in folders_to_dirs.items():
        ensembled_out_dir = ensembled_dir / folder
        ensemble_predictions(
            pred_paths,
            out_dir=ensembled_out_dir,
            threshold=submission_cfg["predictors"]["threshold"],
        )
        ensembled_df = generate_submission_df_from_one_chunked_inference(ensembled_out_dir)
        ensembled_df.to_csv(ensembled_out_dir / "submission.csv")
    print("done!")


if __name__ == "__main__":
    main()
