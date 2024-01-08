from sennet.core.submission_utils import (
    load_config_from_dir,
    load_model_from_dir,
    build_data_loader,
    generate_submission_df_from_one_chunked_inference,
)
from sennet.core.submission_simple import generate_submission_df, ParallelizationSettings
from sennet.core.mmap_arrays import read_mmap_array, create_mmap_array
from sennet.core.tta_model import Tta3DSegmentor
from sennet.core.post_processings import filter_out_small_blobs
from sennet.environments.constants import PROCESSED_DATA_DIR, CONFIG_DIR, MODEL_OUT_DIR
from typing import List, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
from line_profiler_pycharm import profile
import torch
import yaml
import json


class EnsembledPredictions:
    def __init__(self, root_dir: str | Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(exist_ok=True, parents=True)
        self.mean_prob = None
        self.thresholded_prob = None
        self.num_model_count = 0

    def add_chunked_predictions(self, pred_root_path: str | Path):
        pred_root_path = Path(pred_root_path)
        chunk_dirs = sorted([c for c in pred_root_path.glob("chunk*") if c.is_dir()])
        if self.mean_prob is None:
            shape = [0, 0, 0]
            for cd in tqdm(chunk_dirs, desc="init ens mmap"):
                chunk_pred = read_mmap_array(cd / "mean_prob", mode="r").data
                shape[0] += chunk_pred.shape[0]
                shape[1] = chunk_pred.shape[1]
                shape[2] = chunk_pred.shape[2]
            self.mean_prob = create_mmap_array(self.root_dir / "chunk_00" / "mean_prob", shape, float)
            self.thresholded_prob = create_mmap_array(self.root_dir / "chunk_00" / "thresholded_prob", shape, bool)
        if not (self.root_dir / "image_names").is_file():
            (self.root_dir / "image_names").write_text((pred_root_path / "image_names").read_text())
        i = 0
        for cd in tqdm(chunk_dirs, desc="add pred to ens", position=0):
            chunk_pred = read_mmap_array(cd / "mean_prob", mode="r").data
            i_end = i + chunk_pred.shape[0]
            self.mean_prob.data[i: i_end, ...] += chunk_pred
            i = i_end
        self.num_model_count += 1

    def finalise(
            self,
            threshold: float,
            dust_threshold: int = 1000,
            filter_small_blobs: bool = True,
    ):
        self.mean_prob.data[:] /= self.num_model_count
        self.thresholded_prob.data[:] = self.mean_prob.data > threshold
        self.mean_prob.data.flush()
        ensembled_df = generate_submission_df_from_one_chunked_inference(self.root_dir)
        ensembled_df.to_csv(self.root_dir / "submission.csv")

        if filter_small_blobs:
            filtered_dir = self.root_dir.parent.parent / f"{self.root_dir.parent.name}_cc3d" / self.root_dir.name
            filtered_dir.mkdir(exist_ok=True, parents=True)
            (filtered_dir / "image_names").write_text((self.root_dir / "image_names").read_text())
            filter_out_small_blobs(
                thresholded_pred=self.thresholded_prob.data,
                out_path=filtered_dir / "chunk_00" / "thresholded_prob",
                dust_threshold=dust_threshold,
                connectivity=26,
            )
            ensembled_df = generate_submission_df_from_one_chunked_inference(filtered_dir)
            ensembled_df.to_csv(filtered_dir / "submission.csv")


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
    parser.add_argument("--keep-model-chunks", required=False, action="store_true", default=False)
    parser.add_argument("--n-chunks", required=False, type=int, default=None)  # actually deprecated
    parser.add_argument("--run-as-single-process", required=False, action="store_true", default=False)
    parser.add_argument("--no-cc3d", required=False, action="store_true", default=False)
    parser.add_argument("--batch-size", required=False, type=int, default=1)

    submission_cfg_path = CONFIG_DIR / "submission.yaml"
    with open(submission_cfg_path, "rb") as f:
        submission_cfg = yaml.load(f, yaml.FullLoader)

    args, _ = parser.parse_known_args()
    out_dir = Path(args.out_dir)
    run_all = args.run_all
    run_as_single_process = args.run_as_single_process
    keep_model_chunks = args.keep_model_chunks
    no_cc3d = args.no_cc3d
    batch_size = args.batch_size

    if submission_cfg["predictors"]["dust_threshold"] is None:
        print(f"dust_threshold is None, turning off cc3d")
        no_cc3d = True
    if run_as_single_process:
        print(f"{run_as_single_process=}: removing all multi processing")
    if run_all:
        folders_override = sorted([d.relative_to(PROCESSED_DATA_DIR).name for d in PROCESSED_DATA_DIR.glob("*") if d.is_dir()])
        print(f"run_all given, folders overridden to: {folders_override}")
    elif args.folders is not None:
        folders_override = None if args.folders is None else args.folders.split(",")
        print(f"folders given, folders overridden to: {folders_override}")
    else:
        folders_override = None
        print(f"folders not given, running on each model's val set")

    folders_to_models = {}
    for model_name in submission_cfg["predictors"]["models"]:
        model_dir = MODEL_OUT_DIR / model_name
        cfg = load_config_from_dir(model_dir)
        if folders_override is None:
            folders = cfg["val_folders"]
            print(f"[{model_name}]: folder override not given, running on model's original val: {folders}")
        else:
            folders = folders_override
        for folder in folders:
            if folder not in folders_to_models:
                folders_to_models[folder] = []
            folders_to_models[folder].append(model_name)
    print(json.dumps(folders_to_models, indent=4))

    out_dir.mkdir(exist_ok=True, parents=True)
    ensembled_dir = out_dir / "ensembled"
    ensembled_dir.mkdir(exist_ok=True, parents=True)
    for i, (folder, model_names) in enumerate(folders_to_models.items()):
        ensembled_prediction = EnsembledPredictions(
            root_dir=ensembled_dir / folder,
        )

        print(f"[{i}/{len(folders_to_models)}]: {folder} with {len(model_names)} models")
        for model_name in model_names:
            model_dir = MODEL_OUT_DIR / model_name
            cfg, base_model = load_model_from_dir(model_dir)
            model = Tta3DSegmentor(base_model, **submission_cfg["predictors"]["tta_kwargs"])

            if keep_model_chunks:
                data_out_dir = out_dir / model_name / folder
            else:
                data_out_dir = out_dir / "tmp_model_outs" / folder
            data_out_dir.mkdir(exist_ok=True, parents=True)
            print(f"> {model_name}: {folder} -> {data_out_dir}")

            data_loader = build_data_loader(
                folder,
                substride=submission_cfg["predictors"]["substride"],
                cfg=cfg,
                cropping_border=submission_cfg["predictors"]["cropping_border"],
                batch_size=batch_size,
                num_workers=0,
            )
            generate_submission_df(
                model=model,
                data_loader=data_loader,
                threshold=submission_cfg["predictors"]["threshold"],
                percentile_threshold=submission_cfg["predictors"].get("percentile_threshold", None),
                parallelization_settings=ParallelizationSettings(
                    run_as_single_process=run_as_single_process,
                ),
                out_dir=data_out_dir,
                device="cuda",
                save_sub=True,
            )

            ensembled_prediction.add_chunked_predictions(data_out_dir)
        ensembled_prediction.finalise(
            threshold=submission_cfg["predictors"]["threshold"],
            dust_threshold=submission_cfg["predictors"]["dust_threshold"],
            filter_small_blobs=not no_cc3d,
        )

    print("aggregating preds into its final dir")
    final_sub_path = out_dir / "submission.csv"
    out_dir_name = "ensembled" if no_cc3d else "ensembled_cc3d"
    pred_files = list((Path(out_dir) / out_dir_name).rglob("submission.csv"))
    dfs = []
    for p in pred_files:
        print(p)
        df = pd.read_csv(p)[["id", "rle"]]
        dfs.append(df)
    df = pd.concat(dfs, axis=0).set_index("id").sort_index()
    df.to_csv(final_sub_path)
    print("done!")


if __name__ == "__main__":
    main()
