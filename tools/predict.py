from sennet.core.submission_utils import (
    load_config_from_dir,
    load_model_from_dir,
    build_data_loader,
    generate_submission_df_from_memory,
)
from sennet.core.submission_simple import generate_submission_df, ParallelizationSettings
from sennet.core.tta_model import Tta3DSegmentor
from sennet.core.post_processings import largest_k_closest_to_center
from sennet.environments.constants import PROCESSED_DATA_DIR, CONFIG_DIR, MODEL_OUT_DIR
from pathlib import Path
import pandas as pd
import argparse
import yaml
import json


def create_model_and_data_factory(
        model_name: str,
        submission_cfg: dict,
        folder: str,
        cache_mmaps: bool = False,
        batch_size: int = 1,
        fast_mode: bool = False
):
    def factory():
        model_dir = MODEL_OUT_DIR / model_name
        cfg, base_model = load_model_from_dir(model_dir)
        model = Tta3DSegmentor(base_model, **submission_cfg["predictors"]["tta_kwargs"])

        data_loader = build_data_loader(
            folder,
            substride=submission_cfg["predictors"]["substride"],
            cfg=cfg,
            cache_mmaps=cache_mmaps,
            cropping_border=submission_cfg["predictors"]["cropping_border"],
            batch_size=batch_size,
            num_workers=0,
            fast_mode=fast_mode,
        )
        return model, data_loader
    return factory


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
    parser.add_argument("--cache-mmaps", required=False, action="store_true", default=False)
    parser.add_argument("--fast-mode", required=False, action="store_true", default=False)

    submission_cfg_path = CONFIG_DIR / "submission.yaml"
    with open(submission_cfg_path, "rb") as f:
        submission_cfg = yaml.load(f, yaml.FullLoader)

    args, _ = parser.parse_known_args()
    out_dir = Path(args.out_dir)
    run_all = args.run_all
    run_as_single_process = args.run_as_single_process
    assert not args.keep_model_chunks, f"deprecated"
    no_cc3d = args.no_cc3d
    batch_size = args.batch_size
    cache_mmaps = args.cache_mmaps
    fast_mode = args.fast_mode
    if fast_mode:
        print(f"WARNING: {fast_mode=}, this shouldn't be turned on in prod")

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

    for i, (folder, model_names) in enumerate(folders_to_models.items()):
        factories = [create_model_and_data_factory(
            model_name=model_name,
            submission_cfg=submission_cfg,
            folder=folder,
            cache_mmaps=cache_mmaps,
            batch_size=batch_size,
            fast_mode=fast_mode,
        ) for model_name in model_names]

        data_out_dir = out_dir / "ensembled" / folder
        res = generate_submission_df(
            model=None,
            data_loader=None,
            threshold=submission_cfg["predictors"]["threshold"],
            percentile_threshold=submission_cfg["predictors"].get("percentile_threshold", None),
            parallelization_settings=ParallelizationSettings(
                run_as_single_process=run_as_single_process,
            ),
            out_dir=data_out_dir,
            device="cuda",
            save_sub=False,
            keep_in_memory=cache_mmaps,
            model_and_data_loader_factory=factories,
        )
        post_process_kwargs = submission_cfg.get("post_processing", None)
        image_names = (data_out_dir / "image_names").read_text().split("\n")
        if not no_cc3d and post_process_kwargs is not None:
            largest_k_closest_to_center(
                thresholded_pred=res.thresholded_pred,
                out=res.thresholded_pred,
                **post_process_kwargs,
            )
        ensembled_df = generate_submission_df_from_memory(
            res.thresholded_pred,
            image_names=image_names,
        )
        ensembled_df.to_csv(data_out_dir / "submission.csv")

    final_sub_path = out_dir / "submission.csv"
    print(f"aggregating preds into its final dir: {final_sub_path}")
    pred_files = list((Path(out_dir) / "ensembled").rglob("submission.csv"))
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
