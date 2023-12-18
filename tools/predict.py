from sennet.core.submission_utils import load_model_from_dir, build_data_loader
from sennet.core.submission import generate_submission_df, ParallelizationSettings
from sennet.environments.constants import PROCESSED_DATA_DIR
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--n-chunks", required=False, type=int, default=5)
    parser.add_argument("--threshold", required=False, type=float, default=0.5)
    parser.add_argument("--substride", required=False, type=float, default=1.0)
    parser.add_argument("--run-all", required=False, action="store_true")
    parser.add_argument("--folder", required=False, type=str)

    args, _ = parser.parse_known_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    args_folder = args.folder
    substride = args.substride
    threshold = args.threshold
    n_chunks = args.n_chunks
    run_all = args.run_all

    if run_all:
        folders = sorted([d for d in PROCESSED_DATA_DIR.glob("*") if d.is_dir()])
    else:
        folders = [args_folder]

    print("running prediction on:")
    for folder in folders:
        print(folder)

    cfg, model = load_model_from_dir(model_dir)
    for folder in folders:
        data_out_dir = out_dir / folder
        print(f"working on {folder} -> {data_out_dir}")

        data_loader = build_data_loader(
            folder,
            substride,
            cfg,
        )
        generate_submission_df(
            model=model,
            data_loader=data_loader,
            threshold=threshold,
            parallelization_settings=ParallelizationSettings(
                run_as_single_process=False,
                n_chunks=n_chunks,
                finalise_one_by_one=True,
            ),
            out_dir=data_out_dir,
            device="cuda",
            save_sub=True,
        )
    # TODO(Sumo): something around aggregating csvs together here


if __name__ == "__main__":
    main()
