from sennet.core.submission_utils import generate_submission_df_from_one_chunked_inference
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--folder-name", type=str, required=False, default="thresholded_prob")
    parser.add_argument("--out-dir", type=str, required=False)
    args, _ = parser.parse_known_args()
    path = Path(args.path)
    folder_name = args.folder_name
    out_dir = None if args.out_path is None else Path(args.out_dir)

    root_dirs = list(path.glob("*"))
    pred_paths = []
    print(f"{root_dirs = }")
    for root_dir in tqdm(root_dirs):
        root_sub_df = generate_submission_df_from_one_chunked_inference(root_dir, folder_name)
        pred_path = root_dir / "submission.csv"
        root_sub_df.to_csv(pred_path)
        pred_paths.append(pred_path)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"catting everything into: {out_dir}")
        final_sub_path = out_dir / "submission.csv"
        dfs = []
        for p in pred_paths:
            print(p)
            df = pd.read_csv(p)[["id", "rle"]]
            dfs.append(df)
        df = pd.concat(dfs, axis=0).set_index("id").sort_index()
        df.to_csv(final_sub_path)
    print("done!")


if __name__ == "__main__":
    main()
