from sennet.environments.constants import PROCESSED_DATA_DIR
from pathlib import Path
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args, _ = parser.parse_known_args()
    path = Path(args.path)

    rle_df = pd.read_csv(path)
    processed_data_paths = list(sorted(PROCESSED_DATA_DIR.glob("*")))
    
    

    total_written_rows = 0
    for p in processed_data_paths:
        df = rle_df[rle_df["id"].str.startswith(p.name)].copy().reset_index(drop=True)
        # Get height and width from p/image/metadata.json
        metadata = pd.read_json(p / "image" / "metadata.json")
        df["height"] = metadata["shape"][1]
        df["width"] = metadata["shape"][2]
        df.to_csv(p / "rle.csv")
        total_written_rows += len(df)
    print(f"wrote: {total_written_rows}/{len(rle_df)} rows")


if __name__ == "__main__":
    main()
