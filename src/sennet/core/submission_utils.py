from sennet.core.mmap_arrays import read_mmap_array
from sennet.core.rles import rle_encode
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def generate_submission_df_from_one_chunked_inference(
        root_dir: Path,
) -> pd.DataFrame:
    image_names = (root_dir / "image_names").read_text().split("\n")
    chunk_dirs = sorted(list(root_dir.glob("chunk*")))
    i = 0
    data = {"id": [], "rle": []}
    for d in tqdm(chunk_dirs, position=0):
        pred = read_mmap_array(d / "thresholded_prob", mode="r")
        for c in tqdm(range(pred.shape[0]), position=1, leave=False):
            rle = rle_encode(pred.data[c, :, :])
            image_name = image_names[i]
            i += 1
            data["id"].append(image_name)
            data["rle"].append(rle)
    df = pd.DataFrame(data)
    df["rle"] = df["rle"].fillna("1 0")
    return df


if __name__ == "__main__":
    generate_submission_df_from_one_chunked_inference(Path("/home/clay/research/kaggle/sennet/data_dumps/tmp_mmaps/sennet_tmp_2023-12-15-23-01-31"))
