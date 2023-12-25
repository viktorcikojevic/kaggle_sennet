from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.environments.constants import DATA_DIR
import pandas as pd


def main():
    # path = "/home/clay/research/kaggle/sennet/data_dumps/predicted/kidney_1_dense/submission.csv"
    # path = "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_3_dense/submission.csv"
    path = "/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_1_dense/submission.csv"
    print(f"{path = }")
    df = pd.read_csv(path)
    label = pd.read_csv(DATA_DIR / "train_rles.csv")
    # label = pd.read_csv("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_1_dense/rle.csv")
    filtered_label = label.loc[label["id"].isin(df["id"])].copy().sort_values("id").reset_index()
    filtered_label["width"] = df["width"]
    filtered_label["height"] = df["height"]
    score = compute_surface_dice_score(
        submit=df,
        label=filtered_label,
    )
    print(f"{score = }")


if __name__ == "__main__":
    main()
