# from sennet.custom_modules.metrics.surface_dice_metric import compute_surface_dice_score
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.environments.constants import DATA_DIR
import pandas as pd


def main():
    df = pd.read_csv("/home/clay/research/kaggle/sennet/data_dumps/predicted/kidney_1_dense/submission.csv")
    # df = pd.read_csv("/home/clay/research/kaggle/sennet/data_dumps/tmp_mmaps/UNet3D-2023-12-16-17-54-53/submission.csv")
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
