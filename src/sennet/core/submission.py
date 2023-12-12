from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.rles import rle_encode
from sennet.core.mmap_arrays import create_mmap_array
from typing import Optional, Union, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
from datetime import datetime
import shutil


class AppendingSubmissionCsv:
    def __init__(self, csv_path: Union[Path, str]):
        self.csv_path = csv_path
        self.file = open(csv_path, "a")
        self.file.write(f"id,rle")

    def __del__(self):
        self.file.flush()
        self.file.close()

    def add_line(self, items: List[str]):
        self.file.write(f"\n{','.join(items)}")


@profile
def generate_submission_df(
        model: nn.Module,
        data_loader: DataLoader,
        threshold: float,
        sub_out_dir: Union[str, Path],
        raw_pred_out_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
) -> None:
    assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
        f"to generate submission, dataset must be ThreeDSegmentationDataset"
    dataset: ThreeDSegmentationDataset = data_loader.dataset
    model = model.eval().to(device)

    current_total_prob = None
    current_total_count = None
    current_mean_prob = None
    thresholded_prob = None

    current_folder = None

    sub_file = AppendingSubmissionCsv(Path(sub_out_dir) / "submission.csv")
    write_to_tmp_file = raw_pred_out_dir is None
    if write_to_tmp_file:
        raw_pred_out_dir = Path(f"/tmp/sennet_tmp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            pred_batch = model(batch["img"].to(device))[:, 0, :, :, :]
            pred_batch = torch.nn.functional.sigmoid(pred_batch)
            pred_batch = pred_batch.cpu().numpy()
            for i in range(len(pred_batch)):
                folder = batch["folder"][i]
                pred = pred_batch[i]
                bbox = batch["bbox"][i].cpu().numpy()
                if current_folder is None or current_folder != folder:
                    # if current_folder is not None:
                    #     current_mean_prob[:] = current_total_prob / (current_total_count + 1e-6)
                    #     thresholded_prob[:] = current_mean_prob > threshold
                    #
                    #     image_paths = dataset.dataset.image_paths[current_folder]
                    #     for c in range(thresholded_prob.shape[0]):
                    #         prob_rle = rle_encode(thresholded_prob[c])
                    #         sub_file.add_line([image_paths[c], prob_rle])
                    #
                    #     if current_total_prob is not None:
                    #         current_total_prob.flush()
                    #     if current_total_count is not None:
                    #         current_total_count.flush()
                    #     if current_mean_prob is not None:
                    #         current_mean_prob.flush()
                    #     if thresholded_prob is not None:
                    #         thresholded_prob.flush()

                    img_h = int(batch["img_h"][i])
                    img_w = int(batch["img_w"][i])
                    img_c = int(batch["img_c"][i])

                    current_folder = folder
                    current_total_prob = create_mmap_array(raw_pred_out_dir / folder / "total_prob", [img_c, img_h, img_w], float).data
                    current_total_count = create_mmap_array(raw_pred_out_dir / folder / "total_count", [img_c, img_h, img_w], np.int64).data
                    current_mean_prob = create_mmap_array(raw_pred_out_dir / folder / "mean_prob", [img_c, img_h, img_w], float).data
                    thresholded_prob = create_mmap_array(raw_pred_out_dir / folder / "thresholded_prob", [img_c, img_h, img_w], bool).data

                lc = bbox[0]
                lx = bbox[1]
                ly = bbox[2]
                uc = bbox[3]
                ux = bbox[4]
                uy = bbox[5]

                current_total_prob[lc: uc, ly: uy, lx: ux] += pred
                current_total_count[lc: uc, ly: uy, lx: ux] += 1

    if current_folder is not None and current_total_prob is not None and current_total_count is not None:
        np.divide(current_total_prob, current_total_count + 1e-6, out=current_mean_prob)

        image_paths = dataset.dataset.image_paths[current_folder]
        for c in range(thresholded_prob.shape[0]):
            # current_mean_prob[c, ...] = current_total_prob[c, ...] / (current_total_count[c, ...] + 1e-6)
            thresholded_prob[c, ...] = current_mean_prob[c, ...] > threshold

            prob_rle = rle_encode(thresholded_prob[c])
            sub_file.add_line([image_paths[c], prob_rle])

        if current_total_prob is not None:
            current_total_prob.flush()
        if current_total_count is not None:
            current_total_count.flush()
        if current_mean_prob is not None:
            current_mean_prob.flush()
        if thresholded_prob is not None:
            thresholded_prob.flush()

    if write_to_tmp_file:
        shutil.rmtree(raw_pred_out_dir)


if __name__ == "__main__":
    from sennet.custom_modules.models import UNet3D

    _ds = ThreeDSegmentationDataset(
        ["kidney_1_dense"],
        50,
        50,
        output_crop_size=50,
        substride=2.0,
    )
    _dl = DataLoader(
        _ds,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
    )
    _model = UNet3D(1, 1)
    generate_submission_df(_model, _dl, 0.5, "/home/clay/", device="cuda")
