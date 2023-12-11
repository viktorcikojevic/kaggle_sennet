from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.rles import rle_encode
from sennet.core.mmap_arrays import create_mmap_array
from typing import Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile


@profile
def generate_submission_df(
        model: nn.Module,
        data_loader: DataLoader,
        threshold: float,
        raw_pred_out_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
) -> pd.DataFrame:
    assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
        f"to generate submission, dataset must be ThreeDSegmentationDataset"
    dataset: ThreeDSegmentationDataset = data_loader.dataset
    model = model.eval().to(device)
    current_total_prob = None
    current_total_count = None
    current_folder = None
    out_data = {"id": [], "rle": []}
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
                    if current_folder is not None:
                        current_mean_prob = current_total_prob / (current_total_count + 1e-6)
                        thresholded_prob = current_mean_prob > threshold

                        image_paths = dataset.dataset.image_paths[current_folder]
                        for c in range(thresholded_prob.shape[0]):
                            prob_rle = rle_encode(thresholded_prob[c])
                            out_data["id"].append(image_paths[c])
                            out_data["rle"].append(prob_rle)

                        if raw_pred_out_dir is not None:
                            mean_prob_mmap = create_mmap_array(raw_pred_out_dir, list(current_mean_prob.shape), current_mean_prob.dtype)
                            mean_prob_mmap.data[:] = current_mean_prob
                            mean_prob_mmap.data.flush()

                    img_h = int(batch["img_h"][i])
                    img_w = int(batch["img_w"][i])
                    img_c = int(batch["img_c"][i])

                    current_folder = folder
                    current_total_prob = np.zeros((img_c, img_h, img_w), dtype=float)
                    current_total_count = np.zeros((img_c, img_h, img_w), dtype=np.int64)

                # pred = pred.cpu().numpy()

                lc = bbox[0]
                lx = bbox[1]
                ly = bbox[2]
                uc = bbox[3]
                ux = bbox[4]
                uy = bbox[5]

                current_total_prob[lc: uc, ly: uy, lx: ux] += pred
                current_total_count[lc: uc, ly: uy, lx: ux] += 1

            if current_folder is not None:
                current_mean_prob = current_total_prob / (current_total_count + 1e-6)
                thresholded_prob = current_mean_prob > threshold

                image_paths = dataset.dataset.image_paths[current_folder]
                for c in range(thresholded_prob.shape[0]):
                    prob_rle = rle_encode(thresholded_prob[c])
                    out_data["id"].append(image_paths[c])
                    out_data["rle"].append(prob_rle)

                if raw_pred_out_dir is not None:
                    mean_prob_mmap = create_mmap_array(raw_pred_out_dir, list(current_mean_prob.shape), current_mean_prob.dtype)
                    mean_prob_mmap.data[:] = current_mean_prob
                    mean_prob_mmap.data.flush()

    return pd.DataFrame(out_data)


if __name__ == "__main__":
    from sennet.custom_modules.models import UNet3D

    _ds = ThreeDSegmentationDataset(
        ["kidney_1_dense"],
        50,
        50,
        substride=2.0,
    )
    _dl = DataLoader(
        _ds,
        batch_size=10,
        shuffle=True,
        pin_memory=True,
    )
    _model = UNet3D(1, 1)
    _output = generate_submission_df(_model, _dl, 0.5, device="cuda")
