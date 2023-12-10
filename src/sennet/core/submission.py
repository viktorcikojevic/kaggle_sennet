from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.rles import rle_encode
from sennet.core.mmap_arrays import create_mmap_array
from typing import Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np


def generate_submission_df(
        model: nn.Module,
        data_loader: DataLoader,
        threshold: float,
        raw_pred_out_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
        f"to generate submission, dataset must be ThreeDSegmentationDataset"
    dataset: ThreeDSegmentationDataset = data_loader.dataset
    model = model.eval()
    current_total_prob = None
    current_total_count = None
    current_folder = None
    out_data = {"id": [], "rle": []}
    with torch.no_grad():
        for batch in data_loader:
            pred_batch = model(batch["img"])
            for i in range(len(pred_batch)):
                folder = batch["folder"][i]
                pred = torch.nn.functional.sigmoid(pred_batch[i])
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

                pred = pred.cpu().numpy()
                current_total_prob[bbox[0]: bbox[3], bbox[1]: bbox[4], bbox[2]: bbox[5]] += pred
                current_total_count[bbox[0]: bbox[3], bbox[1]: bbox[4], bbox[2]: bbox[5]] += 1
    return pd.DataFrame(out_data)
