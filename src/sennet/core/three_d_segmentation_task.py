import pandas as pd
from sennet.core.submission_utils import evaluate_chunked_inference
from sennet.core.submission import generate_submission_df, ParallelizationSettings
from sennet.core.utils import resize_3d_image
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.environments.constants import PROCESSED_DATA_DIR, TMP_SUB_MMAP_DIR
from sennet.custom_modules.models import Base3DSegmentor
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import numpy as np
from tqdm import tqdm

class ThreeDSegmentationTask(pl.LightningModule):
    def __init__(
            self,
            model: Base3DSegmentor,
            val_loader: DataLoader,
            val_folders: List[str],
            optimiser_spec: Dict[str, Any],
            experiment_name: str,
            criterion: nn.Module,
            eval_threshold: float = 0.2,
            compute_crude_metrics: bool = False,
            batch_transform: nn.Module = None,
            scan_thresholds: bool = False,
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.val_loader = val_loader
        self.val_folders = val_folders
        self.val_rle_df = []
        self.compute_crude_metrics = compute_crude_metrics
        for f in self.val_folders:
            self.val_rle_df.append(pd.read_csv(PROCESSED_DATA_DIR / f / "rle.csv"))
        self.val_rle_df = pd.concat(self.val_rle_df, axis=0)
        self.optimiser_spec = optimiser_spec
        self.criterion = criterion
        self.experiment_name = experiment_name
        self.eval_threshold = eval_threshold
        self.best_surface_dice = 0.0
        self.batch_transform = batch_transform
        self.scan_thresholds = scan_thresholds

        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_val_loss = 0.0
        self.val_count = 0

    def training_step(self, batch: Dict, batch_idx: int):
        if self.batch_transform is not None:
            batch = self.batch_transform(batch)
        self.model = self.model.train()
        seg_pred = self.model.predict(batch["img"])
        preds = seg_pred.pred
        gt_seg_map = batch["gt_seg_map"].float()

        _, pred_d, pred_h, pred_w = preds.shape
        _, _, gt_d, gt_h, gt_w = gt_seg_map.shape
        if (gt_d != pred_d) or (gt_h != pred_h) or (gt_w != pred_w):
            resized_gt = resize_3d_image(gt_seg_map, (pred_w, pred_h, pred_d))[:, 0, :, :, :]
            # resized_pred = resize_3d_image(preds.unsqueeze(1), (gt_w, gt_h, gt_d))[:, 0, :, :, :]
        else:
            resized_gt = gt_seg_map[:, 0, :, :, :]
            # resized_pred = preds

        loss = self.criterion(preds, resized_gt[:, seg_pred.take_indices_start: seg_pred.take_indices_end, :, :])
        # loss = self.criterion(resized_pred, resized_gt[:, seg_pred.take_indices_start: seg_pred.take_indices_end, :, :])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        if not self.compute_crude_metrics:
            return
        with torch.no_grad():
            self.model = self.model.eval()
            seg_pred = self.model.predict(batch["img"])
            preds = torch.nn.functional.sigmoid(seg_pred.pred) > 0.2
            gt_seg_map = batch["gt_seg_map"][:, 0, :, :, :] > 0.2
            loss = self.criterion(seg_pred.pred, batch["gt_seg_map"][:, 0, :, :, :].float())
            # print(f"{seg_pred.pred.max()=}, {seg_pred.pred.min()=}")

            self.total_val_loss += loss.cpu().item()
            self.val_count += 1
            self.total_tp += (preds & gt_seg_map).sum().cpu().item()
            self.total_fp += (preds & ~gt_seg_map).sum().cpu().item()
            self.total_fn += (~preds & gt_seg_map).sum().cpu().item()

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            # these metrics are meant to sanity-check the eval process as they're very unlikely to have bug
            # otherwise they're to be ignored during model selection
            crude_precision = self.total_tp / (self.total_tp + self.total_fp + 1e-6)
            crude_recall = self.total_tp / (self.total_tp + self.total_fn + 1e-6)
            crude_f1 = 2 * crude_precision * crude_recall / (crude_precision + crude_recall + 1e-6)
            crude_val_loss = self.total_val_loss / (self.val_count + 1e-6)
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0
            self.total_val_loss = 0.0
            self.val_count = 0

            # out_dir = TMP_SUB_MMAP_DIR / self.experiment_name
            out_dir = TMP_SUB_MMAP_DIR / "training_tmp"  # prevent me forgetting to remove tmp dirs
            sub = generate_submission_df(
                self.model,
                self.val_loader,
                threshold=self.eval_threshold,
                scan_thresholds=self.scan_thresholds,
                parallelization_settings=ParallelizationSettings(
                    run_as_single_process=False,
                    n_chunks=5,
                    finalise_one_by_one=True,
                ),
                out_dir=out_dir,
                device="cuda",
                save_sub=True,
            )
            sub_df = sub.submission_df
            filtered_label = self.val_rle_df.loc[self.val_rle_df["id"].isin(sub_df["id"])].copy().sort_values("id").reset_index()
            filtered_label["width"] = sub_df["width"]
            filtered_label["height"] = sub_df["height"]
            
            if self.scan_thresholds:
                rle_thresholds = sub_df.columns[sub_df.columns.str.startswith("rle_")]
                rle_thresholds = [float(x.split("_")[1]) for x in rle_thresholds]
                rle_thresholds = np.array(rle_thresholds)
                sub_df_threshold = sub_df.copy()
                best_th_curr = None
                best_surface_dice_curr = 0.0
                
                for th in tqdm(rle_thresholds, total=len(rle_thresholds)):
                    sub_df_threshold["rle"] = sub_df_threshold[f"rle_{th:.3f}"]
                    surface_dice_score = compute_surface_dice_score(
                        submit=sub_df_threshold,
                        label=filtered_label,
                    )
                    if surface_dice_score > best_surface_dice_curr:
                        best_surface_dice_curr = surface_dice_score
                        best_th_curr = th
            
            surface_dice_score = compute_surface_dice_score(
                submit=sub.submission_df,
                label=filtered_label,
            )
            f1_score = evaluate_chunked_inference(
                root_dir=out_dir,
                label_dir=PROCESSED_DATA_DIR / self.val_folders[0]  # TODO(Sumo): adjust this so we can eval more folders
            )
            print("--------------------------------")
            print(f"{f1_score = }")
            print(f"{surface_dice_score = }")
            print(f"{crude_f1 = }")
            print(f"{crude_val_loss = }")
            if self.scan_thresholds:
                print(f"{best_th_curr = }")
                print(f"{best_surface_dice_curr = }")
            print("--------------------------------")
            if self.scan_thresholds and best_surface_dice_curr > self.best_surface_dice:
                self.best_surface_dice = best_surface_dice_curr
            if not self.scan_thresholds and surface_dice_score > self.best_surface_dice:
                self.best_surface_dice = surface_dice_score
            out_dict = {
                "f1_score": f1_score,
                "surface_dice": surface_dice_score,
                "crude_f1": crude_f1,
                "crude_val_loss": crude_val_loss,
            }
            if self.scan_thresholds:
                out_dict["best_surface_dice"] = best_surface_dice_curr
                out_dict["best_threshold"] = best_th_curr
            self.log_dict(out_dict)

    def configure_optimizers(self):
        if self.optimiser_spec["kwargs"]["lr"] is None:
            self.optimiser_spec["kwargs"]["lr"] = 10 ** self.optimiser_spec["log_lr"]
        optimiser = torch.optim.AdamW(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
