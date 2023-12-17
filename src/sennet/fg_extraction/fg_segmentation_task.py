import pytorch_lightning as pl
from typing import Dict, Any
import torch.nn as nn
import torch.optim
import torch
from torchvision.transforms import functional as tvf


class ForegroundSegmentationTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimiser_spec: Dict[str, Any],
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.optimiser_spec = optimiser_spec
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.best_f1_score = 0.0

    def training_step(self, batch: Dict, batch_idx: int):
        self.model = self.model.train()
        gt = batch["gt_seg_map"].float()
        # preds = torch.nn.functional.sigmoid(self.model(batch["img"]))
        # resized_preds = tvf.resize(preds, [gt.shape[2], gt.shape[3]])
        # loss = self.criterion(resized_preds, gt)
        preds = self.model(batch["img"])
        resized_gt = tvf.resize(gt, [preds.shape[2], preds.shape[3]])
        loss = self.criterion(preds, resized_gt)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        with torch.no_grad():
            self.model = self.model.eval()
            gt = batch["gt_seg_map"].float()

            preds = torch.nn.functional.sigmoid(self.model(batch["img"]))
            resized_preds = tvf.resize(preds, [gt.shape[2], gt.shape[3]])
            thresholded_pred = resized_preds[:, 0, :, :] > 0.5
            thresholded_gt = batch["gt_seg_map"][:, 0, :, :].float() > 0.5

            self.total_tp += (thresholded_pred & thresholded_gt).sum().cpu().item()
            self.total_fp += (thresholded_pred & ~thresholded_gt).sum().cpu().item()
            self.total_fn += (~thresholded_pred & thresholded_gt).sum().cpu().item()

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            precision = self.total_tp / (self.total_tp + self.total_fp + 1e-6)
            recall = self.total_tp / (self.total_tp + self.total_fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall)
            print("--------------------------------")
            print(f"f1_score = {f1_score}")
            print("--------------------------------")
            if f1_score > self.best_f1_score:
                self.best_f1_score = f1_score
            self.log_dict({
                "f1_score": f1_score,
            })
            self.total_tp = 0
            self.total_fp = 0
            self.total_fn = 0

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
