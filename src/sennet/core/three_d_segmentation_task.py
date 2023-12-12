import pandas as pd
from sennet.core.submission import generate_submission_df
from sennet.custom_modules.metrics.surface_dice_metric import score as get_surface_dice_score
from sennet.environments.constants import PROCESSED_DATA_DIR
import pytorch_lightning as pl
from typing import Dict, Any
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from pathlib import Path


class ThreeDSegmentationTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            val_loader: DataLoader,
            optimiser_spec: Dict[str, Any],
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.val_loader = val_loader
        self.val_folders = self.val_loader.dataset.folders
        self.val_rle_df = []
        for f in self.val_folders:
            self.val_rle_df.append(pd.read_csv(PROCESSED_DATA_DIR / f / "rle.csv"))
        self.val_rle_df = pd.concat(self.val_rle_df, axis=0)
        self.optimiser_spec = optimiser_spec
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.best_surface_dice = 0.0

    def training_step(self, batch: Dict, batch_idx: int):
        preds = self.model(batch["img"])
        loss = self.criterion(preds, batch["gt_seg_map"].float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            sub_out_dir = Path(self.logger.log_dir)
            generate_submission_df(
                self.model,
                self.val_loader,
                threshold=0.5,
                sub_out_dir=sub_out_dir,
                raw_pred_out_dir=None,
                device="cuda",
            )
            submission_df = pd.read_csv(sub_out_dir / "submission.csv")
            surface_dice_score = get_surface_dice_score(
                solution=submission_df,
                submission=self.val_rle_df,
                row_id_column_name="id",
                rle_column_name="rle",
                tolerance=1.0,
                image_id_column_name=None,
                slice_id_column_name=None,
                resize_fraction=1.0,
            )
            if surface_dice_score > self.best_surface_dice:
                self.best_surface_dice = surface_dice_score

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
