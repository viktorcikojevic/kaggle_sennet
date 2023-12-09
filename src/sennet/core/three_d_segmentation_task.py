from sennet.custom_modules.metrics.surface_dice_metric import score as get_surface_dice_score
import pytorch_lightning as pl
from typing import Dict, Any
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim


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
        self.optimiser_spec = optimiser_spec
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.best_surface_dice = 0.0

    def training_step(self, batch: Dict, batch_idx: int):
        preds = self.model(batch["img"])
        loss = self.criterion(preds, batch["gt_seg_map"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            # TODO(Sumo): your eval script goes here
            surface_dice_score = get_surface_dice_score(
                solution=...,
                submission=...,
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
