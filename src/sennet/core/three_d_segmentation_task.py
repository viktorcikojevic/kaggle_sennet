import pandas as pd
from sennet.core.submission import generate_submission_df, ParallelizationSettings
# from sennet.custom_modules.metrics.surface_dice_metric import compute_surface_dice_score
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score
from sennet.environments.constants import PROCESSED_DATA_DIR, TMP_SUB_MMAP_DIR
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim


class ThreeDSegmentationTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            val_loader: DataLoader,
            val_folders: List[str],
            optimiser_spec: Dict[str, Any],
            experiment_name: str,
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.val_loader = val_loader
        self.val_folders = val_folders
        self.val_rle_df = []
        for f in self.val_folders:
            self.val_rle_df.append(pd.read_csv(PROCESSED_DATA_DIR / f / "rle.csv"))
        self.val_rle_df = pd.concat(self.val_rle_df, axis=0)
        self.optimiser_spec = optimiser_spec
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.experiment_name = experiment_name
        self.best_surface_dice = 0.0

    def training_step(self, batch: Dict, batch_idx: int):
        self.model = self.model.train()
        preds = self.model(batch["img"])
        loss = self.criterion(preds, batch["gt_seg_map"].float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            sub = generate_submission_df(
                self.model,
                self.val_loader,
                threshold=0.5,
                parallelization_settings=ParallelizationSettings(
                    run_as_single_process=False,
                    n_chunks=5,
                    finalise_one_by_one=True,
                ),
                out_dir=TMP_SUB_MMAP_DIR / self.experiment_name,
                device="cuda",
                save_sub=True,
                compute_val_loss=True,
            )
            sub_df = sub.submission_df
            filtered_label = self.val_rle_df.loc[self.val_rle_df["id"].isin(sub_df["id"])].copy().sort_values("id").reset_index()
            filtered_label["width"] = sub_df["width"]
            filtered_label["height"] = sub_df["height"]
            surface_dice_score = compute_surface_dice_score(
                submit=sub.submission_df,
                label=filtered_label,
            )
            print("--------------------------------")
            print(f"val_loss = {sub.val_loss}")
            print(f"f1_score = {sub.f1_score}")
            print(f"{surface_dice_score = }")
            print("--------------------------------")
            if surface_dice_score > self.best_surface_dice:
                self.best_surface_dice = surface_dice_score
            self.log_dict({
                "val_loss": sub.val_loss,
                "f1_score": sub.f1_score,
                "surface_dice": surface_dice_score,
            })

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
