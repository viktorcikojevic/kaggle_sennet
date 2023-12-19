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
        self.criterion = criterion
        self.experiment_name = experiment_name
        self.eval_threshold = eval_threshold
        self.best_surface_dice = 0.0

    def training_step(self, batch: Dict, batch_idx: int):
        self.model = self.model.train()
        seg_pred = self.model.predict(batch["img"])
        preds = seg_pred.pred
        gt_seg_map = batch["gt_seg_map"].float()

        _, pred_d, pred_h, pred_w = preds.shape
        resized_gt = resize_3d_image(gt_seg_map, (pred_w, pred_h, pred_d))[:, 0, :, :, :]

        loss = self.criterion(preds, resized_gt[:, seg_pred.take_indices_start: seg_pred.take_indices_end, :, :])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            out_dir = TMP_SUB_MMAP_DIR / self.experiment_name
            sub = generate_submission_df(
                self.model,
                self.val_loader,
                threshold=self.eval_threshold,
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
            surface_dice_score = compute_surface_dice_score(
                submit=sub.submission_df,
                label=filtered_label,
            )
            f1_score = evaluate_chunked_inference(
                root_dir=out_dir,
                label_dir=PROCESSED_DATA_DIR / self.val_folders[0]  # TODO(Sumo): adjust this so we can eval more folders
            )
            print("--------------------------------")
            print(f"f1_score = {f1_score}")
            print(f"{surface_dice_score = }")
            print("--------------------------------")
            if surface_dice_score > self.best_surface_dice:
                self.best_surface_dice = surface_dice_score
            self.log_dict({
                "f1_score": f1_score,
                "surface_dice": surface_dice_score,
            })

    def configure_optimizers(self):
        if self.optimiser_spec["kwargs"]["lr"] is None:
            self.optimiser_spec["kwargs"]["lr"] = 10 ** self.optimiser_spec["log_lr"]
        optimiser = torch.optim.AdamW(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
