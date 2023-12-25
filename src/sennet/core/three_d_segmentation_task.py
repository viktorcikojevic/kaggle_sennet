import pandas as pd
from sennet.core.submission_utils import evaluate_chunked_inference, ChunkedMetrics
from sennet.core.submission import generate_submission_df, ParallelizationSettings
from sennet.core.utils import resize_3d_image
from sennet.environments.constants import PROCESSED_DATA_DIR, TMP_SUB_MMAP_DIR
from sennet.custom_modules.models import Base3DSegmentor
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn as nn
import torch.optim
import json
import numpy as np

class EMA(nn.Module):
    def __init__(self, model, momentum=0.00001):
        # https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060
        # https://github.com/Lightning-AI/pytorch-lightning/issues/10914
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.momentum = momentum
        self.decay = 1 - self.momentum

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


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
            ema_momentum: float | None = None,
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.ema_momentum = ema_momentum
        if self.ema_momentum is not None:
            print(f"{ema_momentum=} is given, evaluations will be done using ema")
            self.ema_model = EMA(self.model, self.ema_momentum)
        else:
            print(f"{ema_momentum=} not given, evaluations will be done using the model")
            self.ema_model = None
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
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return loss

    def _get_eval_model(self):
        if self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self.model.eval()
        return model

    def validation_step(self, batch: Dict, batch_idx: int):
        if not self.compute_crude_metrics:
            return
        with torch.no_grad():
            model = self._get_eval_model()
            seg_pred = model.predict(batch["img"])
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
            model = self._get_eval_model()
            sub = generate_submission_df(
                model,
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
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            metrics: ChunkedMetrics = evaluate_chunked_inference(
                root_dir=out_dir,
                label_dir=PROCESSED_DATA_DIR / self.val_folders[0],  # TODO(Sumo): adjust this so we can eval more folders
                thresholds=thresholds,
            )
            surface_dice_scores = metrics.surface_dices
            best_dice_current = np.max(surface_dice_scores)
            best_threshold_current = thresholds[np.argmax(surface_dice_scores)]
            print("--------------------------------")
            print(f"f1_score = {metrics.f1_score}")
            print("dice_scores:")
            print(json.dumps({t: d for t, d in zip(thresholds, surface_dice_scores)}, indent=4))
            print(f"best_threshold_current = {best_threshold_current}")
            print(f"best_dice_current = {best_dice_current}")
            print(f"{crude_f1 = }")
            print(f"{crude_val_loss = }")
            print("--------------------------------")
            if best_dice_current > self.best_surface_dice:
                self.best_surface_dice = best_dice_current
            self.log_dict({
                "f1_score": metrics.f1_score,
                "threshold": best_threshold_current,
                "surface_dice": best_dice_current,
                "crude_f1": crude_f1,
                "crude_val_loss": crude_val_loss,
            })

    def configure_optimizers(self):
        if self.optimiser_spec["kwargs"]["lr"] is None:
            self.optimiser_spec["kwargs"]["lr"] = 10 ** self.optimiser_spec["log_lr"]
        optimiser = torch.optim.AdamW(self.model.parameters(), **self.optimiser_spec["kwargs"])
        return {
            "optimizer": optimiser,
        }
