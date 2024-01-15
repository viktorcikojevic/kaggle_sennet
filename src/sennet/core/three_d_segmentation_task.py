from sennet.core.submission_utils import evaluate_chunked_inference, ChunkedMetrics
from sennet.core.submission_simple import generate_submission_df, ParallelizationSettings
from sennet.core.utils import resize_3d_image
from sennet.environments.constants import PROCESSED_DATA_DIR, TMP_SUB_MMAP_DIR
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.custom_modules.models import Base3DSegmentor
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn as nn
import torch.optim
import json
import numpy as np
from tqdm import tqdm


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
            train_loader: DataLoader,
            val_loader: DataLoader,
            val_folders: List[str],
            optimiser_spec: Dict[str, Any],
            experiment_name: str,
            criterion: nn.Module,
            eval_threshold: float = 0.2,
            compute_crude_metrics: bool = False,
            batch_transform: nn.Module = None,
            ema_momentum: float | None = None,
            scheduler_spec: dict[str, Any] = None,
            ignore_border_loss: bool = False,
            accumulate_grad_batches: int = 1,
            **kwargs
    ):
        pl.LightningModule.__init__(self)
        print(f"unused kwargs: {kwargs}")
        self.model = model
        self.freezing_parameters = self.get_freezing_parameters()
        self.ema_momentum = ema_momentum
        if self.ema_momentum is not None:
            print(f"{ema_momentum=} is given, evaluations will be done using ema")
            self.ema_model = EMA(self.model, self.ema_momentum)
        else:
            print(f"{ema_momentum=} not given, evaluations will be done using the model")
            self.ema_model = None
        self.val_loader = val_loader
        self.val_folders = val_folders
        self.compute_crude_metrics = compute_crude_metrics
        self.optimiser_spec = optimiser_spec
        self.scheduler_spec = scheduler_spec
        self.criterion = criterion
        self.experiment_name = experiment_name
        self.eval_threshold = eval_threshold
        self.best_surface_dice = 0.0
        self.best_f1_score = 0.0
        self.batch_transform = batch_transform
        self.ignore_border_loss = ignore_border_loss

        self.train_loader = train_loader
        self.accumulate_grad_batches = accumulate_grad_batches
        assert isinstance(self.val_loader.dataset, ThreeDSegmentationDataset), \
            f"to generate submission, dataset must be ThreeDSegmentationDataset"
        self.cropping_border = self.val_loader.dataset.dataset.cropping_border

        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_val_loss = 0.0
        self.val_count = 0

    def get_freezing_parameters(self):
        return self.model.freezing_parameters if hasattr(self.model, "freezing_parameters") else []


    def training_step(self, batch: Dict, batch_idx: int):
        if self.batch_transform is not None:
            batch = self.batch_transform(batch)
        # self.model = self.model.train()
        # freeze layers
        for name, param in self.model.model.named_parameters():
            if name in self.freezing_parameters:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        seg_pred = self.model.predict(batch["img"])
        preds = seg_pred.pred
        gt_seg_map = batch["gt_seg_map"].float()

        _, pred_d, pred_h, pred_w = preds.shape
        _, _, gt_d, gt_h, gt_w = gt_seg_map.shape
        if (gt_d != pred_d) or (gt_h != pred_h) or (gt_w != pred_w):
            # resized_gt = resize_3d_image(gt_seg_map, (pred_w, pred_h, pred_d))[:, 0, :, :, :]
            resized_pred = resize_3d_image(preds.unsqueeze(1), (gt_w, gt_h, gt_d))[:, 0, :, :, :]
            raise RuntimeError(":D")
        else:
            # resized_gt = gt_seg_map[:, 0, :, :, :]
            resized_pred = preds

        # loss = self.criterion(preds, resized_gt[:, seg_pred.take_indices_start: seg_pred.take_indices_end, :, :])
        if self.ignore_border_loss:
            loss = self.criterion(
                resized_pred[
                    :,
                    :,
                    self.cropping_border: resized_pred.shape[2] - self.cropping_border,
                    self.cropping_border: resized_pred.shape[3] - self.cropping_border,
                ],
                gt_seg_map[
                    :,
                    0,
                    seg_pred.take_indices_start: seg_pred.take_indices_end,
                    self.cropping_border: gt_seg_map.shape[3]-self.cropping_border,
                    self.cropping_border: gt_seg_map.shape[4]-self.cropping_border,
                ],
            )
        else:
            loss = self.criterion(resized_pred, gt_seg_map[:, 0, :, :, :])
        current_lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.log_dict({
            "train_loss": loss,
            "lr": current_lr,
        }, prog_bar=True)
        return loss

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return pl.LightningModule.backward(self, loss, *args, **kwargs)

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

            out_dir = TMP_SUB_MMAP_DIR / "training_tmp"
            model = self._get_eval_model()
            _ = generate_submission_df(
                model,
                self.val_loader,
                threshold=self.eval_threshold,
                parallelization_settings=ParallelizationSettings(
                    run_as_single_process=False,
                ),
                out_dir=out_dir,
                device="cuda",
                save_sub=True,
            )

            # mmap_paths = sorted([p for p in out_dir.glob("chunk_*")])
            # mmap_arrays = [read_mmap_array(p / "thresholded_prob") for p in mmap_paths]
            # mmap = np.concatenate([m.data for m in mmap_arrays], axis=0)
            # cc3d_out_dir = TMP_SUB_MMAP_DIR / "training_tmp_cc3d"
            # cc3d_out_dir.mkdir(exist_ok=True, parents=True)
            # Path(cc3d_out_dir / "image_names").write_text(Path(out_dir / "image_names").read_text())
            # filter_out_small_blobs(
            #     thresholded_pred=mmap,
            #     out_path=cc3d_out_dir / "chunk_00" / "thresholded_prob",
            #     dust_threshold=1000,
            #     connectivity=26,
            # )

            thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.08, 0.1]
            metrics: ChunkedMetrics = evaluate_chunked_inference(
                root_dir=out_dir,
                # root_dir=cc3d_out_dir,
                label_dir=PROCESSED_DATA_DIR / self.val_folders[0],  # TODO(Sumo): adjust this so we can eval more folders
                thresholds=thresholds,
            )
            surface_dice_scores = metrics.surface_dices
            best_dice_current = np.max(surface_dice_scores)
            best_threshold_current = thresholds[np.argmax(surface_dice_scores)]
            best_f1_score = np.max(metrics.f1_scores)
            print("--------------------------------")
            print("precisions:")
            print(json.dumps({t: d for t, d in zip(thresholds, metrics.precisions)}, indent=4))
            print("recalls:")
            print(json.dumps({t: d for t, d in zip(thresholds, metrics.recalls)}, indent=4))
            print(f"f1_scores:")
            print(f"{json.dumps({t: d for t, d in zip(thresholds, metrics.f1_scores)}, indent=4)}")
            print("dice_scores:")
            print(json.dumps({t: d for t, d in zip(thresholds, surface_dice_scores)}, indent=4))
            print(f"best_threshold_current = {best_threshold_current}")
            print(f"best_dice_current = {best_dice_current}")
            print(f"{crude_f1 = }")
            print(f"{crude_val_loss = }")
            print("--------------------------------")
            if best_dice_current > self.best_surface_dice:
                self.best_surface_dice = best_dice_current
            if best_f1_score > self.best_f1_score:
                self.best_f1_score = best_f1_score
            self.log_dict({
                "f1_score": best_f1_score,
                "precision": np.max(metrics.precisions),
                "recall": np.max(metrics.recalls),
                "threshold": best_threshold_current,
                "surface_dice": best_dice_current,
                "crude_f1": crude_f1,
                "crude_val_loss": crude_val_loss,
            })

    def configure_optimizers(self):
        if self.optimiser_spec["kwargs"]["lr"] is None:
            self.optimiser_spec["kwargs"]["lr"] = 10 ** self.optimiser_spec["log_lr"]
        optimiser_class = getattr(torch.optim, self.optimiser_spec["type"])
        optimiser = optimiser_class(self.model.parameters(), **self.optimiser_spec["kwargs"])
        print(f"{optimiser = }")
        ret_val = {
            "optimizer": optimiser,
        }
        if self.scheduler_spec is not None and "type" in self.scheduler_spec:
            scheduler_kwargs = self.scheduler_spec["kwargs"]
            if "override_total_steps" in self.scheduler_spec:
                key = self.scheduler_spec["override_total_steps"]["key"]
                num_epochs = self.scheduler_spec["override_total_steps"]["num_epochs"]
                train_loader = self.train_loader
                scheduler_kwargs[key] = int(num_epochs * len(train_loader) / self.accumulate_grad_batches) + 1
                print(f"scheduler override_total_steps given, now set to {scheduler_kwargs[key]}")
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_spec["type"])
            print(f"{scheduler_kwargs = }")
            scheduler = scheduler_class(
                optimizer=optimiser,
                **scheduler_kwargs,
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",
            }
            print(f"{scheduler = }")
            ret_val["lr_scheduler"] = scheduler_dict
        else:
            print("no scheduler")
        return ret_val


class DenoiseTask(ThreeDSegmentationTask):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        ThreeDSegmentationTask.__init__(self, *args, **kwargs)
        self.std_noise = kwargs["std_noise"]

    def training_step(self, batch: Dict, batch_idx: int):
        self.model = self.model.train()
        
        img = batch["img"]
        
        noise = torch.randn_like(img) * self.std_noise
        img = img + noise
        
        img_pred = self.model.predict(img).pred.squeeze()
        
        # L1 loss between img_true and img_pred
        loss = nn.L1Loss()(img_pred, noise.squeeze())

        self.log_dict({
            "train_loss": loss,
        }, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            # loop over self.val_loader
            val_loss = 0
            for batch in tqdm(self.val_loader, total=len(self.val_loader)):
                self.model = self.model.eval()

                img = batch["img"].to(self.device)
                noise = torch.randn_like(img) * self.std_noise
                noise = noise.to(self.device)
                img = img + noise
                img_pred = self.model.predict(img).pred

                # loss is mse between img_true and img_pred
                loss = nn.L1Loss()(img_pred.squeeze(), noise.squeeze())
                val_loss += loss
            
            val_loss = val_loss / len(self.val_loader)
            self.log_dict({
                "val_loss": val_loss,
            })

