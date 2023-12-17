from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.core.three_d_segmentation_task import ThreeDSegmentationTask
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from sennet.custom_modules.models import UNet3D
from sennet.custom_modules.losses.loss import CombinedLoss
from torch.utils.data import DataLoader, ConcatDataset
import sennet.custom_modules.models as models
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from typing import Dict
from hydra.core.hydra_config import HydraConfig
import hydra
import torch
# import beepy


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = f"{str(cfg.model.type)}-{time_now}"
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{model_out_dir = }")

    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    dataset_kwargs = cfg_dict["dataset"]["kwargs"]
    train_dataset = ConcatDataset([
        ThreeDSegmentationDataset(
            folder=folder,
            substride=cfg.dataset.train_substride,
            **dataset_kwargs,
        )
        for folder in cfg.train_folders
    ])
    # TODO(Sumo): fix this so training works with multiple val sets

    val_dataset_kwargs = deepcopy(dataset_kwargs)
    val_dataset_kwargs["crop_size_range"] = None
    val_dataset_kwargs["channels_jitter"] = None
    val_dataset_kwargs["p_channel_jitter"] = 0.0
    val_dataset_kwargs["crop_location_noise"] = 0
    val_dataset = ThreeDSegmentationDataset(
        folder=cfg.val_folders[0],
        substride=cfg.dataset.val_substride,
        **val_dataset_kwargs,
    )
    # val_dataset = ConcatDataset([
    #     ThreeDSegmentationDataset(
    #         folder=folder,
    #         substride=cfg.dataset.val_substride,
    #         **dataset_kwargs,
    #     )
    #     for folder in cfg.val_folders
    # ])
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(cfg.batch_size / cfg.accumulate_grad_batches)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------
    model_class = getattr(models, cfg_dict["model"]["type"])
    model = model_class(**cfg_dict["model"]["kwargs"])
    if "pretrained" in cfg_dict["model"] and cfg_dict["model"]["pretrained"] is not None:
        ckpt = torch.load(PRETRAINED_DIR / cfg_dict["model"]["pretrained"])
        load_res = model.load_state_dict(ckpt["model_state_dict"])
        print(f"{load_res = }")
    else:
        print("no pretrained model given")
    # ---------------------------------------
    OmegaConf.save(cfg, model_out_dir / "config.yaml")

    criterion = CombinedLoss(cfg_dict)

    task = ThreeDSegmentationTask(
        model,
        val_loader=val_loader,
        val_folders=cfg.val_folders,
        optimiser_spec=cfg_dict["optimiser"],
        experiment_name=experiment_name,
        criterion=criterion
        # **cfg_dict["task"]["kwargs"],
    )
    if cfg.dry_logger:
        logger = None
    else:
        logger = WandbLogger(project=cfg.exp_name, name=experiment_name)
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config["experiment_name"] = experiment_name
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        # pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]
    callbacks += [
        pl.callbacks.EarlyStopping(
            monitor="surface_dice",
            patience=cfg.patience,
            verbose=True,
            mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=1,
            monitor="surface_dice",
            mode="max",
            filename=f"{cfg.model.type}" + "-{epoch:02d}-{surface_dice:.2f}",
        ),
    ]
    val_check_interval = float(cfg.val_check_interval) / len(train_loader)
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        val_check_interval=val_check_interval,
        max_epochs=cfg.max_epochs,
        precision=16,
        log_every_n_steps=20,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,
    )
    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if not cfg.dry_logger:
        logger.experiment.config["best_surface_dice"] = task.best_surface_dice
        logger.experiment.finish()
    return task.best_surface_dice


if __name__ == "__main__":
    main()
