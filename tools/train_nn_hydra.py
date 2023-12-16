from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.core.three_d_segmentation_task import ThreeDSegmentationTask
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.environments.constants import MODEL_OUT_DIR
from sennet.custom_modules.models import UNet3D
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
# import beepy


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = f"model_{time_now}"
    experiment_name = f"{str(cfg.model.type)}-{time_now}"
    model_out_dir = MODEL_OUT_DIR / dir_name
    model_out_dir.mkdir(exist_ok=True, parents=True)

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
    val_dataset = ThreeDSegmentationDataset(
        folder=cfg.val_folders[0],
        substride=cfg.dataset.val_substride,
        **dataset_kwargs,
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
        batch_size=cfg.batch_size,
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
    model = UNet3D(1, 1, final_sigmoid=False)
    # ---------------------------------------
    OmegaConf.save(cfg, model_out_dir / "config.yaml")

    task = ThreeDSegmentationTask(
        model,
        val_loader=val_loader,
        val_folders=cfg.val_folders,
        optimiser_spec=cfg_dict["optimiser"],
        experiment_name=experiment_name
        # **cfg_dict["task"]["kwargs"],
    )
    if cfg.dry_logger:
        logger = None
    else:
        logger = WandbLogger(project=cfg.exp_name, name=experiment_name)
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config["dir_name"] = dir_name
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
