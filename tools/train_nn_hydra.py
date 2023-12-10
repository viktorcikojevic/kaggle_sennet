from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.core.three_d_segmentation_task import ThreeDSegmentationTask
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.environments.constants import MODEL_OUT_DIR
from torch.utils.data import DataLoader
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
# import beepy


from sennet.custom_modules.models.resnet3d import Resnet3D34


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = f"model_{time_now}"

    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    dataset_kwargs = cfg_dict["dataset"]["kwargs"]
    train_dataset = ThreeDSegmentationDataset(
        folders=cfg.train_folders,
        **dataset_kwargs,
    )
    val_dataset = ThreeDSegmentationDataset(
        folders=cfg.val_folders,
        **dataset_kwargs,
    )
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
        batch_size=2*cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------
    model = Resnet3D34(
        n_input_channels=1,
        n_classes=1,
    )
    # ---------------------------------------

    task = ThreeDSegmentationTask(
        model,
        val_loader=val_loader,
        optimiser_spec=cfg_dict["optimiser"],
        # **cfg_dict["task"]["kwargs"],
    )
    if cfg.dry_logger:
        logger = None
    else:
        logger = WandbLogger(project=cfg.exp_name, name=f"{str(cfg_dict['model']['type'])}")
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
            dirpath=MODEL_OUT_DIR,
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
