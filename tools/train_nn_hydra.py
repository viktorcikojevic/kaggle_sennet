from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.core.submission_utils import sanitise_val_dataset_kwargs
from sennet.core.three_d_segmentation_task import ThreeDSegmentationTask
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from sennet.custom_modules.losses.loss import CombinedLoss
from sennet.custom_modules.datasets.transforms.batch_transforms import BatchTransform
from sennet.custom_modules.models.base_model import Base3DSegmentor
from torch.utils.data import DataLoader, ConcatDataset
import sennet.custom_modules.models as models
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from hydra.core.hydra_config import HydraConfig
import hydra
import torch
# import beepy


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)

    # ---------------------------------------
    model_class = getattr(models, cfg_dict["model"]["type"])
    model: Base3DSegmentor = model_class(**cfg_dict["model"]["kwargs"])
    if "pretrained" in cfg_dict["model"] and cfg_dict["model"]["pretrained"] is not None:
        ckpt = torch.load(PRETRAINED_DIR / cfg_dict["model"]["pretrained"])
        load_res = model.load_state_dict(ckpt["model_state_dict"])
        print(f"{load_res = }")
    else:
        print("no pretrained model given")
    # ---------------------------------------

    experiment_name = (
        f"{model.get_name()}"
        f"-c{cfg.dataset.kwargs.crop_size}x{cfg.dataset.kwargs.n_take_channels}"
        f"-bs{cfg.batch_size}"
        f"-llr{cfg.optimiser.log_lr}"
        f"-t{int(cfg.dataset.kwargs.add_depth_along_channel)}{int(cfg.dataset.kwargs.add_depth_along_width)}{int(cfg.dataset.kwargs.add_depth_along_height)}"
        f"-sm{int(cfg.dataset.kwargs.sample_with_mask)}"
        f"-{time_now}"
    )
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{model_out_dir = }")

    dataset_kwargs = cfg_dict["dataset"]["kwargs"]
    augmentation_kwargs = cfg_dict["augmentation"]
    train_dataset = ConcatDataset([
        ThreeDSegmentationDataset(
            folder=folder,
            substride=cfg.dataset.train_substride,
            **dataset_kwargs,
            **augmentation_kwargs,
        )
        for folder in cfg.train_folders
    ])
    # TODO(Sumo): fix this so training works with multiple val sets

    val_dataset_kwargs = sanitise_val_dataset_kwargs(dataset_kwargs, load_ann=True)
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
        batch_size=cfg.apparent_batch_size,
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

    OmegaConf.save(cfg, model_out_dir / "config.yaml")

    criterion = CombinedLoss(cfg_dict)

    # create batch transforms
    batch_transform = BatchTransform(**cfg_dict["batch_transform"]["kwargs"]) if "batch_transform" in cfg_dict else None
    
    task = ThreeDSegmentationTask(
        model,
        val_loader=val_loader,
        val_folders=cfg.val_folders,
        optimiser_spec=cfg_dict["optimiser"],
        experiment_name=experiment_name,
        criterion=criterion,
        batch_transform=batch_transform,
        **cfg_dict["task"]["kwargs"],
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
        pl.callbacks.RichModelSummary(max_depth=3),
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
    val_check_interval = min(float(cfg.val_check_interval) / len(train_loader), 1.0)
    accumulate_grad_batches = max(1, int(cfg.batch_size / cfg.apparent_batch_size))
    print(f"{accumulate_grad_batches = }")
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        val_check_interval=val_check_interval,
        max_epochs=cfg.max_epochs,
        precision="16-mixed",
        log_every_n_steps=20,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_grad_batches,
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
