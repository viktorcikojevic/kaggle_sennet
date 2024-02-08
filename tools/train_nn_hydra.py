from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.core.submission_utils import sanitise_val_dataset_kwargs
import sennet.core.three_d_segmentation_task as tasks
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from sennet.custom_modules.losses.loss import CombinedLoss
from sennet.custom_modules.datasets.transforms.batch_transforms import BatchTransform
from sennet.custom_modules.models.base_model import Base3DSegmentor
# from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import sennet.custom_modules.models as models
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
import torch
import json
# import beepy


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.quit_immediately:
        print("---")
        print(json.dumps(cfg_dict, indent=4))
        print("---")
        return 0
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

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
        f"-abs{cfg.apparent_batch_size}"
        f"-llr{cfg.optimiser.log_lr}"
        f"-t{int(cfg.dataset.kwargs.add_depth_along_channel)}{int(cfg.dataset.kwargs.add_depth_along_width)}{int(cfg.dataset.kwargs.add_depth_along_height)}"
        f"-sm{int(cfg.dataset.kwargs.sample_with_mask)}"
        f"-ema{int(cfg.task.kwargs.ema_momentum if 'ema_momentum' in cfg.task.kwargs else 0.0)}"
        f"-cb{int(cfg.dataset.kwargs.cropping_border)}"
        f"-{time_now}"
    )
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{model_out_dir = }")

    dataset_kwargs = cfg_dict["dataset"]["kwargs"]
    augmentation_kwargs = cfg_dict["augmentation"]
    print("train_dataset_kwargs")
    print(json.dumps(dataset_kwargs, indent=4))
    print("train_aug_kwargs")
    print(json.dumps(augmentation_kwargs, indent=4))
    train_dataset = ConcatDataset([
        ThreeDSegmentationDataset(
            folder=folder,
            substride=cfg.dataset.train_substride,
            loss_weight_by_surface=cfg.dataset.loss_weight_by_surface,
            **dataset_kwargs,
            **augmentation_kwargs,
        )
        for folder in cfg.train_folders
    ])

    # note: the depth along width and height turned off to speed up val
    val_dataset_kwargs = sanitise_val_dataset_kwargs(dataset_kwargs, load_ann=True)
    val_dataset_kwargs["add_depth_along_width"] = False
    val_dataset_kwargs["add_depth_along_height"] = False
    val_dataset_kwargs["n_take_channels"] = val_dataset_kwargs["n_appereant_channels"]
    print("val_dataset_kwargs")
    print(json.dumps(val_dataset_kwargs, indent=4))
    # TODO(Sumo): fix this so training works with multiple val sets
    val_dataset = ThreeDSegmentationDataset(
        folder=cfg.val_folders[0],
        substride=cfg.dataset.val_substride,
        **val_dataset_kwargs,
    )
    dummy_val_loader = DataLoader(
        TensorDataset(torch.ones(1, 100)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
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

    OmegaConf.save(cfg, model_out_dir / "config.yaml", resolve=True)

    criterion = CombinedLoss(cfg_dict)

    # create batch transforms
    batch_transform = BatchTransform(**cfg_dict["batch_transform"]["kwargs"]) if "batch_transform" in cfg_dict else None
    
    accumulate_grad_batches = max(1, int(cfg.batch_size / cfg.apparent_batch_size))
    print(f"{accumulate_grad_batches = }")
    # print(model)
    task = getattr(tasks, cfg_dict["task"]["type"])(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_folders=cfg.val_folders,
        optimiser_spec=cfg_dict["optimiser"],
        experiment_name=experiment_name,
        criterion=criterion,
        batch_transform=batch_transform,
        scheduler_spec=cfg_dict["scheduler"],
        accumulate_grad_batches=accumulate_grad_batches,
        **cfg_dict["task"]["kwargs"],
    )
    callbacks = [
        # pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(max_depth=3),
    ]
    if cfg.dry_logger:
        logger = None
    else:
        logger = WandbLogger(project=cfg.exp_name, name=experiment_name)
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config["experiment_name"] = experiment_name
        logger.experiment.config["aug"] = str(train_dataset.datasets[0].augmenter)
        logger.experiment.config["model_full"] = str(model)
        callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks += [
        pl.callbacks.EarlyStopping(
            monitor=cfg.early_stopping_metric,
            mode="max",
            **cfg.early_stopping,
        ),
        # pl.callbacks.ModelCheckpoint(
        #     dirpath=model_out_dir,
        #     save_top_k=-1,
        #     filename=f"{cfg.model.type}" + "-{epoch:02d}",
        # ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=3,
            monitor="f1_score",
            mode="max",
            filename=f"{cfg.model.type}" + "-{epoch:02d}-{f1_score:.2f}",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=3,
            monitor="surface_dice" if cfg.task.type == "ThreeDSegmentationTask" else "val_loss",
            mode="max",
            filename=f"{cfg.model.type}" + "-{epoch:02d}-{surface_dice:.2f}",
        ),
    ]
    # the weird adjustment is because the original val check interval was designed for apparent batch size of 2
    adjusted_val_check_interval = float(cfg.val_check_interval * (2.0 / cfg.apparent_batch_size))
    print(f"{adjusted_val_check_interval = }")
    val_check_interval = min(adjusted_val_check_interval / len(train_loader), 1.0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        val_check_interval=val_check_interval,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        precision="16",
        benchmark=True,
        log_every_n_steps=20,
        # gradient_clip_val=2.0,
        # gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        # strategy=DeepSpeedStrategy(
        #     stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        # ),
        devices=-1,
    )
    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=dummy_val_loader,
    )
    if not cfg.dry_logger:
        logger.experiment.config["best_surface_dice"] = task.best_surface_dice
        logger.experiment.finish()
    return task.best_surface_dice


if __name__ == "__main__":
    main()
