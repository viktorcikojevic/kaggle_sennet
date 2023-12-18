from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sennet.fg_extraction.dataset import ForegroundSegmentationDataset
from sennet.fg_extraction.fg_segmentation_task import ForegroundSegmentationTask
from sennet.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from torch.utils.data import DataLoader, ConcatDataset
import sennet.custom_modules.models as models
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from copy import deepcopy
import hydra
import torch
# import beepy


@hydra.main(config_path="../configs", config_name="train_fg", version_base="1.2")
def main(cfg: DictConfig):
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = f"{str(cfg.model.type)}-{time_now}"
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{model_out_dir = }")

    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    train_dataset = ConcatDataset([
        ForegroundSegmentationDataset(**kwargs, **cfg_dict["dataset"]["kwargs"])
        for kwargs in cfg.dataset.train_kwargs
    ])
    general_val_kwargs = deepcopy(cfg_dict["dataset"]["kwargs"])
    general_val_kwargs["aug"] = False
    val_dataset = ConcatDataset([
        ForegroundSegmentationDataset(**kwargs, **general_val_kwargs)
        for kwargs in cfg.dataset.val_kwargs
    ])

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

    task = ForegroundSegmentationTask(
        model,
        optimiser_spec=cfg_dict["optimiser"],
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
            monitor="f1_score",
            patience=cfg.patience,
            verbose=True,
            mode="max"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=1,
            monitor="f1_score",
            mode="max",
            filename=f"{cfg.model.type}" + "-{epoch:02d}-{f1_score:.2f}",
        ),
    ]
    # val_check_interval = int(float(cfg.val_check_interval) / len(train_loader))
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        # val_check_interval=val_check_interval,
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
        logger.experiment.config["f1_score"] = task.best_f1_score
        logger.experiment.finish()
    return task.best_f1_score


if __name__ == "__main__":
    main()
