from mmengine.config import Config
from typing import *


def find_wandb_vis(cfg: Config) -> Optional[Config]:
    wandb_configs = [
        vb for vb in
        cfg.visualizer.vis_backends
        if vb["type"] == "WandbVisBackend"
    ]
    assert len(wandb_configs) in (0, 1), f"found {len(wandb_configs)=}"
    if len(wandb_configs) == 0:
        return None
    return wandb_configs[0]


def remove_wandb_vis(cfg: Config) -> Config:
    if hasattr(cfg, "visualizer") and hasattr(cfg.visualizer, "vis_backends"):
        cfg.visualizer.vis_backends = [
            vb for vb in
            cfg.visualizer.vis_backends
            if vb["type"] != "WandbVisBackend"
        ]
    return cfg
