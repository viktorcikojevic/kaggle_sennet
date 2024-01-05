import os

import sennet.custom_modules.models as models
from sennet.custom_modules.metrics.surface_dice_metric_fast import compute_surface_dice_score_from_mmap
from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.mmap_arrays import read_mmap_array
from sennet.core.rles import rle_encode
from typing import Union, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC, BinaryF1Score
from collections import OrderedDict
import dataclasses
import yaml
import torch
from copy import deepcopy
import numpy as np


def generate_submission_df_from_one_chunked_inference(
        root_dir: Path,
) -> pd.DataFrame:
    image_names = (root_dir / "image_names").read_text().split("\n")
    chunk_dirs = sorted(list(root_dir.glob("chunk*")))
    i = 0
    data = {"id": [], "rle": [], "height": [], "width": []}
    for d in tqdm(chunk_dirs, position=0):
        pred = read_mmap_array(d / "thresholded_prob", mode="r")
        for c in tqdm(range(pred.shape[0]), position=1, leave=False):
            rle = rle_encode(pred.data[c, :, :])
            if rle == "":
                rle = "1 0"
            image_name = image_names[i]
            i += 1
            data["id"].append(image_name)
            data["rle"].append(rle)
            data["height"].append(int(pred.data.shape[1]))
            data["width"].append(int(pred.data.shape[2]))
    df = pd.DataFrame(data).sort_values("id")
    # df = df.set_index("id").sort_index()
    return df


@dataclasses.dataclass
class ChunkedMetrics:
    f1_score: float
    # binary_au_roc: float
    surface_dices: list[float]


def evaluate_chunked_inference(
        root_dir: Union[str, Path],
        label_dir: Union[str, Path],
        thresholds: list[float] = (0.2,),
        device: str = "cuda",
) -> ChunkedMetrics:
    with torch.no_grad():
        root_dir = Path(root_dir)
        label_dir = Path(label_dir)

        chunk_dirs = sorted(list(root_dir.glob("chunk*")))
        label = read_mmap_array(label_dir / "label", mode="r")

        surface_dices = []
        for t in tqdm(thresholds, desc="dice"):
            dice = compute_surface_dice_score_from_mmap(
                mean_prob_chunks=[
                    read_mmap_array(d / "mean_prob", mode="r").data
                    for d in chunk_dirs
                ],
                label=label.data,
                threshold=t,
            )
            surface_dices.append(dice)

        f1_metric = BinaryF1Score()
        # au_roc_metric = BinaryAUROC()

        i = 0
        for d in tqdm(chunk_dirs, position=0):
            pred = read_mmap_array(d / "thresholded_prob", mode="r")
            for c in tqdm(range(pred.shape[0]), position=1, leave=False):
                pred_tensor = torch.from_numpy(pred.data[c, :, :].copy()).reshape(-1).to(device)
                # mean_prob_tensor = torch.from_numpy(mean_prob.data[c, :, :].copy()).reshape(-1).to(device)
                target_tensor = torch.from_numpy(label.data[i, :, :].copy()).reshape(-1).to(device)

                f1_metric.update(pred_tensor, target_tensor)
                # au_roc_metric.update(mean_prob_tensor[::10], target_tensor[::10])

                i += 1
        f1_score = float(f1_metric.compute().cpu().item())
        # au_roc_score = float(au_roc_metric.compute().cpu().item())
        return ChunkedMetrics(
            f1_score=f1_score,
            # binary_au_roc=au_roc_score,
            surface_dices=surface_dices,
        )


def load_config_from_dir(model_dir: str | Path) -> Dict:
    with open(model_dir / "config.yaml", "rb") as f:
        cfg = yaml.load(f, yaml.FullLoader)
    return cfg


def load_model_from_dir(model_dir: str | Path) -> Tuple[Dict, Optional[models.Base3DSegmentor]]:
    trimmed_prefix = "AAA_trimmed_"

    model_dir = Path(model_dir)
    cfg = load_config_from_dir(model_dir)
    ckpt_paths = sorted(list(model_dir.glob("*.ckpt")))
    ckpt_path = ckpt_paths[0]
    is_ckpt_path_trimmed = "trimmed" in ckpt_path.name
    if len(ckpt_paths) > 1 and is_ckpt_path_trimmed:
        outdated = ckpt_path.name.replace(trimmed_prefix, "") != ckpt_paths[1].name
        if outdated:
            print(f"trimmed checkpoint is outdated: trimmed={ckpt_path.name}, found={ckpt_paths[1].name}")
            is_ckpt_path_trimmed = False
            os.remove(ckpt_path)  # prevent it from coming up again
            ckpt_path = ckpt_paths[1]

    print(f"using ckpt: {ckpt_path}")
    model_class = getattr(models, cfg["model"]["type"])

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    if "pretrained" in cfg["model"]["kwargs"]:
        print(f"model kwargs contains pretrained, replacing it with None")
        cfg["model"]["kwargs"]["pretrained"] = None
    if "encoder_weights" in cfg["model"]["kwargs"]:
        print(f"model kwargs contains encoder_weights, replacing it with None")
        cfg["model"]["kwargs"]["encoder_weights"] = None

    if is_ckpt_path_trimmed:
        print("trimmed ckpt found")
        model_state_dict = ckpt["state_dict"]
    elif any(k.startswith("ema_") for k in state_dict.keys()):
        print("ema weights found, loading ema weights")
        model_state_dict = OrderedDict([
            (k, v)
            for k, v in state_dict.items()
            if k.startswith("ema_model.")
        ])
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="ema_model.module.")
    else:
        print("ema weights not found, loading model")
        model_state_dict = OrderedDict([
            (k, v)
            for k, v in state_dict.items()
            if k.startswith("model.")
        ])
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_state_dict, prefix="model.")
    model = model_class(**cfg["model"]["kwargs"])
    load_status = model.load_state_dict(model_state_dict)
    print(load_status)
    model = model.eval()

    # trim down checkpoint and save the trimmed version, probably can shrink ckpt sizes by like 50%
    if is_ckpt_path_trimmed:
        print("loaded ckpt is trimmed so no need to trim it again")
    else:
        ckpt = {"state_dict": model_state_dict}
        trimmed_ckpt_path = ckpt_path.parent / f"{trimmed_prefix}{ckpt_path.name}"
        torch.save(ckpt, trimmed_ckpt_path)
        print(f"saved trimmed ckpt path: {trimmed_ckpt_path}")
    
    if 'n_appereant_channels' in cfg['dataset']['kwargs']:
        cfg['dataset']['kwargs']['n_take_channels'] = cfg['dataset']['kwargs']['n_appereant_channels']
    
    return cfg, model


def sanitise_val_dataset_kwargs(kwargs, load_ann: bool = False) -> dict[str, any]:
    kwargs = deepcopy(kwargs)
    kwargs["crop_size_range"] = None
    kwargs["load_ann"] = load_ann
    kwargs["assert_label_exists"] = load_ann
    kwargs["crop_location_noise"] = 0
    kwargs["p_crop_location_noise"] = 0
    kwargs["p_crop_size_noise"] = 0
    kwargs["augmenter_class"] = None
    kwargs["augmenter_kwargs"] = None
    kwargs["p_random_3d_rotation"] = 0.0
    return kwargs


def build_data_loader(folder: str, substride: float, cfg: Dict, cropping_border: int | None = None):
    kwargs = sanitise_val_dataset_kwargs(cfg["dataset"]["kwargs"], load_ann=False)
    if cropping_border is not None:
        kwargs["cropping_border"] = cropping_border
    dataset = ThreeDSegmentationDataset(
        folder,
        substride=substride,
        **kwargs,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader


if __name__ == "__main__":
    # _root_dir = Path("/home/clay/research/kaggle/sennet/data_dumps/tmp_mmaps/Resnet3D34-2023-12-16-10-06-40")
    # _df = generate_submission_df_from_one_chunked_inference(_root_dir)
    # # _df.to_csv(_root_dir / "submission.csv")
    #
    # # _df = pd.read_csv(_root_dir / "submission.csv")
    # _label = pd.read_csv(DATA_DIR / "train_rles.csv")
    # _filtered_label = _label.loc[_label["id"].isin(_df["id"])].copy().sort_values("id").reset_index()
    # _filtered_label["width"] = _df["width"]
    # _filtered_label["height"] = _df["height"]
    # _score = compute_surface_dice_score(
    #     submit=_df,
    #     label=_filtered_label,
    # )
    # print(f"{_score = }")

    # _score = evaluate_chunked_inference(
    #     "/home/clay/research/kaggle/sennet/data_dumps/tmp_mmaps/WrappedUNet2D-c512x1-2023-12-18-17-16-31",
    #     "/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_3_sparse",
    # )
    # print(f"{_score = }")
    _cfg, _model = load_model_from_dir(
        "/home/clay/research/kaggle/sennet/data_dumps/models/SMP(Unet_resnet50_imagenet)-c512x1-bs32-llr-3-t111-sm0-2023-12-24-22-11-35",
    )
    # _cfg, _model = load_model_from_dir(
    #     "/home/clay/research/kaggle/sennet/data_dumps/models/SMP(Unet_resnet50_imagenet)-c512x1-bs32-llr-3-t111-sm0-2023-12-24-20-53-43",
    # )
