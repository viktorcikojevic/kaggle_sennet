import numpy as np

from sennet.custom_modules import *
from sennet.core.submission_utils import sanitise_val_dataset_kwargs
from sennet.environments.constants import AUG_DUMP_DIR
from sennet.core.dataset import ThreeDSegmentationDataset
from tqdm import tqdm
from torch.utils.data import Subset
from omegaconf import DictConfig, OmegaConf
import hydra


def _browse_dataset(cfg, dataset: ThreeDSegmentationDataset, save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)

    save_imgs = True
    mid_section = int(cfg.dataset.kwargs.n_take_channels / 2)
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        img = item["img"][0, [mid_section, mid_section, mid_section], ...].permute(1, 2, 0).numpy()
        seg_map = item["gt_seg_map"][0, mid_section, ...].numpy()
        seg_count = np.sum(seg_map)
        if seg_count == 0:
            continue
        seg_map = np.stack([
            np.zeros_like(seg_map),
            seg_map,
            np.zeros_like(seg_map),
        ], axis=2)
        decoded_img = np.clip((img - img.min()) / (np.ptp(img) + 1e-6) * 255, 0, 255).astype(np.uint8)
        annotated_img = cv2.addWeighted(decoded_img, 0.5, seg_map * 255, 0.5, 0.0)
        if save_imgs:
            cv2.imwrite(str(save_dir / f"{str(i).zfill(3)}_{'_'.join([str(x) for x in item['bbox']])}_{item['bbox_type']}_{img.min():.2f}_{img.max():.2f}.png"), annotated_img)
        # print(f"dumped image: {img.min()=} {img.max()=} {decoded_img.min()=} {decoded_img.max()=}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    train_dataset = ThreeDSegmentationDataset(
        folder=cfg.train_folders[0],
        substride=1.0,
        **cfg.dataset.kwargs,
        **cfg.augmentation,
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    val_dataset = ThreeDSegmentationDataset(
        folder=cfg.val_folders[0],
        substride=1.0,
        **sanitise_val_dataset_kwargs(cfg_dict["dataset"]["kwargs"], load_ann=True),
    )

    train_dataset = Subset(train_dataset, list(range(0, len(train_dataset), 100)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), 100)))

    train_save_dir = AUG_DUMP_DIR / "train"
    val_save_dir = AUG_DUMP_DIR / "val"
    # _browse_dataset(cfg, val_dataset, val_save_dir)
    _browse_dataset(cfg, train_dataset, train_save_dir)


if __name__ == "__main__":
    main()
