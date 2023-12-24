from sennet.custom_modules import *
from sennet.environments.constants import AUG_DUMP_DIR
from sennet.core.dataset import ThreeDSegmentationDataset
from tqdm import tqdm
from torch.utils.data import Subset
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    dataset = ThreeDSegmentationDataset(
        folder=cfg.train_folders[0],
        substride=1.0,
        **cfg.dataset.kwargs,
        **cfg.augmentation,
    )
    dataset = Subset(dataset, list(range(0, len(dataset), 100)))

    save_dir = AUG_DUMP_DIR / "train"
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
        # annotated_img = cv2.addWeighted(np.clip(img * 60 + 127, 0, 255).astype(np.uint8), 0.5, seg_map * 255, 0.5, 0.0)
        annotated_img = cv2.addWeighted(np.clip((img * 0.235 + 0.5)*255, 0, 255).astype(np.uint8), 0.5, seg_map * 255, 0.5, 0.0)
        if save_imgs:
            cv2.imwrite(str(save_dir / f"{str(i).zfill(3)}_{str(seg_count).zfill(8)}.png"), annotated_img)
        print("dumped image")


if __name__ == "__main__":
    main()
