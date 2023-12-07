import numpy as np
from torch.utils.data import DataLoader
from mmengine.dataset.utils import pseudo_collate
from mmengine.dataset import DefaultSampler

from sennet.custom_modules import *
from sennet.core.mmseg_handlings import remove_wandb_vis
from sennet.environments.constants import AUG_DUMP_DIR
from pathlib import Path
import argparse

from mmengine import Config
from mmengine.registry import init_default_scope

from mmseg.registry import DATASETS, VISUALIZERS
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Browse a dataset")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--not-show", default=False, action="store_true")
    parser.add_argument("--val", default=False, action="store_true")
    parser.add_argument(
        "--show-interval",
        type=float,
        default=2,
        help="the interval of show (s)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = remove_wandb_vis(cfg)

    # register all modules in mmseg into the registries
    init_default_scope("mmseg")

    if args.val:
        dataset = DATASETS.build(cfg.val_dataloader.dataset)
    else:
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
    save_dir = AUG_DUMP_DIR / ("val" if args.val else "train")
    save_dir.mkdir(exist_ok=True, parents=True)
    cfg.visualizer["save_dir"] = str(save_dir)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    # visualizer.dataset_meta = dataset.METAINFO
    visualizer.dataset_meta = MultiChannelDataset.METAINFO

    benchmark_dataloader = False
    save_imgs = True
    if benchmark_dataloader:
        num_iters = 100
        batch_size = 2
        num_workers = 1
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=pseudo_collate,
            # sampler=InfiniteSampler(dataset, shuffle=True),
            sampler=DefaultSampler(dataset, shuffle=True),
        )
        print(f"{len(dataloader) = }")
        total_time = 0
        t0 = time.time()
        i = 0
        should_break = False
        with tqdm(total=num_iters) as pbar:
            while True:
                for batch in dataloader:
                    t1 = time.time()
                    total_time += t1-t0
                    t0 = t1
                    pbar.update(1)
                    i += 1
                    should_break = i > num_iters
                    if should_break:
                        break
                    if save_imgs:
                        for j in range(len(batch["inputs"])):
                            img = batch["inputs"][j].permute(1, 2, 0).numpy()
                            for c in range(img.shape[2]):
                                print(c, np.min(img[:, :, c]), np.max(img[:, :, c]))
                            print("===")
                            img = img[..., [0, 1, 2]]
                            if cfg.normalise_img:
                                std = np.array(batch["data_samples"][j].metainfo["std"])[cfg.take_channels]
                                mean = np.array(batch["data_samples"][j].metainfo["mean"])[cfg.take_channels]
                                img = np.clip(img * std[None, None, :] + mean[None, None, :], 0, 255).astype(np.uint8)
                            else:
                                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                            visualizer.add_datasample(
                                f"sample_{str(i).zfill(3)}",
                                img,
                                batch["data_samples"][j],
                                show=not args.not_show,
                                wait_time=args.show_interval
                            )

                if should_break:
                    break
        print(total_time / num_iters)
    else:
        for i, item in tqdm(enumerate(dataset), total=len(dataset)):
            img = item["inputs"].permute(1, 2, 0).numpy()
            data_sample = item["data_samples"].numpy()

            img = img[..., [0, 1, 2]]
            seg_map = (data_sample.gt_sem_seg.data[..., 0] * 255).astype(np.uint8)
            seg_count = np.sum(data_sample.gt_sem_seg.data[..., 0])
            if seg_count == 0:
                continue
            seg_map = np.stack([
                np.zeros_like(seg_map),
                seg_map,
                np.zeros_like(seg_map),
            ], axis=2)
            annotated_img = cv2.addWeighted(img, 0.5, seg_map, 0.5, 0.0)
            if save_imgs:
                cv2.imwrite(str(save_dir / f"{str(i).zfill(3)}_{str(seg_count).zfill(8)}.png"), annotated_img)


if __name__ == "__main__":
    main()
