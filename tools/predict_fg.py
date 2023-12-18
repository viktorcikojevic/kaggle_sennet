from sennet.core.submission_utils import load_model_from_dir
from sennet.fg_extraction.dataset import ForegroundSegmentationDataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torchvision.transforms.functional as tvf
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args, _ = parser.parse_known_args()

    model_dir = args.model
    data_path = args.path
    output_dir = Path(args.output_dir)

    cfg, model = load_model_from_dir(model_dir)
    dataset_kwargs = cfg["dataset"].get("kwargs", {})
    dataset_kwargs["aug"] = False
    dataset = ForegroundSegmentationDataset(data_path, **dataset_kwargs)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    device = "cuda"
    model = model.to(device).eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            pred_prob = torch.nn.functional.sigmoid(model(batch["img"].to(device)))[:, 0, :, :]
            resized_pred_prob = tvf.resize(pred_prob, [batch["img_h"][0], batch["img_w"][0]])
            for i in range(resized_pred_prob.shape[0]):
                pred = ((resized_pred_prob[i] > 0.5).cpu().numpy() * 255).astype(np.uint8)
                img_path = batch["img_path"][i]
                out_path = output_dir / f"{Path(img_path).stem}_mask.png"
                cv2.imwrite(str(out_path), pred)


if __name__ == "__main__":
    main()
