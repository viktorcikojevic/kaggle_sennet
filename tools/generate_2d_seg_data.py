import numpy as np

from sennet.environments.constants import PROCESSED_DATA_DIR, PROCESSED_2D_DATA_DIR
from sennet.core.mmap_arrays import read_mmap_array
from tqdm import tqdm
import cv2


def main():
    processed_paths = sorted([
        d for d in PROCESSED_DATA_DIR.glob("*")
        if ((d / "image").is_dir() and (d / "mask").is_dir())
    ])
    stride = 20
    for p in tqdm(processed_paths, position=0):
        rel_path = p.relative_to(PROCESSED_DATA_DIR)
        out_dir = PROCESSED_2D_DATA_DIR / rel_path
        image = read_mmap_array(p / "image", mode="r")
        mask = read_mmap_array(p / "mask", mode="r")
        out_dir.mkdir(exist_ok=True, parents=True)
        for c in tqdm(range(0, image.shape[0], stride), position=1, leave=False):
            img_c = image.data[c, :, :]
            mask_c = mask.data[c, :, :]
            img_id = str(c).zfill(4)
            img_out_path = out_dir / f"{img_id}_{out_dir.name}_img.png"
            mask_out_path = out_dir / f"{img_id}_{out_dir.name}_mask.png"
            cv2.imwrite(str(img_out_path), img_c)
            cv2.imwrite(str(mask_out_path), (mask_c * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
