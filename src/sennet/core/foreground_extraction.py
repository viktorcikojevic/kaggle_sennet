import numpy as np
import cv2
# from line_profiler_pycharm import profile
from scipy.signal import find_peaks


# @profile
def get_foreground_mask(
        img: np.ndarray,
        hist_bins: int = 20,
) -> np.ndarray:
    n_bins = hist_bins

    bin_freq, bin_edges = np.histogram(img, bins=n_bins)
    sorted_means = bin_edges[find_peaks(bin_freq)[0]]
    if len(sorted_means) < 2:
        return np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    idx0 = np.argmin(np.abs(bin_edges - sorted_means[0]))
    idx1 = np.argmin(np.abs(bin_edges - sorted_means[1]))
    if idx0 == idx1:
        return np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    split_val = bin_edges[np.argmin(bin_freq[idx0: idx1]) + idx0]

    mask = ((img > split_val) * 255).astype(np.uint8)
    kernel_size = 7
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    return mask > 0


if __name__ == "__main__":
    import time
    _img = cv2.imread("/home/clay/research/kaggle/sennet/data/blood-vessel-segmentation/train/kidney_1_dense/images/0110.tif", 0)
    _t0 = time.monotonic()
    _mask = get_foreground_mask(_img)
    _t1 = time.monotonic()
    print(f"{_t1 - _t0} s")
    cv2.imwrite("/home/clay/a_img.png", _img)
    cv2.imwrite("/home/clay/a_mask.png", (_mask * 255).astype(np.uint8))
    cv2.imwrite("/home/clay/a_combined.png", np.clip(_img.astype(float) + _mask.astype(float) * 100, 0, 255).astype(np.uint8))
