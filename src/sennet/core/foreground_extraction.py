import numpy as np
import cv2
# from line_profiler_pycharm import profile
from scipy.signal import find_peaks


# @profile
def get_foreground_mask(
        img: np.ndarray,
        hist_bins: int = 50,
        inlier_p: float = 1.0
) -> np.ndarray:
    n_bins = hist_bins
    # n_bins = min(len(np.unique(img)), hist_bins)

    low_percentile, median, high_percentile = np.percentile(img, [inlier_p, 50, 100 - inlier_p])
    inter_percentile_range = high_percentile - low_percentile
    outlier_lb = low_percentile - 2.0 * inter_percentile_range
    outlier_ub = high_percentile + 2.0 * inter_percentile_range

    bin_freq, bin_edges = np.histogram(img[(img > outlier_lb) & (img < outlier_ub)], bins=n_bins)
    all_peaks, _ = find_peaks(bin_freq, distance=float(5.0 / (bin_edges[1] - bin_edges[0])))
    if len(all_peaks) < 2:
        print(f"all_peaks < 2: {all_peaks}")
        return np.ones((img.shape[0], img.shape[1]), dtype=bool)
    peak_heights = bin_freq[all_peaks]
    tallest_peak_indices = np.argsort(peak_heights)[-2:]
    peaks = all_peaks[tallest_peak_indices]
    sorted_means = np.sort(bin_edges[peaks])
    if len(sorted_means) < 2:
        print(f"sorted_means < 2: {sorted_means}")
        return np.ones((img.shape[0], img.shape[1]), dtype=bool)

    idx0 = np.argmin(np.abs(bin_edges - sorted_means[0]))
    idx1 = np.argmin(np.abs(bin_edges - sorted_means[1]))
    if idx0 == idx1:
        print(f"idx0 == idx1: {idx0}")
        return np.ones((img.shape[0], img.shape[1]), dtype=bool)

    # min_freq = np.min(bin_freq[idx0: idx1])
    # print(f"{min_freq=}")
    # print(f"indices: {np.argwhere(bin_freq[idx0: idx1] == min_freq)[0]}")
    # split_val = bin_edges[int(np.mean(np.argwhere(bin_freq[idx0: idx1] == min_freq)[0]) + idx0)]
    # print(f"{split_val=}")
    split_val = 0.5 * (bin_edges[idx0] + bin_edges[idx1])

    mask = ((img > split_val) * 255).astype(np.uint8)
    kernel_size = 7
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))

    if (mask > 0).mean() < 0.002:
        return np.ones((img.shape[0], img.shape[1]), dtype=bool)
    #     print(f"huh")
    # if (mask > 0).mean() > 0.9:
    #     print(f"big mask found")

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
