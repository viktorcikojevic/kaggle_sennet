from sklearn.mixture import GaussianMixture
import numpy as np
import cv2


def get_foreground_mask(
        img: np.ndarray,
        hist_bins: int = 50,
        contour_area_pct_threshold: float = 0.1,
) -> np.ndarray:
    flattened_img = img.flatten()

    gmm = GaussianMixture(n_components=2)
    gmm.fit(flattened_img.reshape((-1, 1)))

    bin_freq, bin_edges = np.histogram(flattened_img, bins=hist_bins)

    sorted_means = sorted(gmm.means_)
    idx0 = np.argmin(np.abs(bin_edges - sorted_means[0]))
    idx1 = np.argmin(np.abs(bin_edges - sorted_means[1]))
    split_val = bin_edges[np.argmin(bin_freq[idx0: idx1]) + idx0]

    thr = split_val
    mask = ((img > thr) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask[:] = 0
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area / img.shape[0] / img.shape[1] < contour_area_pct_threshold:
            continue
        mask = cv2.drawContours(mask, [hull], 0, 255, -1)

    return mask > 0
