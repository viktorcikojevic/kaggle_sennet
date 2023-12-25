from sennet.core.mmap_arrays import create_mmap_array, MmapArray
from pathlib import Path
import numpy as np
import cc3d


def filter_out_small_blobs(
        thresholded_pred: np.ndarray,
        out_path: str | Path,
        dust_threshold: int = 1000,
        connectivity: int = 26,
):
    print(f"computing cc3d object: {thresholded_pred.shape}")
    print(f"computing dust")
    labels_out = cc3d.dust(
        thresholded_pred,
        threshold=dust_threshold,
        connectivity=connectivity,
        in_place=True,
    )
    print("dumping output mmap")
    out_mmap = create_mmap_array(
        Path(out_path),
        list(thresholded_pred.shape),
        bool,
    )
    out_mmap.data[:] = False
    out_mmap.data[labels_out > 0] = True
    print("flushing out mmap")
    out_mmap.data.flush()
