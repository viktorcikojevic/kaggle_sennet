from sennet.core.mmap_arrays import create_mmap_array, read_mmap_array
from typing import Tuple, List
from pathlib import Path
import numpy as np
import mmap


def round_to_be_divisible(n: int, d: int) -> int:
    return int(d * (n // d))


class PartiallyFlushable3DMmap:
    def __init__(
            self,
            output_dir: Path,
            shape: Tuple[int, int, int],
            dtype: type,
    ):
        self.output_dir = output_dir
        self.shape = shape
        self.dtype = dtype

        self.arr = None
        self.file = None
        self.mmap = None

        # probably can generalise this even more, but eh
        self.item_size_bytes = np.dtype(self.dtype).itemsize
        self.row_stride_in_bytes = self.item_size_bytes * self.shape[2]
        self.channel_stride_in_bytes = self.item_size_bytes * self.shape[1] * self.shape[2]
        self.total_size_in_bytes = self.item_size_bytes * self.shape[0] * self.shape[1] * self.shape[2]

    def init(self):
        create_mmap_array(self.output_dir, list(self.shape), self.dtype)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_dir / "data.npy", "r+")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)

    def __del__(self):
        self.mmap.close()
        self.file.close()

    def _block_to_contiguous_list(self, block: List[int]):
        lc, lx, ly, uc, ux, uy = block
        chunk_size = (ux - lx) * self.item_size_bytes
        contiguous_list = [
            (
                self.channel_stride_in_bytes * c + self.row_stride_in_bytes * y + self.item_size_bytes * lx,
                chunk_size
            )
            for c in range(lc, uc)
            for y in range(ly, uy)
        ]  # start, size
        return contiguous_list

    def write_block(self, block: List[int], data: np.ndarray):
        lc, lx, ly, uc, ux, uy = block
        chunks = self._block_to_contiguous_list(block)
        i = 0
        for c in range(uc - lc):
            for y in range(uy - ly):
                offset, size = chunks[i]
                self.mmap.seek(offset)
                self.mmap.write(data[c, y, :].tobytes())

                # flush needs offset to be divisible by the page size
                # rounded_offset = round_to_be_divisible(offset, mmap.PAGESIZE)
                # rounded_size = size + (offset - rounded_offset)
                # self.mmap.flush(rounded_offset, rounded_size)
                i += 1


if __name__ == "__main__":
    from sennet.environments.constants import DATA_DUMPS_DIR
    import cv2

    _out_dir = DATA_DUMPS_DIR / "dummy_image"
    _img_shape = (3, 100, 200)
    _img = PartiallyFlushable3DMmap(_out_dir, tuple(_img_shape), float)
    _img.init()

    _img.write_block([0, 10, 20, 2, 100, 80], np.ones((2, 60, 100), dtype=float) * 127)
    _img.write_block([1, 50, 50, 3, 200, 100], np.ones((2, 50, 150), dtype=float) * 50)
    _read_mmap = read_mmap_array(_out_dir)
    _show_img = np.stack([_read_mmap.data[c, :, :] for c in range(3)], axis=2).astype(np.uint8)
    cv2.imwrite(str(_out_dir / "img.png"), _show_img)
