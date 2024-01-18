import numpy as np
from collections import deque
from line_profiler_pycharm import profile
from tqdm import tqdm


class RegionGrow3D:
    def __init__(
            self,
            image: np.ndarray,
            pred: np.ndarray,
            image_diff_threshold: float,
            label_upper_threshold: float,
            label_lower_bound: float,
            neighbor_mode: str
    ):
        self.image = image
        self.pred = pred
        self.output_mask = np.zeros(self.image.shape, dtype=bool)

        self.image_diff_threshold = image_diff_threshold
        self.label_upper_threshold = label_upper_threshold
        self.label_lower_bound = label_lower_bound
        self.neighbor_mode = neighbor_mode
        self.queue = deque()

    @profile
    def main(self, seed_image: np.ndarray):
        self.output_mask[seed_image] = 1

        print(f"seeding")
        # seed_stride = 2
        # seeds = np.where(seed_image[::seed_stride, ::seed_stride, ::seed_stride])
        # seeds = np.stack(seeds, axis=-1) * seed_stride
        seeds = np.where(seed_image)
        seeds = np.stack(seeds, axis=-1)
        for seed in seeds:
            self.queue.append((seed[0], seed[1], seed[2]))
        print(f"done seeding")

        with tqdm() as pbar:
            while len(self.queue) != 0:
                pbar.set_description(f"{len(self.queue) = }")
                point = self.queue.pop()
                neighbors = self.get_neighbors(point)
                for neighbor in neighbors:
                    self.check_neighbour(neighbor, point)
        return self.output_mask

    @profile
    def get_neighbors(self, newItem: np.ndarray) -> np.ndarray:
        if self.neighbor_mode == "26n":
            neighbors = [
                [newItem[0]-1, newItem[1]-1, newItem[2]-1],   [newItem[0]-1, newItem[1]-1, newItem[2]],   [newItem[0]-1, newItem[1]-1, newItem[2]+1],
                [newItem[0]-1, newItem[1], newItem[2]-1],     [newItem[0]-1, newItem[1], newItem[2]],     [newItem[0]-1, newItem[1], newItem[2]+1],
                [newItem[0]-1, newItem[1]+1, newItem[2]-1],   [newItem[0]-1, newItem[1]+1, newItem[2]],   [newItem[0]-1, newItem[1]+1, newItem[2]+1],
                [newItem[0], newItem[1]-1, newItem[2]-1],     [newItem[0], newItem[1]-1, newItem[2]],     [newItem[0], newItem[1]-1, newItem[2]+1],
                [newItem[0], newItem[1], newItem[2]-1],       [newItem[0], newItem[1], newItem[2]+1],     [newItem[0], newItem[1]+1, newItem[2]-1],
                [newItem[0], newItem[1]+1, newItem[2]],       [newItem[0], newItem[1]+1, newItem[2]+1],   [newItem[0]+1, newItem[1]-1, newItem[2]-1],
                [newItem[0]+1, newItem[1]-1, newItem[2]],     [newItem[0]+1, newItem[1]-1, newItem[2]+1], [newItem[0]+1, newItem[1], newItem[2]-1],
                [newItem[0]+1, newItem[1], newItem[2]],       [newItem[0]+1, newItem[1], newItem[2]+1],   [newItem[0]+1, newItem[1]+1, newItem[2]-1],
                [newItem[0]+1, newItem[1]+1, newItem[2]],     [newItem[0]+1, newItem[1]+1, newItem[2]+1]
            ]
        elif self.neighbor_mode == "6n":
            neighbors = [
                [newItem[0]-1, newItem[1], newItem[2]],
                [newItem[0]+1, newItem[1], newItem[2]],
                [newItem[0], newItem[1]-1, newItem[2]],
                [newItem[0], newItem[1]+1, newItem[2]],
                [newItem[0], newItem[1], newItem[2]-1],
                [newItem[0], newItem[1], newItem[2]+1],
            ]
        else:
            raise RuntimeError(f"unknown {self.neighbor_mode=}")
        return np.array(neighbors).astype(int)

    @profile
    def check_neighbour(self, new_zyx: np.ndarray, point: np.ndarray):
        new_z, new_y, new_x = new_zyx
        if not (
                -1 < new_x < self.image.shape[2]
                and -1 < new_y < self.image.shape[1]
                and -1 < new_z < self.image.shape[0]
        ):
            return
        new_label_val = self.pred[new_z, new_y, new_x]
        if self.output_mask[new_z, new_y, new_x]:
            return
        if new_label_val < self.label_lower_bound:
            return
        p_z, p_y, p_x = point
        abs_image_diff = abs(float(self.image[new_z, new_y, new_x]) - float(self.image[p_z, p_y, p_x]))
        image_passes_check = abs_image_diff < self.image_diff_threshold
        if image_passes_check:
            self.output_mask[new_z, new_y, new_x] = True
            self.queue.append((new_z, new_y, new_x))


if __name__ == "__main__":
    from sennet.core.mmap_arrays import read_mmap_array
    from pathlib import Path
    import numpy as np
    import json

    _stats = json.loads(Path("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_2/stats.json").read_text())
    print(f"copying image")
    # _img = np.ascontiguousarray(read_mmap_array("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_2/image").data.copy())
    _img = read_mmap_array("/home/clay/research/kaggle/sennet/data_dumps/processed/kidney_2/image").data
    print(f"copying pred")
    _preds = read_mmap_array("/home/clay/research/kaggle/sennet/data_dumps/predicted/ensembled/kidney_2/chunk_00/mean_prob").data
    print(f"launching")
    # _img = (_img - _stats["percentiles"]["0.001"]) / (_stats["percentiles"]["99.999"] - _stats["percentiles"]["0.001"])

    # _n = 100
    # _img = np.random.uniform(0, 255, (_n, _n, _n)).astype(np.uint8)
    # _preds = np.random.uniform(0, 1.0, (_n, _n, _n))

    rg = RegionGrow3D(
        image=_img,
        pred=_preds,
        # image_diff_threshold=0.05,
        image_diff_threshold=20,
        label_upper_threshold=0.05,
        label_lower_bound=0.001,
        neighbor_mode="6n"
    )
    rg.main(_preds > 0.2)
