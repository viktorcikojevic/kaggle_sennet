from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.rles import rle_encode
from sennet.core.mmap_arrays import create_mmap_array
from sennet.environments.constants import TMP_SUB_MMAP_DIR
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from line_profiler_pycharm import profile
from datetime import datetime
import shutil
import pandas as pd
import multiprocessing as mp
from dataclasses import dataclass


class AppendingSubmissionCsv:
    def __init__(self, csv_path: Union[Path, str]):
        self.csv_path = csv_path
        Path(self.csv_path).write_text(f"id,rle")
        self.file = open(csv_path, "a")

    def __del__(self):
        self.file.flush()
        self.file.close()

    def add_line(self, items: List[str]):
        self.file.write(f"\n{','.join(items)}")


@dataclass
class Data:
    pred: Optional[torch.Tensor] = None
    batch: Optional[Dict[str, torch.Tensor]] = None
    terminate: bool = False


class MockQueue:
    def __init__(self):
        self.item = None

    def get(self, *a, **kw):
        return self.item

    def put(self, item, *a, **kw):
        self.item = item


class TensorReceivingProcess:
    def __init__(
            self,
            data_queue: mp.Queue,
            threshold: float,
            all_image_paths: Dict[str, List[str]],
            sub_out_dir: str = "/tmp",
            raw_pred_out_dir: Optional[Union[str, Path]] = None,
    ):
        self.data_queue = data_queue
        self.threshold = threshold
        self.all_image_paths = all_image_paths
        self.sub_out_dir = sub_out_dir
        self.raw_pred_out_dir = raw_pred_out_dir

        self.current_total_count = None
        self.current_mean_prob = None
        self.thresholded_prob = None

        self.current_folder = None
        self.sub_file = None
        self.write_to_tmp_file = False

    def _finalise_image_if_holding_any(self):
        if self.current_folder is not None and self.current_mean_prob is not None and self.current_total_count is not None:
            image_paths = self.all_image_paths[self.current_folder]
            for c in tqdm(range(self.thresholded_prob.shape[0])):
                self.current_mean_prob[c, ...] /= (self.current_total_count[c, ...] + 1e-6)
                self.thresholded_prob[c, ...] = self.current_mean_prob[c, ...] > self.threshold

                prob_rle = rle_encode(self.thresholded_prob[c])
                self.sub_file.add_line([image_paths[c], prob_rle])
            print(f"flushing total count")
            self.current_total_count.flush()
            print(f"flushing mean prob")
            self.current_mean_prob.flush()
            print(f"flushing thresholded prob")
            self.thresholded_prob.flush()
            print(f"done")

    def start(self):
        # just appear as if it's a process for debugging
        self.setup()

    def join(self):
        # just appear as if it's a process for debugging
        self.finalise()

    def spin(self):
        self.setup()
        while True:
            should_terminate = self.spin_once()
            if should_terminate:
                self.finalise()
                return

    def setup(self):
        self.sub_file = AppendingSubmissionCsv(Path(self.sub_out_dir) / "submission.csv")
        self.write_to_tmp_file = self.raw_pred_out_dir is None
        if self.write_to_tmp_file:
            self.raw_pred_out_dir = TMP_SUB_MMAP_DIR / f"sennet_tmp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    @profile
    def spin_once(self) -> bool:
        data: Data = self.data_queue.get()
        pred_batch = data.pred
        batch = data.batch
        terminate = data.terminate

        if terminate:
            return True

        for i in range(len(pred_batch)):
            folder = batch["folder"][i]
            pred = pred_batch[i].cpu().numpy()
            bbox = batch["bbox"][i].cpu().numpy()
            if self.current_folder is None or self.current_folder != folder:
                self._finalise_image_if_holding_any()

                img_h = int(batch["img_h"][i])
                img_w = int(batch["img_w"][i])
                img_c = int(batch["img_c"][i])

                self.current_folder = folder
                folder_name = Path(folder).name
                self.current_total_count = create_mmap_array(self.raw_pred_out_dir / folder_name / "total_count", [img_c, img_h, img_w], np.uint8).data
                self.current_mean_prob = create_mmap_array(self.raw_pred_out_dir / folder_name / "mean_prob", [img_c, img_h, img_w], float).data
                self.thresholded_prob = create_mmap_array(self.raw_pred_out_dir / folder_name / "thresholded_prob", [img_c, img_h, img_w], bool).data

            lc = bbox[0]
            lx = bbox[1]
            ly = bbox[2]
            uc = bbox[3]
            ux = bbox[4]
            uy = bbox[5]

            self.current_mean_prob[lc: uc, ly: uy, lx: ux] += pred
            self.current_total_count[lc: uc, ly: uy, lx: ux] += 1

        return False

    def finalise(self):
        self._finalise_image_if_holding_any()
        if self.write_to_tmp_file and self.raw_pred_out_dir.is_dir():
            shutil.rmtree(self.raw_pred_out_dir)


@dataclass
class ParallelizationSettings:
    run_as_single_process: bool = False


@profile
def generate_submission_df(
        model: nn.Module,
        data_loader: DataLoader,
        threshold: float,
        parallelization_settings: ParallelizationSettings,
        sub_out_dir: Union[str, Path] = "/tmp",
        raw_pred_out_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
) -> pd.DataFrame:
    if parallelization_settings.run_as_single_process:
        data_queue = MockQueue()
    else:
        mp.set_start_method("spawn", force=True)
        data_queue = mp.Queue(maxsize=10)

    assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
        f"to generate submission, dataset must be ThreeDSegmentationDataset"
    dataset: ThreeDSegmentationDataset = data_loader.dataset
    model = model.eval().to(device)

    tensor_receiving_process = TensorReceivingProcess(
        data_queue=data_queue,
        threshold=threshold,
        all_image_paths=dataset.dataset.image_paths,
        sub_out_dir=str(sub_out_dir),
        raw_pred_out_dir=raw_pred_out_dir,
    )
    if parallelization_settings.run_as_single_process:
        saver_process = tensor_receiving_process
    else:
        saver_process = mp.Process(
            target=tensor_receiving_process.spin,
        )
    saver_process.start()

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            pred_batch = model(batch["img"].to(device))[:, 0, :, :, :]
            pred_batch = torch.nn.functional.sigmoid(pred_batch)
            data_queue.put(Data(pred=pred_batch, batch=batch))
            if parallelization_settings.run_as_single_process:
                saver_process.spin_once()
    data_queue.put(Data(terminate=True,))
    saver_process.join()

    submission_path = Path(sub_out_dir) / "submission.csv"
    df_out = pd.read_csv(submission_path)
    # replace nan rle with "1 0"
    df_out['rle'] = df_out['rle'].fillna("1 0")
    return df_out


if __name__ == "__main__":
    from sennet.custom_modules.models import UNet3D

    _crop_size = 100
    _ds = ThreeDSegmentationDataset(
        "kidney_1_dense",
        _crop_size,
        _crop_size,
        output_crop_size=_crop_size,
        substride=1.0,
    )
    _dl = DataLoader(
        _ds,
        batch_size=2,
        shuffle=True,   # deliberate
        pin_memory=True,
    )
    _model = UNet3D(1, 1)
    generate_submission_df(
        _model,
        _dl,
        0.5,
        ParallelizationSettings(
            run_as_single_process=False,
        ),
        "/home/clay/",
        device="cuda",
    )
