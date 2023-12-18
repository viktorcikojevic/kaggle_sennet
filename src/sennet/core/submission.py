from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.submission_utils import generate_submission_df_from_one_chunked_inference
from sennet.core.mp.tensor_distributor import TensorDistributor, TensorChunk
from sennet.core.mmap_arrays import create_mmap_array
from sennet.custom_modules.models import Base3DSegmentor, SegmentorOutput
from sennet.environments.constants import TMP_SUB_MMAP_DIR
from sennet.core.utils import resize_3d_image, DEPTH_ALONG_WIDTH, DEPTH_ALONG_HEIGHT, DEPTH_ALONG_CHANNEL
from typing import Optional, Union, Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from line_profiler_pycharm import profile
from datetime import datetime
import pandas as pd
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class Data:
    terminate: bool = False
    idx: Optional[int] = None
    pred: Optional[TensorChunk] = None
    batch: Optional[Dict[str, torch.Tensor]] = None
    bbox_type: Optional[int] = None
    model_start_idx: Optional[int] = None
    model_end_idx: Optional[int] = None


class MockQueue:
    def __init__(self):
        self.item = None
        self.has_val = False

    def get(self, *a, **kw):
        self.has_val = False
        return self.item

    def put(self, item, *a, **kw):
        self.item = item
        self.has_val = True


class TensorReceivingProcess:
    def __init__(
            self,
            data_queue: mp.Queue,
            threshold: float,
            chunk_boundary: Tuple[int, int],
            out_dir: Optional[Union[str, Path]] = None,
    ):
        self.data_queue = data_queue
        self.threshold = threshold
        self.out_dir = out_dir
        self.chunk_boundary = chunk_boundary

        self.current_total_count = None
        self.current_mean_prob = None
        self.thresholded_prob = None

        self.current_folder = None

    def _finalise_image_if_holding_any(self):
        if self.current_folder is not None and self.current_mean_prob is not None and self.current_total_count is not None:
            # image_paths = self.all_image_paths[self.current_folder]
            print(f"{self.chunk_boundary}: computing mean")
            self.current_mean_prob /= (self.current_total_count + 1e-6)

            print(f"{self.chunk_boundary}: flushing mean prob")
            self.current_mean_prob.flush()

            print(f"{self.chunk_boundary}: flushing total count")
            self.current_total_count.flush()

            print(f"{self.chunk_boundary}: thresholding prob")
            self.thresholded_prob[:] = self.current_mean_prob > self.threshold

            print(f"{self.chunk_boundary}: flushing thresholded prob")
            self.thresholded_prob.flush()

            print(f"{self.chunk_boundary}: done")

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
        pass
        # self.out_dir.mkdir(exist_ok=True, parents=True)
        # for d in self.out_dir.glob("*"):
        #     if d.is_dir():
        #         print(f"removing {d.absolute().resolve()}")
        #         shutil.rmtree(str(d.absolute().resolve()))

    @profile
    def spin_once(self) -> bool:
        data: Data = self.data_queue.get()
        i = data.idx
        pred_chunk = data.pred
        batch = data.batch
        terminate = data.terminate
        bbox_type = data.bbox_type

        if terminate:
            return True

        assert pred_chunk is not None, f"pred can't be None if terminate is False"
        assert bbox_type is not None, f"bbox_type be None if terminate is False"
        assert data.model_start_idx is not None, f"data.model_start_idx be None if terminate is False"
        assert data.model_end_idx is not None, f"data.model_end_idx be None if terminate is False"
        pred = pred_chunk.tensor.cpu().numpy()
        bbox = batch["bbox"][i].cpu().numpy()
        folder = batch["folder"][i]

        if bbox_type == DEPTH_ALONG_CHANNEL:
            lc = pred_chunk.fill_start
            uc = pred_chunk.fill_end
            lx = bbox[1]
            ux = bbox[4]
            ly = bbox[2]
            uy = bbox[5]
        elif bbox_type == DEPTH_ALONG_WIDTH:
            lc = pred_chunk.fill_start
            uc = pred_chunk.fill_end
            lx = bbox[1] + data.model_start_idx
            ux = bbox[1] + data.model_end_idx
            ly = bbox[2]
            uy = bbox[5]
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            lc = pred_chunk.fill_start
            uc = pred_chunk.fill_end
            lx = bbox[1]
            ux = bbox[4]
            ly = bbox[2] + data.model_start_idx
            uy = bbox[2] + data.model_end_idx
        else:
            raise RuntimeError(f"unknown bbox type at {self.__class__.__name__}: {bbox_type=}")

        if self.current_folder is None:
            img_h = int(batch["img_h"][i])
            img_w = int(batch["img_w"][i])
            # img_c = int(batch["img_c"][i])
            img_c = self.chunk_boundary[1] - self.chunk_boundary[0]

            self.current_folder = folder
            self.current_total_count = create_mmap_array(self.out_dir / "total_count", [img_c, img_h, img_w], np.uint8).data
            self.current_mean_prob = create_mmap_array(self.out_dir / "mean_prob", [img_c, img_h, img_w], float).data
            self.thresholded_prob = create_mmap_array(self.out_dir / "thresholded_prob", [img_c, img_h, img_w], bool).data
        else:
            assert self.current_folder == folder, f"can't handle more than one folder: {self.current_folder=}, {folder=}"

        self.current_mean_prob[lc: uc, ly: uy, lx: ux] += pred
        self.current_total_count[lc: uc, ly: uy, lx: ux] += 1

        # print(f"{self.chunk_boundary} wrote")
        return False

    def finalise(self):
        self._finalise_image_if_holding_any()


@dataclass
class ParallelizationSettings:
    run_as_single_process: bool = False
    n_chunks: int = 5
    finalise_one_by_one: bool = True


@dataclass
class SubmissionOutput:
    submission_df: pd.DataFrame


@profile
def generate_submission_df(
        model: Base3DSegmentor,
        data_loader: DataLoader,
        threshold: float,
        parallelization_settings: ParallelizationSettings,
        out_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        save_sub: bool = True,
) -> SubmissionOutput:
    ps = parallelization_settings
    if ps.run_as_single_process:
        data_queues = [MockQueue() for _ in range(ps.n_chunks)]
    else:
        mp.set_start_method("spawn", force=True)
        data_queues = [mp.Queue(maxsize=10) for _ in range(ps.n_chunks)]

    assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
        f"to generate submission, dataset must be ThreeDSegmentationDataset"
    dataset: ThreeDSegmentationDataset = data_loader.dataset
    n_channels = dataset.dataset.general_metadata["img_c"]
    distributor = TensorDistributor(n_channels, ps.n_chunks)

    model = model.eval().to(device)

    if out_dir is None:
        out_dir = TMP_SUB_MMAP_DIR / f"sennet_tmp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "image_names").write_text("\n".join(dataset.dataset.image_paths))
    tensor_receiving_processes = [
        TensorReceivingProcess(
            data_queue=q,
            threshold=threshold,
            # all_image_paths=dataset.dataset.image_paths,
            out_dir=out_dir / f"chunk_{str(i).zfill(2)}",
            chunk_boundary=cb
        )
        for i, (q, cb) in enumerate(zip(data_queues, distributor.chunk_boundaries))
    ]
    if ps.run_as_single_process:
        saver_processes = tensor_receiving_processes
    else:
        saver_processes = [mp.Process(
            target=p.spin,
        ) for p in tensor_receiving_processes]
    for s in saver_processes:
        s.start()

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            seg_pred: SegmentorOutput = model.predict(batch["img"].to(device))
            raw_pred_batch = seg_pred.pred
            un_reshaped_pred_batch = torch.nn.functional.sigmoid(raw_pred_batch)

            # reshape pred to original image size for submission
            _, _, img_d, img_h, img_w = batch["img"].shape
            pred_batch = resize_3d_image(un_reshaped_pred_batch.unsqueeze(1), (img_w, img_h, img_d))[:, 0, :, :, :]
            if "gt_seg_map" in batch:
                batch.pop("gt_seg_map")

            batch.pop("img")  # no need to ship image across process
            for i, pred in enumerate(pred_batch):
                # pred = (c, h, w)
                bbox_type = batch["bbox_type"][i].cpu().item()

                if bbox_type == DEPTH_ALONG_CHANNEL:
                    chunked_tensors = distributor.distribute_tensor(
                        pred,
                        batch["bbox"][i, 0] + seg_pred.take_indices_start,
                        batch["bbox"][i, 0] + seg_pred.take_indices_end,
                    )
                elif bbox_type == DEPTH_ALONG_HEIGHT:
                    permuted_pred = torch.concat([
                        pred[c, :, :].unsqueeze(1)
                        for c in range(pred.shape[0])
                    ], dim=1)
                    chunked_tensors = distributor.distribute_tensor(
                        permuted_pred,
                        batch["bbox"][i, 0],
                        batch["bbox"][i, 3],
                    )
                elif bbox_type == DEPTH_ALONG_WIDTH:
                    permuted_pred = torch.concat([
                        pred[c, :, :].unsqueeze(2)
                        for c in range(pred.shape[0])
                    ], dim=2)
                    chunked_tensors = distributor.distribute_tensor(
                        permuted_pred,
                        batch["bbox"][i, 0],
                        batch["bbox"][i, 3],
                    )
                else:
                    raise RuntimeError(f"unknown {bbox_type=}")

                for q, c in zip(data_queues, chunked_tensors):
                    if c is not None:
                        q.put(Data(
                            idx=i,
                            pred=c,
                            batch=batch,
                            bbox_type=bbox_type,
                            model_start_idx=seg_pred.take_indices_start,
                            model_end_idx=seg_pred.take_indices_end,
                        ))
                        if ps.run_as_single_process:
                            for _q, s in zip(data_queues, saver_processes):
                                if _q.has_val:
                                    s.spin_once()

    if ps.finalise_one_by_one:
        # trigger stopping one-by-one to avoid ram exploding
        for q, s in zip(data_queues, saver_processes):
            q.put(Data(terminate=True))
            s.join()
    else:
        # all at once for speed up
        for q in data_queues:
            q.put(Data(terminate=True))
        for s in saver_processes:
            s.join()

    df_out = generate_submission_df_from_one_chunked_inference(out_dir)
    if save_sub:
        df_out.to_csv(out_dir / "submission.csv")
    return SubmissionOutput(
        submission_df=df_out,
    )


if __name__ == "__main__":
    # from sennet.custom_modules.models import UNet3D
    from sennet.custom_modules.models import WrappedUNet3D

    _crop_size = 512
    _n_take_channels = 12
    _ds = ThreeDSegmentationDataset(
        "kidney_1_dense",
        crop_size=_crop_size,
        n_take_channels=_n_take_channels,
        output_crop_size=_crop_size,
        substride=1.0,
        sample_with_mask=True,
    )
    _dl = DataLoader(
        _ds,
        batch_size=2,
        shuffle=True,   # deliberate
        pin_memory=True,
    )
    _model = WrappedUNet3D(in_channels=1, out_channels=1, f_maps=32, is_segmentation=False)
    _sub = generate_submission_df(
        _model,
        _dl,
        0.5,
        ParallelizationSettings(
            run_as_single_process=False,
            # run_as_single_process=True,
            n_chunks=5,
        ),
        device="cuda",
    )
    print(_sub.submission_df.shape)
