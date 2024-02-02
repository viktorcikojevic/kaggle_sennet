from sennet.core.dataset import ThreeDSegmentationDataset
from sennet.core.submission_utils import generate_submission_df_from_one_chunked_inference
from sennet.core.mmap_arrays import create_mmap_array
from sennet.custom_modules.models import Base3DSegmentor, SegmentorOutput
from sennet.environments.constants import TMP_SUB_MMAP_DIR
from sennet.core.utils import resize_3d_image, DEPTH_ALONG_WIDTH, DEPTH_ALONG_HEIGHT, DEPTH_ALONG_CHANNEL
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
from typing import Callable


@dataclass
class Data:
    terminate: bool = False
    idx: int | None = None
    pred: torch.Tensor | None = None
    batch: dict[str, torch.Tensor] | None = None
    bbox_type: int | None = None
    model_start_idx: int | None = None
    model_end_idx: int | None = None


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
            out_dir: str | Path | None = None,
            cropping_border: int = 0,
            percentile_threshold: float | None = None,
            keep_in_memory: bool = False,
            num_models: int = 1,
            device: str = "cuda",
    ):
        self.data_queue = data_queue
        self.threshold = threshold
        self.percentile_threshold = percentile_threshold
        self.out_dir = out_dir

        self.current_total_count = None
        self.current_total_prob = None
        self.current_mean_prob = None
        self.num_models = num_models

        self.current_folder = None
        self.cropping_border = cropping_border
        self.keep_in_memory = keep_in_memory
        self.device = device

    @profile
    def _finalise_image_if_holding_any(self):
        current_mean_prob_is_none = self.current_mean_prob is None
        current_total_prob_is_none = self.current_total_prob is None
        current_total_count_is_none = self.current_total_count is None
        if (
                current_mean_prob_is_none
                or current_total_prob_is_none
                or current_total_count_is_none
        ):
            print(f"WARNING: {current_mean_prob_is_none=}, {current_total_prob_is_none=}, {current_total_count_is_none=}, returning early")
            return

        for c in tqdm(range(self.current_mean_prob.shape[0]), desc="computing mean"):
            self.current_mean_prob[c, ...] = self.current_total_prob[c, ...] / (self.current_total_count[c, ...] + 1e-8) / self.num_models

        if self.keep_in_memory:
            self.current_total_prob[:] = 0.0
            self.current_total_count[:] = 0
            return

        if self.current_folder is not None and self.current_mean_prob is not None and self.current_total_count is not None:
            current_total_count = create_mmap_array(self.out_dir / "total_count", self.current_total_count.shape, np.uint8).data
            current_mean_prob = create_mmap_array(self.out_dir / "mean_prob", self.current_mean_prob.shape, float).data
            thresholded_prob = create_mmap_array(self.out_dir / "thresholded_prob", self.current_mean_prob.shape, bool).data

            for c in tqdm(range(self.current_mean_prob.shape[0]), desc="moving to np"):
                _mean_prob_slice = self.current_mean_prob[c, ...].cpu().numpy()
                _total_count_slice = self.current_total_count[c, ...].cpu().numpy()
                current_mean_prob[c, ...] = _mean_prob_slice
                current_total_count[c, ...] = _total_count_slice
                thresholded_prob[c, ...] = current_mean_prob[c, ...] > self.threshold  # revive percentile threshold?

            print(f"flushing mean prob")
            current_mean_prob.flush()

            print(f"flushing total count")
            current_total_count.flush()

            print(f"flushing thresholded prob")
            thresholded_prob.flush()

            self.current_total_prob[:] = 0.0
            self.current_total_count[:] = 0
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
        pass

    @profile
    def spin_once(self) -> bool:
        data: Data = self.data_queue.get()
        i = data.idx
        pred = data.pred
        batch = data.batch
        terminate = data.terminate
        bbox_type = data.bbox_type

        if terminate:
            return True

        assert pred is not None, f"pred can't be None if terminate is False"
        assert bbox_type is not None, f"bbox_type be None if terminate is False"
        assert data.model_start_idx is not None, f"data.model_start_idx be None if terminate is False"
        assert data.model_end_idx is not None, f"data.model_end_idx be None if terminate is False"
        bbox = batch["bbox"][i].cpu().numpy()
        folder = batch["folder"][i]

        if bbox_type == DEPTH_ALONG_CHANNEL:
            lc = bbox[0] + data.model_start_idx
            uc = bbox[0] + data.model_end_idx
            lx = bbox[1] + self.cropping_border
            ux = bbox[4] - self.cropping_border
            ly = bbox[2] + self.cropping_border
            uy = bbox[5] - self.cropping_border
            cropped_pred = pred[
                :,
                self.cropping_border: pred.shape[1]-self.cropping_border,
                self.cropping_border: pred.shape[2]-self.cropping_border
            ]
        elif bbox_type == DEPTH_ALONG_WIDTH:
            lc = bbox[0] + self.cropping_border
            uc = bbox[3] - self.cropping_border
            lx = bbox[1] + data.model_start_idx
            ux = bbox[1] + data.model_end_idx
            ly = bbox[2] + self.cropping_border
            uy = bbox[5] - self.cropping_border
            cropped_pred = pred[
                self.cropping_border: pred.shape[0]-self.cropping_border,
                self.cropping_border: pred.shape[1]-self.cropping_border
                :,
            ]
        elif bbox_type == DEPTH_ALONG_HEIGHT:
            lc = bbox[0] + self.cropping_border
            uc = bbox[3] - self.cropping_border
            lx = bbox[1] + self.cropping_border
            ux = bbox[4] - self.cropping_border
            ly = bbox[2] + data.model_start_idx
            uy = bbox[2] + data.model_end_idx
            cropped_pred = pred[
                self.cropping_border: pred.shape[0]-self.cropping_border,
                :,
                self.cropping_border: pred.shape[2]-self.cropping_border
            ]
        else:
            raise RuntimeError(f"unknown bbox type at {self.__class__.__name__}: {bbox_type=}")

        if self.current_folder is None:
            img_h = int(batch["img_h"][i])
            img_w = int(batch["img_w"][i])
            img_c = int(batch["img_c"][i])

            self.current_folder = folder
            self.current_total_count = torch.zeros((img_c, img_h, img_w), dtype=torch.uint8, device=self.device)
            self.current_total_prob = torch.zeros((img_c, img_h, img_w), dtype=torch.float16, device=self.device)
            self.current_mean_prob = torch.zeros((img_c, img_h, img_w), dtype=torch.float16, device=self.device)
        else:
            assert self.current_folder == folder, f"can't handle more than one folder: {self.current_folder=}, {folder=}"

        self.current_total_prob[lc: uc, ly: uy, lx: ux] += cropped_pred.to(self.device)
        self.current_total_count[lc: uc, ly: uy, lx: ux] += 1

        # print(f"{self.chunk_boundary} wrote")
        return False

    def finalise(self):
        self._finalise_image_if_holding_any()


@dataclass
class ParallelizationSettings:
    run_as_single_process: bool = False


@dataclass
class SubmissionOutput:
    submission_df: pd.DataFrame
    mean_pred: np.ndarray | None = None
    thresholded_pred: np.ndarray | None = None


# very sorry for the god so many types this function takes
@torch.autocast(device_type="cuda")
@torch.no_grad()
@profile
def generate_submission_df(
        model: Base3DSegmentor | None,
        data_loader: DataLoader | None,
        threshold: float,
        parallelization_settings: ParallelizationSettings,
        out_dir: str | Path | None = None,
        device: str = "cuda",
        save_sub: bool = True,
        percentile_threshold: float | None = None,
        keep_in_memory: bool = False,
        model_and_data_loader_factory: None | list[Callable[[], tuple[Base3DSegmentor, DataLoader]]] = None,  # function to generate model and dataloader
        data_device: str | None = None,
) -> SubmissionOutput:
    ps = parallelization_settings
    if data_device is None:
        data_device = device
    if keep_in_memory:
        assert ps.run_as_single_process, f"keep in memory needs to be run as a single process"
    if ps.run_as_single_process:
        q = MockQueue()
    else:
        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=10)

    if model_and_data_loader_factory is None:
        model_and_data_loader_factory = [lambda: (model, data_loader)]

    if out_dir is None:
        out_dir = TMP_SUB_MMAP_DIR / f"sennet_tmp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    out_dir.mkdir(exist_ok=True, parents=True)
    tensor_receiving_process = TensorReceivingProcess(
        data_queue=q,
        threshold=threshold,
        percentile_threshold=percentile_threshold,
        # all_image_paths=dataset.dataset.image_paths,
        out_dir=out_dir / f"chunk_00",
        cropping_border=0,
        keep_in_memory=keep_in_memory,
        num_models=len(model_and_data_loader_factory),
        device=data_device,
    )
    if ps.run_as_single_process:
        saver_process = tensor_receiving_process
    else:
        saver_process = mp.Process(
            target=tensor_receiving_process.spin,
        )
    saver_process.start()

    for i_model, factory in enumerate(model_and_data_loader_factory):
        print(f"running model [{i_model+1}/{len(model_and_data_loader_factory)}]")
        model, data_loader = factory()
        model = model.eval().to(device)

        assert isinstance(data_loader.dataset, ThreeDSegmentationDataset), \
            f"to generate submission, dataset[{i_model}] must be ThreeDSegmentationDataset"
        dataset: ThreeDSegmentationDataset = data_loader.dataset
        cropping_border = dataset.dataset.cropping_border
        (out_dir / "image_names").write_text("\n".join(dataset.dataset.image_paths))
        tensor_receiving_process.cropping_border = cropping_border

        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_img = batch["img"].to(device)
            seg_pred: SegmentorOutput = model.predict(batch_img)
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
                    permuted_pred = pred
                elif bbox_type == DEPTH_ALONG_HEIGHT:
                    # permuted_pred = torch.concat([
                    #     pred[c, :, :].unsqueeze(1)
                    #     for c in range(pred.shape[0])
                    # ], dim=1)
                    # equivalent, checked
                    permuted_pred = pred.permute((1, 0, 2))
                elif bbox_type == DEPTH_ALONG_WIDTH:
                    # permuted_pred = torch.concat([
                    #     pred[c, :, :].unsqueeze(2)
                    #     for c in range(pred.shape[0])
                    # ], dim=2)
                    # equivalent, checked
                    permuted_pred = pred.permute((1, 2, 0))
                else:
                    raise RuntimeError(f"unknown {bbox_type=}")
                q.put(Data(
                    idx=i,
                    pred=permuted_pred,
                    batch=batch,
                    bbox_type=bbox_type,
                    model_start_idx=seg_pred.take_indices_start,
                    model_end_idx=seg_pred.take_indices_end,
                ))
                if ps.run_as_single_process:
                    if q.has_val:
                        saver_process.spin_once()

    q.put(Data(terminate=True))
    saver_process.join()
    df_out = generate_submission_df_from_one_chunked_inference(out_dir)
    if save_sub:
        df_out.to_csv(out_dir / "submission.csv")
    res = SubmissionOutput(submission_df=df_out)
    if keep_in_memory and tensor_receiving_process.current_mean_prob is not None:
        res.mean_pred = tensor_receiving_process.current_mean_prob.cpu().numpy()
        res.thresholded_pred = np.zeros_like(res.mean_pred, dtype=bool)
        for c in tqdm(range(res.mean_pred.shape[0]), desc="keep_in_memory ret"):
            res.thresholded_pred[c, ...] = res.mean_pred[c, ...] > threshold
    return res


if __name__ == "__main__":
    # from sennet.custom_modules.models import UNet3D
    from sennet.custom_modules.models import SMPModel

    _crop_size = 512 - 32
    _n_take_channels = 1
    _ds = ThreeDSegmentationDataset(
        "kidney_3_dense",
        crop_size=_crop_size,
        n_take_channels=_n_take_channels,
        output_crop_size=_crop_size,
        substride=1.0,
        cropping_border=16,
        sample_with_mask=False,
        add_depth_along_width=True,
        add_depth_along_height=True,
        add_depth_along_channel=True,
    )
    _dl = DataLoader(
        _ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
    )
    _model = SMPModel(
        version="Unet",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).eval()
    _sub = generate_submission_df(
        _model,
        _dl,
        0.5,
        ParallelizationSettings(
            # run_as_single_process=False,
            run_as_single_process=True,
        ),
        device="cuda",
        out_dir=TMP_SUB_MMAP_DIR / "debug_from_pycharm"
    )
    print(_sub.submission_df.shape)
