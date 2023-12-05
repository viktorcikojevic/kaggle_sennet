# import cv2


# =======================================================
# ================= hyper-parameters ====================
# =======================================================
# target_scale = 0.5
# crop_size = 64
batch_size = "{{$MMSEG_BATCH_SIZE:4}}"
batch_size = int(batch_size)

target_scale_override = "{{$MMSEG_TARGET_SCALE:0}}"
if target_scale_override != "0":
    target_scale = float(target_scale_override)
else:
    target_scale = 1.0
crop_size = "{{$MMSEG_CROP_SIZE:896}}"
crop_size = int(crop_size)
out_crop_size_ratio = 1.0  # crop size will shrink by this ratio
channels_jitter = 1
p_channel_jitter = 0.5
normalise_img = True
train_dataset_stride = 3
crop_location_noise = 1
# train_dataset_stride = 10
val_dataset_stride = max(int(200 * target_scale), 1)
take_channel_override = "{{$MMSEG_TAKE_CHANNELS:0}}"
if take_channel_override == "NONE":
    take_channels = []
elif take_channel_override != "0":
    take_channels = [int(x) for x in take_channel_override.split(",")]
else:
    take_channels = list(range(32, 35, 1))
# take_channels = None
# NOTE: don't random resize to reduce the image size, the annotation will offset with the image
reduce_zero_label = False
use_mosaic = False
warmup_iters = 1000
scheduler_max_iters = 50000
max_iters = "{{$MMSEG_MAX_ITER:25000}}"
max_iters = int(max_iters)
train_fold_name = "{{$MMSEG_TRAIN_FOLD_NAME:fold_0}}"
val_fold_name = "{{$MMSEG_VAL_FOLD_NAME:fold_1}}"
additional_data_root = "{{$MMSEG_ADDITIONAL_DATA_ROOT:0}}"
num_additional_channels = "{{$MMSEG_NUM_ADDITIONAL_CHANNELS:0}}"
num_additional_channels = int(num_additional_channels)
use_box = False
sample_around_seg = False

# train_fold_name = "fold_1"
# val_fold_name = "fold_2"
# additional_data_root = "/home/clay/research/kaggle/vesuvius/data/data_dumps/models_predictions"
# num_additional_channels = 5
eval_every = 1000
# --- data switching here ----
# data_folder = "splitted_sanity_check_data"
# data_folder = "splitted_data"
data_folder = "splitted_data_per_img"
# ----------------------------
if additional_data_root != "0":
    train_additional_data_root = f"{additional_data_root}/{train_fold_name}"
    val_additional_data_root = f"{additional_data_root}/{val_fold_name}"
else:
    train_additional_data_root = ""
    val_additional_data_root = ""
data_root = f"/home/clay/research/kaggle/vesuvius/data"
tags = []
meta_keys = [
    "img_idx", "row_idx", "col_idx", "total_rows", "total_cols",
    "ori_shape", "img_shape", "pad_shape", "mean", "std",
    "scale_factor", "flip", "flip_direction", "reduce_zero_label",
    "pixel_x", "pixel_y", "full_width", "full_height", "scaled_width", "scaled_height",
    "crop_bbox", "name",
]
notes = "first model iteration on real data"
num_train_workers = 4
num_val_workers = 2
# =======================================================

# ---- inferred settings -----
train_data_root = f"{data_root}/{data_folder}/{train_fold_name}"
val_data_root = f"{data_root}/{data_folder}/{val_fold_name}"
test_data_root = val_data_root
num_classes = 1 if reduce_zero_label else 2
seg_pad_val = 255 if reduce_zero_label else 0
assert crop_size % 2 == 0, f"crop size needs to be divisible by 2 for mosaic"
out_crop_size = int(out_crop_size_ratio * crop_size)
val_out_crop_size = out_crop_size
train_out_crop_size = int(out_crop_size / 2) if use_mosaic else out_crop_size  # so mosaic can combine them into full size
val_crop_size_range = (crop_size, crop_size)
train_crop_size = int(crop_size / 2) if use_mosaic else int(crop_size)
val_crop_size = crop_size
train_crop_size_range = (
    # (int(0.8*train_crop_size), int(1.2*train_crop_size))
    (int(train_crop_size), int(train_crop_size))
)  # so mosaic can combine them into full size
class_weight = None if reduce_zero_label else [0.2, 1.0]

if take_channels is None:
    take_channels = list(range(65))
num_channels = len(take_channels)
num_channels += num_additional_channels
print(f"{num_channels = } ({len(take_channels)} + {num_additional_channels})")

# =======================================================
# ================= dataset settings ====================
# =======================================================
batch_train_pipeline = [
    dict(
        type="RandomMosaic",
        prob=1.0 if use_mosaic else 0.0,
        img_scale=(train_out_crop_size, train_out_crop_size),
        center_ratio_range=(0.5, 1.5),
        pad_val=0,
        seg_pad_val=seg_pad_val,
    ),
    dict(
        type="RandomCutOut",
        prob=0.5,
        n_holes=(1, 3),
        cutout_ratio=[
            (0.25, 0.25),
            (0.50, 0.25),
            (0.25, 0.50),
        ],
        fill_in=[0.0]*num_channels,
        seg_fill_in=seg_pad_val,
    ),
    # dict(
    #     type="Palelify",
    #     prob=0.25,
    #     pale_ratio=(0.3, 0.5),
    # ),
    dict(
        type="RandomRotate",
        prob=0.5,
        seg_pad_val=seg_pad_val,
        degree=(-180.0, 180.0),
    ),
    dict(type="RandomFlip", prob=0.5),
    # dict(
    #     type="RandomFlip",
    #     direction="horizontal",
    #     prob=0.5
    # ),
    # dict(
    #     type="RandomFlip",
    #     direction="vertical",
    #     prob=0.5
    # ),
    # dict(
    #     type="RandomBrightnessContrast",
    #     brightness_limit=0.2,
    #     contrast_limit=0.2,
    #     prob=0.25,
    # ),
    dict(
        type="Albu",
        transforms=[
            dict(
                type="GridDistortion",
                num_steps=5,
                distort_limit=0.3,
                border_mode=0,
                value=0,
                mask_value=seg_pad_val,
                p=0.5,
            ),
            # dict(
            #     type="OneOf",
            #     p=0.25,
            #     transforms=[
            #         dict(type="GaussianBlur", p=1.0),
            #         dict(type="MotionBlur", p=1.0),
            #     ],
            # ),
        ]
    ),
    # dict(type="GaussianNoise", prob=0.5, mean=0, std=0.1),
    dict(type="PackSegInputs", meta_keys=meta_keys)
]
train_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=train_crop_size_range,
        output_crop_size=train_out_crop_size,
        channels=take_channels,
        channels_jitter=channels_jitter,
        p_channel_jitter=p_channel_jitter,
        seg_fill_val=seg_pad_val,
        normalise=normalise_img,
        load_gt_boxes=True,
        crop_location_noise=crop_location_noise,
    ),
]
val_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=val_crop_size_range,
        output_crop_size=val_out_crop_size,
        seg_fill_val=seg_pad_val,
        channels=take_channels,
        normalise=normalise_img,
        load_gt_boxes=True,
    ),
    dict(type="PackSegInputs", meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=val_crop_size_range,
        output_crop_size=val_out_crop_size,
        load_ann=False,
        seg_fill_val=seg_pad_val,
        channels=take_channels,
        normalise=normalise_img,
        load_gt_boxes=True,
    ),
    dict(type="PackSegInputs", meta_keys=meta_keys)
]

dataset_type = "MultiChannelDataset"
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_train_workers,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    # sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="MultiImageMixDataset",
        pipeline=batch_train_pipeline,
        dataset=dict(
            type="PerPixelMultiChannelDataset",
            crop_size=train_crop_size,
            data_root=train_data_root,
            reduce_zero_label=reduce_zero_label,
            target_scale=target_scale,
            stride=train_dataset_stride,
            pipeline=train_pipeline,
            additional_data_root=train_additional_data_root,
            use_box=use_box,
            sample_around_seg=sample_around_seg,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,  # they can't do batch inference
    num_workers=num_val_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PerPixelMultiChannelDataset",
        crop_size=val_crop_size,
        data_root=val_data_root,
        reduce_zero_label=reduce_zero_label,
        target_scale=target_scale,
        stride=val_dataset_stride,
        pipeline=val_pipeline,
        additional_data_root=val_additional_data_root,
        use_box=use_box,
        sample_around_seg=sample_around_seg,
    )
)
test_dataloader = dict(
    batch_size=1,  # they can't do batch inference
    num_workers=num_val_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="PerPixelMultiChannelDataset",
        crop_size=val_crop_size,
        data_root=test_data_root,
        reduce_zero_label=reduce_zero_label,
        target_scale=target_scale,
        stride=val_dataset_stride,
        pipeline=test_pipeline,
        additional_data_root=val_additional_data_root,
        use_box=use_box,
        sample_around_seg=sample_around_seg,
    )
)
val_evaluator = dict(
    prefix="seg",
    type="SpecificClassIouMetric",
    beta=0.5,
    class_of_interest=1,
    iou_metrics=["mIoU", "mFscore"]
)
test_evaluator = val_evaluator
# =======================================================
# =======================================================
# =======================================================


# =======================================================
# ==================== runtimes =========================
# =======================================================
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
# training schedule for 160k
train_cfg = dict(
    type="IterBasedTrainLoop", max_iters=max_iters, val_interval=eval_every)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=eval_every,
        save_best="seg/mFscore_class_1",
        rule="greater",
        max_keep_ckpts=5,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook")
)
# custom_hooks = [
#     dict(
#         type="EMAHook",  # look for params in the ExponentialMovingAverage class
#         ema_type="ExponentialMovingAverage",
#         momentum=0.0002,
#         interval=1,
#     )
# ]


default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="vesuvius-seg",
            tags=tags,
            config=dict(
                data_folder=data_folder,
            ),
            notes=notes,
        ),
    ),
]
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    alpha=0.5,
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
resume = False

tta_model = dict(type="SegTTAModel")

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=warmup_iters
    ),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=warmup_iters,
        end=scheduler_max_iters,
        by_epoch=False,
    ),
    # dict(
    #     type="CosineAnnealingLR",
    #     eta_min=0.0,
    #     begin=warmup_iters,
    #     end=scheduler_max_iters,
    #     by_epoch=False,
    # ),
]
# =======================================================
# =======================================================
# =======================================================
