# =======================================================
# ================= hyper-parameters ====================
# =======================================================
project_name = "sennet_naive"

batch_size = "{{$MMSEG_BATCH_SIZE:4}}"
batch_size = int(batch_size)

crop_size = 512
output_crop_size = 512

# ------------------ data-parameters --------------------
train_folders = [
    "kidney_1_dense",
]
val_folders = [
    "kidney_3_sparse",
]
n_take_channels = 3
# -------------------------------------------------------

# ------------------- aug-parameters --------------------
train_crop_size_range = [int(0.5*crop_size), int(1.5*crop_size)]
val_crop_size_range = [crop_size, crop_size]
crop_location_noise = 10
channels_jitter = 1
p_channel_jitter = 0.5
use_mosaic = False
train_dataset_stride = 20
val_dataset_stride = 10
# -------------------------------------------------------

# ------------------ optim parameters -------------------
warmup_iters = 1000
# -------------------------------------------------------

# ----------------- boring parameters -------------------
reduce_zero_label = False
seg_pad_val = 255 if reduce_zero_label else 0
num_train_workers = 4
num_val_workers = 2
meta_keys = [
    "img_idx", "row_idx", "col_idx", "total_rows", "total_cols",
    "ori_shape", "img_shape", "pad_shape", "mean", "std",
    "scale_factor", "flip", "flip_direction", "reduce_zero_label",
    "pixel_x", "pixel_y", "full_width", "full_height", "scaled_width", "scaled_height",
    "crop_bbox", "name",
]
scheduler_max_iters = 50000
max_iters = "{{$MMSEG_MAX_ITER:25000}}"
max_iters = int(max_iters)
eval_every = 1000

# =======================================================
# ================= dataset settings ====================
# =======================================================
# batch_train_pipeline = [
#     # dict(
#     #     type="RandomMosaic",
#     #     prob=1.0 if use_mosaic else 0.0,
#     #     img_scale=(output_crop_size, output_crop_size),
#     #     center_ratio_range=(0.5, 1.5),
#     #     pad_val=0,
#     #     seg_pad_val=seg_pad_val,
#     # ),
#     # dict(
#     #     type="RandomCutOut",
#     #     prob=0.5,
#     #     n_holes=(1, 3),
#     #     cutout_ratio=[
#     #         (0.25, 0.25),
#     #         (0.50, 0.25),
#     #         (0.25, 0.50),
#     #     ],
#     #     fill_in=[0.0]*n_take_channels,
#     #     seg_fill_in=seg_pad_val,
#     # ),
#     # dict(
#     #     type="Palelify",
#     #     prob=0.25,
#     #     pale_ratio=(0.3, 0.5),
#     # ),
#     # dict(
#     #     type="RandomRotate",
#     #     prob=0.5,
#     #     seg_pad_val=seg_pad_val,
#     #     degree=(-180.0, 180.0),
#     # ),
#     # dict(type="RandomFlip", prob=0.5),
#     # dict(
#     #     type="RandomFlip",
#     #     direction="horizontal",
#     #     prob=0.5
#     # ),
#     # dict(
#     #     type="RandomFlip",
#     #     direction="vertical",
#     #     prob=0.5
#     # ),
#     # dict(
#     #     type="RandomBrightnessContrast",
#     #     brightness_limit=0.2,
#     #     contrast_limit=0.2,
#     #     prob=0.25,
#     # ),
#     dict(
#         type="Albu",
#         transforms=[
#             dict(
#                 type="GridDistortion",
#                 num_steps=5,
#                 distort_limit=0.3,
#                 border_mode=0,
#                 value=0,
#                 mask_value=seg_pad_val,
#                 p=0.5,
#             ),
#             # dict(
#             #     type="OneOf",
#             #     p=0.25,
#             #     transforms=[
#             #         dict(type="GaussianBlur", p=1.0),
#             #         dict(type="MotionBlur", p=1.0),
#             #     ],
#             # ),
#         ]
#     ),
#     # dict(type="GaussianNoise", prob=0.5, mean=0, std=0.1),
#     dict(type="PackSegInputs", meta_keys=meta_keys)
# ]
train_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=train_crop_size_range,
        output_crop_size=output_crop_size,
        load_ann=True,
        seg_fill_val=seg_pad_val,
        channels_jitter=channels_jitter,
        p_channel_jitter=p_channel_jitter,
        crop_location_noise=crop_location_noise,
    ),
    dict(type="PackSegInputs", meta_keys=meta_keys)
]
val_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=val_crop_size_range,
        output_crop_size=output_crop_size,
        load_ann=True,
        seg_fill_val=seg_pad_val,
    ),
    dict(type="PackSegInputs", meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type="LoadMultiChannelImageAndAnnotationsFromFile",
        crop_size_range=val_crop_size_range,
        output_crop_size=output_crop_size,
        load_ann=True,
        seg_fill_val=seg_pad_val,
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
        type="MultiChannelDataset",
        folders=train_folders,
        crop_size=crop_size,
        channel_start=1000,
        n_take_channels=n_take_channels,
        reduce_zero_label=reduce_zero_label,
        assert_label_exists=True,
        stride=train_dataset_stride,
        pipeline=train_pipeline,
    ),
    # dataset=dict(
    #     type="MultiImageMixDataset",
    #     pipeline=batch_train_pipeline,
    #     dataset=dict(
    #         type="MultiChannelDataset",
    #         folders=train_folders,
    #         crop_size=crop_size,
    #         n_take_channels=n_take_channels,
    #         reduce_zero_label=reduce_zero_label,
    #         assert_label_exists=True,
    #         stride=train_dataset_stride,
    #         pipeline=train_pipeline,
    #     ),
    # ),
)
val_dataloader = dict(
    batch_size=1,  # they can't do batch inference
    num_workers=num_val_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MultiChannelDataset",
        folders=val_folders,
        crop_size=crop_size,
        n_take_channels=n_take_channels,
        reduce_zero_label=reduce_zero_label,
        assert_label_exists=True,
        stride=val_dataset_stride,
        pipeline=val_pipeline,
    )
)
test_dataloader = dict(
    batch_size=1,  # they can't do batch inference
    num_workers=num_val_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MultiChannelDataset",
        folders=val_folders,
        crop_size=crop_size,
        n_take_channels=n_take_channels,
        reduce_zero_label=reduce_zero_label,
        assert_label_exists=True,
        stride=val_dataset_stride,
        pipeline=test_pipeline,
    )
)
val_evaluator = dict(
    prefix="seg",
    type="SpecificClassIouMetric",
    beta=0.5,
    class_of_interest=1,
    iou_metrics=["mIoU", "mFscore"]  # TODO(Sumo): add your surface dice here
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
        save_best="seg/mFscore_class_1",  # TODO(Sumo): take best val surface dice
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
            project=project_name,
            tags=[],
            config=dict(),
            notes="",
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
