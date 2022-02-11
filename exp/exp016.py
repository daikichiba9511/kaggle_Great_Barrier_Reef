"""exp016

mmdetection cascade rcnn 50
"""
from pathlib import Path
from matplotlib.colors import to_rgb

import numpy as np
import os
import pandas as pd

import mmdet
import mmcv

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, set_random_seed

import zipfile

from box import Box


config = Box(dict(seed=42))

set_random_seed(config.seed, deterministic=False)

images_path = Path("./input/coco-images")
images_path.mkdir(parents=True, exist_ok=True)
extract = False
if extract:
    with zipfile.ZipFile("./input/simple-yolox-dataset-generator-coco-json/train2017.zip", "r") as zip_ref:
        zip_ref.extractall(str(images_path))

    with zipfile.ZipFile("./input/simple-yolox-dataset-generator-coco-json/val2017.zip", "r") as zip_ref:
        zip_ref.extractall(str(images_path))

with open("./labels.txt", "w") as f:
    f.write("cots")


# Model Config

config_content = """

_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2,2,18,2],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0,1,2,3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
        ),
    ),
    neck=dict(
        in_channels=[96,192,384,768]
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1,0.5,0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
            ),
            out_channels=256,
            featmap_strides=[4,8,16,32],
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.,],
                    target_stds=[0.1,0.1,0.2,0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
                reg_decoded_bbox=True,
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0,
                )
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.],
                    target_stds=[0.05,0.05,0.1,0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
                reg_decoded_bbox=True,
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.,
                )
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.],
                    target_stds=[0.033,0.033,0.067,0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0
                ),
                reg_decoded_bbox=True,
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0
                )
            )
        ]
    )
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0004,
    betas=(0.9,0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }
    )
)

lr_config=dict(
    warmup_iters=500,
    step=[8,11],
)

runner=dict(
    max_epochs=14
)
"""

config_path = "./mmdetection/configs/swin/TFGBR_swin_base_faster_rcnn_fp16.py"
with open(config_path, "w") as f:
    f.write(config_content)


cfg = Config.fromfile(config_path)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type="LoadImageFromFile",
        to_float32=True,
    ),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (4890, 1333),
                        (512, 1333),
                        (544, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                dict(
                    type="Resize",
                    img_scale=[
                        (4890, 1333),
                        (512, 1333),
                        (544, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
                dict(
                    type="PhotoMetricDistortion",
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18,
                ),
                dict(
                    type="CutOut",
                    n_holes=(5, 10),
                    cutout_shape=[
                        (4, 4),
                        (4, 8),
                        (8, 4),
                        (16, 32),
                        (32, 16),
                        (32, 32),
                        (16, 32),
                        (32, 16),
                        (32, 32),
                        (32, 48),
                        (48, 32),
                        (48, 48),
                    ],
                ),
            ]
        ],
    ),
    dict(
        type="RandomFlip",
        flip_ratio=0.5,
    ),
    dict(
        type="Normalize",
        **img_norm_cfg,
    ),
    dict(
        type="Pad",
        size_divisor=32,
    ),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True,),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

ROOT = Path(__file__).resolve().parents[1]

cfg.classes =  str(ROOT / "./labels.txt")
cfg.work_dir = str(ROOT)
cfg.data_root = str(ROOT / "./input")

cfg.data.test.type = "CocoDataset"
cfg.data.test.ann_file = str(ROOT / "./input/simple-yolox-dataset-generator-coco-json/annotaions_valid.json")
cfg.data.test.img_prefix = str(ROOT / "./input/coco-images/valid/images")
cfg.data.test.classes = str(ROOT / "./labels.txt")

cfg.data.train.type = "CocoDataset"
cfg.data.train.ann_file = str(ROOT / "./input/simple-yolox-dataset-generator-coco-json/annotations_train.json")
cfg.data.train.img_prefix = str(ROOT / "./input/coco-images/train/images")
cfg.data.train.classes = str(ROOT / "labels.txt")

cfg.data.val.type = "CocoDataset"
cfg.data.val.ann_file = str(ROOT / "./input/simple-yolox-dataset-generator-coco-json/annotations_valid.json")
cfg.data.val.img_prefix = str(ROOT / "./input/coco-images/valid/images")
cfg.data.val.classes = str(ROOT / "labels.txt")

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2

cfg.train_pipeline = train_pipeline
cfg.val_pipeline = test_pipeline
cfg.test_pipeline = test_pipeline

cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

cfg.lr_config = dict(
    policy="CosineAnnealing", by_epoch=False, warmup="linear", warmup_iters=1000, warmup_ratio=1 / 10, min_lr=1e-7
)

cfg.evaluation.interval = 2
cfg.evaluation.save_best = "auto"

cfg.seed = config.seed
cfg.gpu_ids = range(1)

cfg.fp16 = dict(loss_scale=dict(init_scale=512.0))

cfg.log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

meta = dict()
meta["config"] = cfg.pretty_text

datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.init_weights()
model.CLASSES = datasets[0].CLASSES


mmcv.mkdir_or_exist(str(Path(cfg.work_dir).resolve()))
train_detector(model, datasets, cfg, distributed=False, validate=True, meta=meta)
