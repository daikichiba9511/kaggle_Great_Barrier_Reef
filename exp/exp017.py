"""exp017

mmdetection cascade rcnn 50
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

import mmcv
import mmdet
import numpy as np
import pandas as pd
from box import Box
from genericpath import exists
from matplotlib.colors import to_rgb
from mercantile import parent
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]

config = Box(
    dict(
        seed=42,
        out_img_dir=str(ROOT / "./output/datasets/images"),
        out_lbl_dir=str(ROOT / "./output/datasets/labels"),
        images_dir=str(ROOT / "./images"),
        train_ann_path=str(ROOT / "./annotations_train.json"),
        val_ann_path=str(ROOT / "./annotations_val.json"),
        remove_nobbox=True,
        bg_rate=0.12,
    )
)

set_random_seed(config.seed, deterministic=False)

images_path = Path("./images")
images_path.mkdir(parents=True, exist_ok=True)
extract = False
if extract:
    with zipfile.ZipFile("./input/simple-yolox-dataset-generator-coco-json/train2017.zip", "r") as zip_ref:
        zip_ref.extractall(str(images_path))

    with zipfile.ZipFile("./input/simple-yolox-dataset-generator-coco-json/val2017.zip", "r") as zip_ref:
        zip_ref.extractall(str(images_path))

with open("./labels.txt", "w") as f:
    f.write("cots")


def get_df(config, data_dir: str = "./input/tensorflow-great-barrier-reef") -> pd.DataFrame:
    df = pd.read_csv(str(ROOT / "input" / "cross-validation" / "train-10folds.csv"))
    df.loc[:, ["old_img_path"]] = df["old_image_path"] = (
        f"{data_dir}/train_images/video_" + df["video_id"].astype(str) + "/" + df["video_frame"].astype(str) + ".jpg"
    )
    df.loc[:, ["image_path"]] = f"{Path(config['out_img_dir']).resolve()}/" + df["image_id"] + ".jpg"
    df.loc[:, ["label_path"]] = f"{Path(config['out_lbl_dir']).resolve()}/" + df["image_id"] + ".txt"
    df.loc[:, ["annotations"]] = df["annotations"].map(eval)
    return df


def coco(df):
    """
    Ref:
    [1] https://www.kaggle.com/coldfir3/simple-yolox-dataset-generator-coco-json
    """
    annotion_id = 0
    images = []
    annotations = []

    categories = [{"id": 0, "name": "cots"}]
    img_ids = df["image_id"].values
    annos = df["annotations"].values

    for idx, (img_id, bboxes) in enumerate(tqdm(zip(img_ids, annos))):
        images.append(
            {
                "id": idx,
                "file_name": f"{img_id}.jpg",
                "height": 720,
                "width": 1280,
            }
        )
        for bbox in bboxes:
            annotations.append(
                {
                    "id": annotion_id,
                    "image_id": idx,
                    "category_id": 0,
                    "bbox": list(bbox.values()),
                    "area": bbox["width"] * bbox["height"],
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotion_id += 1

    json_file = {"categories": categories, "images": images, "annotations": annotations}
    return json_file


def make_datasets(df, fold, config):
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]

    train_jsons = coco(train_df)
    val_jsons = coco(val_df)

    with Path(config.train_ann_path).open("w", encoding="utf-8") as f:
        json.dump(train_jsons, f, ensure_ascii=True, indent=4)
    with Path(config.val_ann_path).open("w", encoding="utf-8") as f:
        json.dump(val_jsons, f, ensure_ascii=True, indent=4)

    def copy_imgs(df, mode: str = "train"):
        assert mode in {"train", "val"}
        dst_path = Path(config.images_dir) / mode
        dst_path.mkdir(exist_ok=True, parents=True)
        img_ids = df["image_id"].values
        img_paths = df["image_path"].values
        for img_id, img_path in tqdm(zip(img_ids, img_paths)):
            dst_file_path = dst_path / f"{img_id}.jpg"
            shutil.copyfile(img_path, dst_file_path)

    copy_imgs(train_df, "train")
    copy_imgs(val_df, "val")


df = get_df(config)
make_datasets(df, fold=0, config=config)

df.loc[:, ["num_bbox"]] = df["annotations"].map(lambda x: len(x))
data = (df.num_bbox > 0).value_counts(normalize=True) * 100
print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")
raw_df = df
if config.remove_nobbox:
    df = raw_df.query("num_bbox>0")
    print("shape of df which is num_bbox>0: ", df.shape)
    if config.bg_rate:
        bg_size = int(df.shape[0] * config.bg_rate)
        sampled_bg_df = raw_df.query("num_bbox==0").sample(bg_size, random_state=42)
        df = pd.concat([df, sampled_bg_df], axis=0)
        print(f"shape of df which is num_bbox>0 and including num_bbox==0 with {bg_size} :", df.shape)

# generate config file for mmdetection
config_contents = """

_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

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
    max_epochs=30
)

"""

expname = Path(__file__).stem
config_path = ROOT / f"./mmdetection/configs/cascade_rcnn/TFGBR_{expname}_cascade_rcnn_r50_fpn_1x_coco.py"
with config_path.open("w") as f:
    f.write(config_contents)

cfg = Config.fromfile(str(config_path))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        # (4890, 1333),
                        # (512, 1333),
                        # (544, 1333),
                        # (608, 1333),
                        # (640, 1333),
                        # (672, 1333),
                        # (704, 1333),
                        # (736, 1333),
                        # (768, 1333),
                        (800, 1333),
                        (3600, 3600),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                # dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                # dict(
                #     type="Resize",
                #     img_scale=[
                #         (4890, 1333),
                #         (512, 1333),
                #         (544, 1333),
                #         (608, 1333),
                #         (640, 1333),
                #         (672, 1333),
                #         (704, 1333),
                #         (736, 1333),
                #         (768, 1333),
                #         (800, 1333),
                #     ],
                #     multiscale_mode="value",
                #     override=True,
                #     keep_ratio=True,
                # ),
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
    # Not Found at CocoDataset...
    # dict(type="Blur", blur_limit=3, p=0.05),
    # dict(type="MedianBlur", blur_limit=3, p=1.0),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(3600, 3600),
        flip=False,
        transforms=[
            dict(
                type="Resize",
                keep_ratio=True,
            ),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


cfg.classes = str(ROOT / "./labels.txt")
cfg.work_dir = str(Path("./").resolve())
cfg.data_root = str(ROOT / "./input")

train_ann_path = ROOT / "./annotations_train.json"
valid_ann_path = ROOT / "./annotations_val.json"

train_imgs_path = ROOT / "./images/train"
valid_imgs_path = ROOT / "./images/val"

# check dirs
for path in [train_ann_path, valid_ann_path, train_imgs_path, valid_imgs_path]:
    assert path.exists(), f"{path} does not exist"

cfg.data.train.type = "CocoDataset"
cfg.data.train.ann_file = str(train_ann_path)
cfg.data.train.img_prefix = str(train_imgs_path)
cfg.data.train.classes = str(ROOT / "labels.txt")

cfg.data.val.type = "CocoDataset"
cfg.data.val.ann_file = str(valid_ann_path)
cfg.data.val.img_prefix = str(valid_imgs_path)
cfg.data.val.classes = str(ROOT / "labels.txt")

cfg.data.test.type = "CocoDataset"
cfg.data.test.classes = str(ROOT / "./labels.txt")
cfg.data.test.ann_file = str(valid_ann_path)
cfg.data.test.img_prefix = str(valid_imgs_path)

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2

cfg.train_pipeline = train_pipeline
cfg.val_pipeline = test_pipeline
cfg.test_pipeline = test_pipeline

cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

cfg.lr_config = dict(
    policy="CosineAnnealing", by_epoch=True, warmup="linear", warmup_iters=1000, warmup_ratio=1 / 10, min_lr=1e-7
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
