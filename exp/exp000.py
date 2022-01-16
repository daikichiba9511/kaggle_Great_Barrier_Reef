"""exp000

baseline

submission format is COCO means [x_min, y_min, width, height]

Come metrics `F2` tolerates some FP in order to ensure very few starfish missed

Ref:
[1] https://www.kaggle.com/daimarusui3/great-barrier-reef-yolov5-train/edit
"""

import subprocess

import numpy as np
import yaml
from tqdm.auto import tqdm

tqdm.pandas()
import glob
import os
import shutil
import sys
from pathlib import Path
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from box import Box

sys.path.append("../input/tensorflow-great-barrier-reef")

from bbox.utils import (
    annot2str,
    clip_bbox,
    coco2voc,
    coco2yolo,
    draw_bboxes,
    load_image,
    str2annot,
    voc2yolo,
)

# from IPython.display import HTML, display
from joblib import Parallel, delayed
from matplotlib import animation, rc

rc("animation", html="jshtml")


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


def get_imgsize(row, imagesize):
    row["width"], row["height"] = imagesize.get(row["image_path"])
    return row


file = __file__ or "../"
ROOT = Path(file).resolve().parents[1]
print(ROOT)

config = {
    "seed": 42,
    "train": True,
    "inference": False,
    "fold": 4,  # validation fold
    "n_splits": 5,
    "epochs": 25,
    "dim": 1280,
    "model": {"name": "yolov5m"},
    "batch_size": -1,  # if batch_size == 1, yolov5 trainer estimates batch_size
    "remove_nobbox": True,
    "data_dir": "./input/tensorflow-great-barrier-reef",
    "out_img_dir": "./output/datasets/images",
    "out_lbl_dir": "./output/datasets/labels",
}

config = Box(config)
np.random.seed(config.seed)
Path(config["out_img_dir"]).mkdir(parents=True, exist_ok=True)
Path(config["out_lbl_dir"]).mkdir(parents=True, exist_ok=True)


def get_df(
    config, data_dir: str = "./input/tensorflow-great-barrier-reef"
) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/train.csv")
    df.loc[:, ["old_img_path"]] = df["old_image_path"] = (
        f"{data_dir}/train_images/video_"
        + df.video_id.astype(str)
        + "/"
        + df.video_frame.astype(str)
        + ".jpg"
    )
    df.loc[:, ["image_path"]] = f"{Path(config['out_img_dir']).resolve()}/" + df.image_id + ".jpg"
    df.loc[:, ["label_path"]] = f"{Path(config['out_lbl_dir']).resolve()}/" + df.image_id + ".txt"
    df.loc[:, ["annotations"]] = df["annotations"].progress_apply(eval)
    print(df.head(2))
    return df


def make_copy(row):
    shutil.copyfile(row.old_image_path, row.image_path)
    return


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """convert format of COCO to one of YOLO.

    dataset format is COCO : [x_center, y_center,width, height]

    we need YOLO format : [x_min, y_min, width, height] to use yolov5 tainer.

    """
    cnt = 0
    all_bboxes = []
    bboxes_info = []
    for idx, row_idx in enumerate(tqdm(range(df.shape[0]))):
        row = df.iloc[row_idx]
        image_height = row.height
        image_width = row.width
        bboxes_coco = np.array(row.bboxes).astype(np.float32).copy()
        num_bbox = len(bboxes_coco)
        names = ["cots"] * num_bbox
        labels = np.array([0] * num_bbox)[..., None].astype(str)
        # Create Annotation(YOLO)
        with open(row.label_path, "w") as f:
            if num_bbox < 1:
                annot = ""
                f.write(annot)
                cnt += 1
                continue
            bboxes_voc = coco2voc(bboxes_coco, image_height, image_width)
            bboxes_voc = clip_bbox(bboxes_voc, image_height, image_width)
            bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
            all_bboxes.extend(bboxes_yolo.astype(float))
            bboxes_info.extend(
                [[row.image_id, row.video_id, row.sequence]] * len(bboxes_yolo)
            )
            annots = np.concatenate([labels, bboxes_yolo], axis=1)
            string = annot2str(annots)
            f.write(string)
        if idx % 2000 == 0:
            print(f"idx:{idx}, {row.label_path}")
            print(string)
    print("Missing:", cnt)
    return df, bboxes_info, all_bboxes


def create_fold(df: pd.DataFrame, config) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    kf = GroupKFold(n_splits=config.n_splits)
    df = df.reset_index(drop=True)
    df.loc[:, ["fold"]] = -1
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(df, y=df.video_id.tolist(), groups=df.sequence)
    ):
        df.loc[val_idx, "fold"] = fold
    print(df.fold.value_counts())
    return df


def bbox_distribution(df, bboxes_info, all_bboxes):
    bbox_df = pd.DataFrame(
        np.concatenate([bboxes_info, all_bboxes], axis=1),
        columns=["image_id", "video_id", "sequence", "xmid", "ymid", "w", "h"],
    )
    bbox_df.loc[:, ["xmid", "ymid", "w", "h"]] = bbox_df[
        ["xmid", "ymid", "w", "h"]
    ].astype(float)
    bbox_df.loc[:, ["area"]] = bbox_df.w * bbox_df.h * 1280 * 720
    bbox_df = bbox_df.merge(df[["image_id", "fold"]], on="image_id", how="left")
    return bbox_df


def create_dataset(df, config):
    fold = config.fold
    train_files = []
    val_files = []
    train_df = df.query(f"fold!={fold}")
    valid_df = df.query(f"fold=={fold}")
    train_files += list(train_df.image_path.unique())
    val_files += list(valid_df.image_path.unique())
    print(
        "train_files: " + str(len(train_files)), ", val_files: " + str(len(val_files))
    )
    return train_df, valid_df, train_files, val_files


def create_config(train_df: pd.DataFrame, valid_df: pd.DataFrame, cwd: Path) -> None:
    with (cwd / "train.txt").open("w") as f:
        for path in train_df["image_path"].tolist():
            f.write(path + "\n")

    with (cwd / "val.txt").open("w") as f:
        for path in valid_df["image_path"].tolist():
            f.write(path + "\n")

    data = dict(
        # dataset dir
        path="./",
        train=str(cwd / "train.txt"),
        val=str(cwd / "val.txt"),
        nc=1,  # the number of classes of ground truth
        names=["cots"],
    )
    print(" check data config")
    pprint(data)

    with (cwd / "gbr.yml").open("w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    with (cwd / "gbr.yml").open("r") as f:
        print("#" * 15)
        print("\n yaml: ")
        print(f.read())


def main(config):
    pprint(config)

    df = get_df(config)

    df.loc[:, ["num_bbox"]] = df["annotations"].progress_apply(lambda x: len(x))
    data = (df.num_bbox > 0).value_counts(normalize=True) * 100
    print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

    if config.remove_nobbox:
        df = df.query("num_bbox>0")

    image_paths = df.old_image_path.tolist()

    _ = Parallel(n_jobs=-1, backend="threading")(
        delayed(make_copy)(row) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    df.loc[:, ["bboxes"]] = df.annotations.progress_apply(get_bbox)

    # get image size
    df.loc[:, ["width"]] = 1280
    df.loc[:, ["height"]] = 720

    df, bboxes_info, all_bboxes = create_labels(df)
    df = create_fold(df, config)
    bbox_df = bbox_distribution(df, bboxes_info, all_bboxes)
    train_df, valid_df, train_files, val_files = create_dataset(df, config)
    create_config(train_df, valid_df, cwd=Path(config.data_dir).resolve())

    commands = [
        "python",
        "yolov5/train.py",
        "--img",
        f"{config.dim}",
        "--batch",
        f"{config.batch_size}",
        "--epochs",
        f"{config.epochs}",
        "--data",
        f"{config.data_dir}/gbr.yml",
        "--hyp",
        f"{config.data_dir}/hyp.yaml",
        "--weights",
        f"{config.model.name}.pt",
        "--project",
        "great-barrier-reef-public",
        "--name",
        f"{config.model.name}-dim{config.dim}-fold{config.fold}",
        "--exist-ok",
    ]

    subprocess.run(commands)


if __name__ == "__main__":
    main(config)
