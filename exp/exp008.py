"""exp007

forked from exp006

subsequence 5-fold CV


submission format is COCO means [x_min, y_min, width, height]

Come metrics `F2` tolerates some FP in order to ensure very few starfish missed

Ref:
[1] https://www.kaggle.com/daimarusui3/great-barrier-reef-yolov5-train/edit
"""

import subprocess

import numpy as np
import yaml
from sympy import randMatrix
from tqdm.auto import tqdm

tqdm.pandas()
import glob
import os
import shutil
import sys
from pathlib import Path
from pprint import pprint
from typing import List, Tuple
import torch
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
    "fold": 0,  # validation fold if you run all, fold is updated
    "n_splits": 5,
    "train_fold": [0, 1, 2, 3, 4],
    "epochs": 25,
    "dim": 4000,
    "model": {"name": "yolov5s6"},
    "batch_size": 2,  # if batch_size == 1, yolov5 trainer estimates batch_size
    "remove_nobbox": True,
    "with_background": True,
    "bg_rate": 0.1,
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
    df = pd.read_csv(
        ROOT / "input" / "cross-validation" / f"train-{config.n_splits}folds.csv"
    )
    df.loc[:, ["old_img_path"]] = df["old_image_path"] = (
        f"{data_dir}/train_images/video_"
        + df.video_id.astype(str)
        + "/"
        + df.video_frame.astype(str)
        + ".jpg"
    )
    df.loc[:, ["image_path"]] = (
        f"{Path(config['out_img_dir']).resolve()}/" + df.image_id + ".jpg"
    )
    df.loc[:, ["label_path"]] = (
        f"{Path(config['out_lbl_dir']).resolve()}/" + df.image_id + ".txt"
    )
    df.loc[:, ["annotations"]] = df["annotations"].progress_apply(eval)
    print("#" * 20 + " check dataframe " + "#" * 20)
    print(df.head())
    print("image_path: ")
    print(df["image_path"].values[0])
    print("annotations: ")
    print(df["annotations"].values)
    print(df.columns)
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
    print("Missing:", cnt)
    return df, bboxes_info, all_bboxes


def create_fold(df: pd.DataFrame, config, is_skf: bool = False) -> pd.DataFrame:
    """

    Reference:
        [1] https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
    """
    if is_skf:
        # NOTE : it might be bugs or some mistake
        print("\n" + "#" * 10 + " Stratified KFold " + "#" * 10 + "\n")
        from sklearn.model_selection import StratifiedKFold

        kf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
        print(df["subsequence_id"].unique())
        for fold_idx, (_, val_idx) in enumerate(
            kf.split(df["subsequence_id"], y=df["has_annotations"])
        ):
            print(fold_idx)
            subseq_val_idx = df["subsequence_id"].iloc[val_idx]
            df.loc[df["subsequence_id"].isin(subseq_val_idx), "fold"] = fold_idx
        df.loc[:, "fold"] = df["fold"].astype(int)

        print(df["fold"].unique())
        print(df["fold"].value_counts(dropna=False))
    else:
        df.loc[:, "fold"] = df.loc[:, "image_path"].map(
            lambda x: Path(x).stem.split("-")[0]
        )
        df.loc[:, "fold"] = df.loc[:, "fold"].astype(int)
        print(df.fold.value_counts())
        assert (
            config.n_splits == df.fold.unique().shape[0]
        ), f" {config.n_splits} is not same as {df.fold.unique().shape[0]}"
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
    print(f"\n ###### fold{fold} datasets is made. ####### ")
    train_files = []
    val_files = []
    train_df = df.query(f"fold!={fold}")
    valid_df = df.query(f"fold=={fold}")
    print("train_df.shape: ", train_df.shape)
    print("valid_df.shape: ", valid_df.shape)
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


Coordinates = Tuple[float, float, float, float]


def calc_iou(a: Coordinates, b: Coordinates) -> float:
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b

    if a == b:
        return 1.0
    elif ((a_x1 <= b_x1 and a_x2 > b_x1) or (a_x1 >= b_x1 and b_x2 > a_x1)) and (
        (a_y1 <= b_y1 and a_y2 > b_y1) or (a_y1 >= b_y1 and b_y2 > a_y1)
    ):
        intersection = (min(a_x2, b_x2) - max(a_x1, b_x1)) * (
            min(a_y2, b_y2) - max(a_y1, b_y1)
        )
        union = (
            (a_x2 - a_x1) * (a_y2 - a_y1) + (b_x2 - b_x1) * (b_y2 - b_y1) - intersection
        )
        return intersection / union
    else:
        return 0.0


def fuse_wbf(
    bboxes: List[Coordinates], scores: List[float], iou_threshold: float, n: int
) -> Tuple[List, List]:
    """Weighted Boxes Fusion

    Ref:
    [1] https://ohke.hateblo.jp/entry/2020/06/20/230000
    """
    lists: List[List[Coordinates]] = []
    fusions: List[Coordinates] = []
    confidences: List[List[float]] = []

    # scoresに格納されたconfidenceを基準に降順にソートしたindex
    indexes = sorted(range(len(bboxes)), key=scores.__getitem__)[::-1]
    for i in indexes:
        new_fusion = True
        for j in range(len(fusions)):
            if calc_iou(bboxes[i], fusions[j]) > iou_threshold:
                lists[j].append(bboxes[i])
                confidences[j].append(scores[i])
                fusions[j] = (
                    sum(coords[0] * c for coords, c in zip(lists[j], confidences[j]))
                    / sum(confidences[j]),
                    sum(coords[1] * c for coords, c in zip(lists[j], confidences[j]))
                    / sum(confidences[j]),
                    sum(coords[2] * c for coords, c in zip(lists[j], confidences[j]))
                    / sum(confidences[j]),
                    sum(coords[3] * c for coords, c in zip(lists[j], confidences[j]))
                    / sum(confidences[j]),
                )
                new_fusion = False
        if new_fusion:
            lists.append([bboxes[i]])
            confidences.append([scores[i]])
            fusions.append(bboxes[i])

    confidences = [(sum(c) / len(c)) * (min(n, len(c)) / n) for c in confidences]

    return fusions, confidences


def update_config(config):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fold", default=-1, type=int, nargs="*")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.train_fold != -1:
        assert isinstance(args.train_fold, List)
        assert len(args.train_fold) <= config.n_splits
        config["train_fold"] = args.train_fold

    if args.debug:
        config["epochs"] = 1

    return config


def train(config):
    pprint(config)

    df = get_df(config)

    df.loc[:, ["num_bbox"]] = df["annotations"].progress_apply(lambda x: len(x))
    data = (df.num_bbox > 0).value_counts(normalize=True) * 100
    print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

    if config.with_background:
        bg_size = int(df.shape[0] * config.bg_rate)
        sampled_bg_df = df.query("num_bbox==0").sample(bg_size)
    if config.remove_nobbox:
        df = df.query("num_bbox>0")
        print("shape of df which is num_bbox>0: ", df.shape)
        if config.with_background:
            df = pd.concat([df, sampled_bg_df], axis=0)
            print(f"shape of df which is num_bbox>0 and including num_bbox==0 with {bg_size} :", df.shape)


    image_paths = df.old_image_path.tolist()

    _ = Parallel(n_jobs=-1, backend="threading")(
        delayed(make_copy)(row) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    df.loc[:, ["bboxes"]] = df.annotations.progress_apply(get_bbox)

    # get image size
    # 縦横比2:1
    df.loc[:, ["width"]] = 1280
    df.loc[:, ["height"]] = 720

    df, bboxes_info, all_bboxes = create_labels(df)
    # df = create_fold(df, config, is_skf=True)
    # bbox_df = bbox_distribution(df, bboxes_info, all_bboxes)
    train_df, valid_df, train_files, val_files = create_dataset(df, config)
    create_config(train_df, valid_df, cwd=Path(config.data_dir).resolve())

    torch.cuda.empty_cache()

    print("✅ check the amount of each fold ↓")
    print(df["fold"].value_counts())
    expname = Path(__file__).stem
    commands = [
        "python",
        "yolov5/train.py",
        "--img-size",
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
        "--optimizer",
        "AdamW",
        "--project",
        "great-barrier-reef-yolov5",
        "--name",
        f"{expname}-{config.model.name}-dim{config.dim}-fold{config.fold}-epoch{config.epochs}",
        "--exist-ok",
        "--cache",
        "disk"
    ]

    subprocess.run(commands)

    torch.cuda.empty_cache()


def main(config):
    config = update_config(config)
    for fold in config.train_fold:
        config["fold"] = fold
        train(config)


if __name__ == "__main__":
    main(config)
