""" exp012

Ref:
[1] https://www.kaggle.com/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507
"""
import subprocess
import warnings

from exp.exp011 import create_dataset

warnings.filterwarnings("ignore")

import ast
import importlib
import json
import os
import pprint
from pathlib import Path
from shutil import copyfile
from typing import List

import cv2
import pandas as pd
import torch
from tqdm.notebook import tqdm

tqdm.pandas()
from string import Template

from box import Box
from IPython.display import display
from PIL import Image

expname = Path(__file__).name

config = Box(
    dict(
        expname=expname,
        data_root="./input/tensorflow-great-barrier-reef",
        seed=42,
        n_splits=10,
        train_fold=[0],
        val_fold=0,
        model=dict(name="yolox-l"),
        batch_size=32,
        epochs=15,
    )
)


def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


def get_path(row):
    row["image_path"] = f"{config.data_root}/train_images/video_{row.video_id}/{row.video_frame}.jpg"
    return row


def create_datasets(df: pd.DataFrame, config: Box, val_fold: int = 0, home_dir: Path = Path(".")) -> None:
    dataset_path = home_dir / "input" / "dataset" / "images"
    config["dataset_path"] = dataset_path
    train_path = dataset_path / "train2017"
    config["train_data_path"] = train_path
    val_path = dataset_path / "val2017"
    config["val_data_path"] = val_path
    anno_path = dataset_path / "annotations"
    config["annotations_path"] = anno_path

    dataset_path.mkdir(exist_ok=True, parents=True)
    train_path.mkdir(exist_ok=True, parents=True)
    val_path.mkdir(exist_ok=True, parents=True)
    anno_path.mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(len(df))):
        row = df.loc[i]
        if int(row.fold) != val_fold:
            copyfile(f"{row.image_path}", f"{train_path}/{row.image_id}.jpg")
        else:
            copyfile(f"{row.image_path}", f"{val_path}/{row.image_id}.jpg")

    print("The number of training files: ", len(list(train_path.iterdir())))
    print("The number of validation files: ", len(list(val_path.iterdir())))


def save_annot_json(json_annotation, filename) -> None:
    with open(filename, "w") as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)


annotion_id = 0


def dataset2coco(df, dest_path):

    global annotion_id

    annotations_json = {"info": [], "licenses": [], "categories": [], "images": [], "annotations": []}

    info = {
        "year": "2021",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "https://kaggle.com",
        "date_created": "2021-11-30T15:01:26+00:00",
    }
    annotations_json["info"].append(info)

    lic = {"id": 1, "url": "", "name": "Unknown"}
    annotations_json["licenses"].append(lic)

    classes = {"id": 0, "name": "starfish", "supercategory": "none"}

    annotations_json["categories"].append(classes)

    for ann_row in df.itertuples():

        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + ".jpg",
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2021-11-30T15:01:26+00:00",
        }

        annotations_json["images"].append(images)

        bbox_list = ann_row.bboxes

        for bbox in bbox_list:
            b_width = bbox[2]
            b_height = bbox[3]

            # some boxes in COTS are outside the image height and width
            if bbox[0] + bbox[2] > 1280:
                b_width = bbox[0] - 1280
            if bbox[1] + bbox[3] > 720:
                b_height = bbox[1] - 720

            image_annotations = {
                "id": annotion_id,
                "image_id": ann_row[0],
                "category_id": 0,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0,
            }

            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)

    print(f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
    return annotations_json


def update_config(config):
    import argparse
    import json

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
        with open("tmp_config.json", "w") as f:
            json.dump(config.to_dict(), f)

    return config


def train(fold: int, config: Box) -> None:
    config["val_fold"] = fold
    df = pd.read_csv(f"{config.data_root}/cross-validation/train-{config.n_splits}.csv")
    df["num_bbox"] = df["annotations"].apply(lambda x: str.count(x, "x"))
    df_train = df[df["num_bbox"] > 0]

    # Annotations
    df_train["annotations"] = df_train["annotations"].progress_apply(lambda x: ast.literal_eval(x))
    df_train["bboxes"] = df_train.annotations.progress_apply(get_bbox)

    # Images resolution
    df_train["width"] = 1280
    df_train["height"] = 720

    # Path of images
    df_train = df_train.progress_apply(get_path, axis=1)

    create_datasets(df_train, config=config, val_fold=config.val_fold)

    # Convert COTS dataset to JSON COCO
    train_annot_json = dataset2coco(df_train[df_train.fold != config.val_fold], str(config.train_data_path))
    val_annot_json = dataset2coco(df_train[df_train.fold == config.val_fold], str(config.val_data_path))

    save_annot_json(train_annot_json, str(config.annotaions_path / "train.json"))
    save_annot_json(val_annot_json, str(config.annotaions_path / "valid.json"))

    override_cmds = ["python", "./overwrite_yolox.py"]
    subprocess.run(override_cmds)

    run_cmds = [
        "python",
        "YOLOX/train.py",
        "-n",
        f"{config.expname}-{config.model.name}.pt" "-f",
        "cots_config.py",
        "-d",  # devices
        "1",
        "-b",
        f"{config.batch_size}",
        "--fp16",
        "-o",
        "-c",
        f"{config.model.name}.pth",
    ]
    print("Run CMD: \n", " ".join(run_cmds))
    subprocess.run(run_cmds)


def main(config):
    config = update_config(config)
    pprint.pprint(config)
    for fold in config.train_fold:
        train(fold, config)


if __name__ == "__main__":
    main(config)
