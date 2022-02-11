import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from box import Box
from norfair import Detection, Tracker
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path("./")

config = Box(
    dict(
        model=dict(name="yolov5l6", ckpt_path="./input/reef_baseline_fold12/l6_3600_uflip_vm5_f12_up/f1/best.pt"),
    ),
    batch_size=32,
    device="cpu",
    n_splits=10,
    data_dir="./input/tensorflow-great-barrier-reef",
    out_img_dir="./output/datasets/images",
    out_lbl_dir="./output/datasets/labels",
)


class RGBDataset(Dataset):
    def __init__(self, df, config):
        self._df = df
        self._config = config

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        path = self._df["image_path"].values[index]
        image = self.__load_image(path)
        return (
            index,
            torch.tensor(image).permute(2, 0, 1).to(dtype=torch.float32),
        )

    def __load_image(self, path: str) -> np.ndarray:
        image = np.array(Image.open(path).convert("RGB"))
        return image


def load_model(ckpt_path, conf=0.25, iou=0.50, device: str = "cpu"):
    model = torch.hub.load(
        "./yolov5", model="custom", path=ckpt_path, source="local", force_reload=True, device=device
    )  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model

def load_model_exp(path: Path, fold: int = 0):
    from exp.exp015 import MyLitModel, config
    model = MyLitModel(
        config=config,
        fold=fold
    )
    model.load_from_checkpoint(str(path))
    return model

def get_df(config, data_dir: str = "./input/tensorflow-great-barrier-reef") -> pd.DataFrame:
    df = pd.read_csv(ROOT / "input" / "cross-validation" / f"train-{config.n_splits}folds.csv")
    df.loc[:, ["old_img_path"]] = df["old_image_path"] = (
        f"{data_dir}/train_images/video_" + df.video_id.astype(str) + "/" + df.video_frame.astype(str) + ".jpg"
    )
    df.loc[:, ["image_path"]] = f"{Path(config['out_img_dir']).resolve()}/" + df.image_id + ".jpg"
    df.loc[:, ["label_path"]] = f"{Path(config['out_lbl_dir']).resolve()}/" + df.image_id + ".txt"
    df.loc[:, ["annotations"]] = (
        df["annotations"].map(eval).map(lambda bboxes: [[x["x"], x["y"], x["width"], x["height"]] for x in bboxes])
    )
    print("#" * 20 + " check dataframe " + "#" * 20)
    print(df.head())
    print("image_path: ")
    print(df["image_path"].values[0])
    print("annotations: ")
    print(df["annotations"].values)
    print(df.columns)
    return df


# Helper to convert bbox in format [x_min, y_min, x_max, y_max, score] to norfair.Detection class
def to_norfair(detects, frame_id):
    result = []
    for x_min, y_min, x_max, y_max, score in detects:
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        result.append(Detection(points=np.array([xc, yc]), scores=np.array([score]), data=np.array([w, h, frame_id])))

    return result


# Euclidean distance function to match detections on this frame with tracked_objects from previous frames
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def main(config):
    use_no_bbox = False
    df = get_df(config)

    # Number of annotaions == 0
    if use_no_bbox:
        df.loc[:, ["num_bbox"]] = df["annotations"].map(lambda x: len(x))
        data = (df.num_bbox > 0).value_counts(normalize=True) * 100
        print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")
        val_no_bbox_df = df[df["num_bbox"] == 0]
        print("shape of val_no_bbox_df: ", val_no_bbox_df.shape)
        val_no_bbox_dataset = RGBDataset(df=val_no_bbox_df, config=config)
        val_loader = DataLoader(
            val_no_bbox_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=False
        )
        model = load_model(config.model.ckpt_path, conf=0.20, iou=0.4, device=config.device)
    else:
        fold = 0
        model_path = Path("output") / ""

        val_df = df[df["fold"] == fold]
        val_dataset = RGBDataset(df=val_df, config=config)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        model = load_model_exp(path=model_path, fold=fold)

    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=30,
        hit_inertia_min=3,
        hit_inertia_max=6,
        initialization_delay=1,
    )
    frame_id = 0
    pred_df = val_no_bbox_df.copy()

    # TODO: batch_size=1用になってるからbatch推論ができるならそっちに書き換える
    for idx, batch in enumerate(tqdm(val_loader)):
        indxes, image = batch[0], batch[1].to(device=config.device)
        detects = []
        anno = ""
        print("Image Shape: ", image.shape)
        with torch.inference_mode():        preds = torch.where(preds >= 0.5, 1, 0).to(dtype=torch.int32)
            output = model(image, size=10000, augment=True)
        print(type(output))
        break
        if output.pandas().xyxy[0].shape[0] == 0:
            anno = ""
        else:
            for idx, row in output.pandas().xyxy[0].iterrows():
                if row.confidence > 0.28:
                    anno += f"{row.confidence} {int(row.xmin)} {int(row.ymin)} {int(row.xmax - row.xmin)} {int(row.ymax - row.ymin)}"
                    detects.append([int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.confidence])
            tracked_objects = tracker.update(detections=to_norfair(detects, frame_id))
            for t_obj in tracked_objects:
                bbox_width, bbox_height, last_detected_frame_id = t_obj.last_detection.data
                if last_detected_frame_id == frame_id:
                    continue
                xc, yc = t_obj.estimate[0]
                x_min, y_min = int(round(xc - bbox_width / 2)), int(round(yc - bbox_height / 2))
                score = t_obj.last_detection_scores[0]
                anno += f"{score} {x_min} {y_min} {bbox_width} {bbox_height}"

        preds = anno.strip(" ")
        pred_df.iloc[indxes, "preds"] = preds
        pred_df.iloc[indxes, "label"] = label

    pred_df.to_csv("./output/pred-nobbox-df.csv")


if __name__ == "__main__":
    main(config)
