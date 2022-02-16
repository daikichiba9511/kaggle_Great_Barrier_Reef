"""exp015

forked from exp011

subsequence 5-fold CV

by high resolution, model can detect mini cots but may increase FP.

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
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision.transforms as T
from box import Box
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.append("../input/tensorflow-great-barrier-reef")

from pprint import pprint

from bbox.utils import annot2str, clip_bbox, coco2voc, coco2yolo, draw_bboxes, load_image, str2annot, voc2yolo

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
    "n_splits": 10,
    "train_fold": [0],
    "epochs": 10,
    "dim": int(32 * 100),  # 32 * 40 = 1280
    "model": {"name": "resnet18", "pretrained": True, "layer_freeze": False},
    "batch_size": 128,  # if batch_size == 1, yolov5 trainer estimates batch_size
    "data_dir": "./input/tensorflow-great-barrier-reef",
    "binary_dir": "./input/binary-cots",
    "out_img_dir": "./output/datasets/images",
    "out_lbl_dir": "./output/datasets/labels",
    "optimizer": {
        "name": "optim.AdamW",
        "params": {"lr": 1e-4},
    },
    "scheduler": {
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {
            "T_0": 10,
            "eta_min": 1e-6,
        },
    },
    "loss": "nn.BCEWithLogitsLoss",
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 2,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
        # "precision": 16,
    },
}

config = Box(config)
pprint(config)
np.random.seed(config.seed)


class RGBDataset(Dataset):
    def __init__(self, df, config: Box, img_size: Optional[Tuple[int, int]] = (512, 512)):
        self._df = df
        self._config = config
        self._img_size = img_size

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        path = self._df["image_path"].values[index]
        image = self.__load_image(path)
        label = self._df["label"].values[index]
        return (torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label, dtype=torch.float32))

    def __load_image(self, path: str) -> np.ndarray:
        # default (720, 1200)
        image = Image.open(path).convert("RGB")
        if self._img_size is not None:
            image = image.resize(self._img_size)
        return np.array(image)


class MyDataModule(LightningDataModule):
    def __init__(self, train_df, val_df, cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.cfg = cfg

    def __create_dataset(self, train=True):
        return RGBDataset(self.train_df, self.cfg) if train else RGBDataset(self.val_df, self.cfg)

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)


class MLP(torch.nn.Module):
    def __init__(self, out_features: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.LazyLinear(out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        z1 = self.fc1(x)
        z1 = self.dropout1(z1)
        logits = self.fc2(z1)
        return logits


class MyLitModel(LightningModule):
    def __init__(self, cfg, fold):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.__build_model(cfg.model.layer_freeze)
        self._criterion = eval(self.cfg.loss)()
        self.save_hyperparameters(cfg)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def __build_model(self, layer_freeze: bool = False):
        self.model = timm.create_model(
            self.cfg.model.name,
            pretrained=self.cfg.model.pretrained,
            num_classes=0,
            in_chans=3,
        )
        if layer_freeze:
            for param in self.model.parameters():
                param.require_grad = False
        # self.head = torch.nn.LazyLinear(out_features=1)
        self.head = MLP(1, 512)

    def forward(self, image):
        z = self.model(image)
        logits = self.head(z)
        return logits

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {
            "loss": loss,
            "pred": pred,
            "labels": labels,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels, "loss": loss}

    def __share_step(self, batch, mode):
        images, labels = batch

        logits = self.forward(images)

        labels = labels.reshape(-1, 1)
        loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu()
        labels = labels.detach().cpu()
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        losses = []
        for out in outputs:
            pred, label, loss = out["pred"], out["labels"], out["loss"]
            preds.append(pred)
            labels.append(label)
            losses.append(loss.detach().cpu().numpy())
        preds = torch.cat(preds).to(dtype=torch.float32)
        labels = torch.cat(labels).to(dtype=torch.int32)

        # print("preds: ", preds.shape, "labels: ", labels.shape)
        # acc_mask = torch.where(preds >= 0.5, 1, 0) == labels
        # acc = (acc_mask).sum() / len(acc_mask)
        # assert 0 <= acc <= 1, f"acc : {acc}, preds.shape: {preds.shape}, labels.shpae: {labels.shape} "
        # print("ACC: ", acc)
        # self.log(f"{mode}_acc", acc)

        if mode == "train":
            self.train_acc(preds, labels)
            self.log("train_acc", self.train_acc, on_epoch=True)
        elif mode == "val":
            print("\nVal: \n", preds)
            self.val_acc(preds, labels)
            self.log("val_acc", self.val_acc, on_epoch=True)
            # print("VAL_ACC: ", self.val_acc)

        self.log(f"{mode}_loss", np.mean(losses))

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(self.parameters(), **self.cfg.optimizer.params)
        scheduler = eval(self.cfg.scheduler.name)(optimizer, **self.cfg.scheduler.params)
        return [optimizer], [scheduler]


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
        config["trainer"]["limit_train_batches"] = 0.01
        config["trainer"]["limit_val_batches"] = 0.01

    return config


def get_df(config):
    cots_dir_path = Path(config.binary_dir)
    cots_dir = cots_dir_path / "cots_crops"
    not_cots_dir = cots_dir_path / "notcots_crops"

    cots_image_paths = list(cots_dir.glob("*.jpg"))
    not_cots_image_paths = list(not_cots_dir.glob("*.jpg"))
    cots = {"image_path": cots_image_paths, "label": [1] * len(cots_image_paths)}
    not_cots = {"image_path": not_cots_image_paths, "label": [0] * len(cots_image_paths)}

    df = pd.DataFrame(cots)
    df = pd.concat([df, pd.DataFrame(not_cots)], axis=0)

    print("df.shape: ", df.shape)
    print("df['label'].value_counts: \n", df["label"].value_counts())
    return df


def train(config):
    df = get_df(config)

    stf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    for fold, (train_idx, val_idx) in enumerate(stf.split(df, y=df["label"])):
        if fold not in config.train_fold:
            continue
        print("#" * 8 + f"  Fold: {fold}  " + "#" * 8)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        datamodule = MyDataModule(train_df, val_df, config)
        model = MyLitModel(config, fold)
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=f"./output/{config.model.name}",
            filename=f"best_loss_{fold}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(save_dir="./output/tb_logs", name=f"{config.model.name}")

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epochs,
            callbacks=[lr_monitor, loss_checkpoint],  # , earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)


def main(config):
    config = update_config(config)
    train(config)


if __name__ == "__main__":
    main(config)
