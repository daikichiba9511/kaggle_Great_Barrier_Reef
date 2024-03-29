"""

this codes are from Ref[1]

Ref
[1] https://www.kaggle.com/bamps53/competition-metric-implementation
"""
import numpy as np


def calc_iou(bboxes1, bboxes2, bbox_mode="xywh"):
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4

    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()

    if bbox_mode == "xywh":
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def f_beta(tp, fp, fn, beta=2):
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)


def imagewise__score_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()

    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes):  # fixed in ver.7
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou > iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - (k + 1)  # fix in ver.7
            break

    fn = len(gt_bboxes)
    score = f_beta(tp, fp, fn, beta=2)
    if verbose:
        print(f"iou_th:{iou_th.round(2):<4} tp:{tp:<2}, fp:{fp:<2}, fn:{fn:<2} :{score:.3}")
    return score


def imagewise__score(gt_bboxes, pred_bboxes, verbose=False):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    # v2: add corner case hundling.
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        return 1.0
    elif len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return 0.0

    pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()[::-1]]  # sort by conf

    scores = []
    for iou_th in np.arange(0.3, 0.85, 0.05):
        scores.append(imagewise__score_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose))
    return np.mean(scores)


def calc_f_beta(recall, precision, beta: int = 1):
    return ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
