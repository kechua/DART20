import os
from collections import defaultdict
from typing import Sequence, Callable
import torch

from tqdm import tqdm
import numpy as np

from dpipe.commands import load_from_folder
from dpipe.io import save_json
from dpipe.itertools import zip_equal
from dpipe.torch.functional import weighted_cross_entropy_with_logits

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def aggregate_metric_probably_with_ids(xs, ys, ids, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    try:
        return aggregate_fn([metric(x, y, i) for x, y, i in zip_equal(xs, ys, ids)])
    except TypeError:
        return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def evaluate_with_ids(y_true: Sequence, y_pred: Sequence, ids: Sequence[str], metrics: dict) -> dict:
    return {name: metric(y_true, y_pred, ids) for name, metric in metrics.items()}


def compute_metrics_probably_with_ids(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str],
                                      metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)


def evaluate_individual_metrics_probably_with_ids(load_y_true, metrics: dict, predictions_path, results_path,
                                                  exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][identifier] = metric(target, prediction, identifier)
            except TypeError:
                results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

def evaluate_individual_metrics_probably_with_ids_no_pred(
        load_y, load_x, predict, metrics: dict, test_ids, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        prediction = predict(load_x(_id))

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

def evaluate_individual_metrics_probably_with_ids_no_pred_DANN(
        load_y, load_x, predict, metrics: dict, test_ids, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        prediction = predict(load_x(_id), None)[0]

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

# DANN -- segmentation/classification
def compute_metrics_segm_DANN(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i), None)[0] for i in ids], metrics)

def compute_metrics_clr_DANN(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    return evaluate(list(map(load_y, ids)), [predict(load_x(i), None)[1] for i in ids], metrics)

def accWithOneHotY (x,y):
    yTrue = []
    for el in y:
        yTrue.append(int(np.argmax(el)))
    return accuracy_score(x,yTrue)

def ROC_AUC_multiclass(x,y):
    yTrue = []
    for el in y:
        yTrue.append(np.exp(el)/sum(np.exp(el)))
    return roc_auc_score(x,yTrue,multi_class='ovr')

# [image_l, image_u, mask_l, label_l, label_u] - the last 3 dims form the target
def DANNloss(x, xClrL, xClrU, x_true, xClrL_true, xClrU_true):
    # loss = criterion(architecture(*inputs), *targets) -- dpipe style
    segmLoss = weighted_cross_entropy_with_logits(x,x_true)
    clLossL = weighted_cross_entropy_with_logits(xClrL,xClrL_true)
    clLossU = weighted_cross_entropy_with_logits(xClrU, xClrU_true)
    return segmLoss+clLossL+clLossU