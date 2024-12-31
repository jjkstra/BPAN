import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os
from tqdm import tqdm
from common import Accumulator
from PIL import ImageFilter
from collections import OrderedDict


def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.


def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def l2_distance(x1, x2):
    return -torch.cdist(x1, x2)


def cosine_distance(x1, x2):
    return F.normalize(x1, dim=-1) @ F.normalize(x2, dim=-1).transpose(-2, -1)


def symmetric_kl_divergence(score1, score2):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return (kl_loss(F.log_softmax(score1, dim=1), F.softmax(score2, dim=1))
            + kl_loss(F.log_softmax(score2, dim=1), F.softmax(score1, dim=1)))


def strip_prefix(state_dict: OrderedDict, prefix: str):
    """
    Strip a prefix from the keys of a state_dict. Can be used to address compatibility issues from
    a loaded state_dict to a models with slightly different parameter names.
    Example usage:
        state_dict = torch.load("models.pth")
        # state_dict contains keys like "module.encoder.0.weight" but the models expects keys like "encoder.0.weight"
        state_dict = strip_prefix(state_dict, "module.")
        models.load_state_dict(state_dict)
    Args:
        state_dict: pytorch state_dict, as returned by models.state_dict() or loaded via torch.load()
            Keys are the names of the parameters and values are the parameter tensors.
        prefix: prefix to strip from the keys of the state_dict. Usually ends with a dot.

    Returns:
        copy of the state_dict with the prefix stripped from the keys
    """
    return OrderedDict(
        [
            (k[len(prefix):] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


def get_grid_location(size, ratio, grid_size):
    raw_grid_size = int(size / grid_size)
    enlarged_grid_size = int(size / grid_size * ratio)
    center_location = raw_grid_size // 2
    location_list = []
    for i in range(grid_size):
        location_list.append((max(0, center_location - enlarged_grid_size // 2),
                              min(size, center_location + enlarged_grid_size // 2)))
        center_location = center_location + raw_grid_size

    return location_list


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
