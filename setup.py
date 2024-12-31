import torch
import random
import numpy as np


def setup_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_num_class(dataset):
    if dataset == 'mini_imagenet':
        return 64
    elif dataset == 'tiered_imagenet':
        return 351
    elif dataset == 'fc100':
        return 60
    elif dataset == 'cifar_fs':
        return 64
    elif dataset == 'cub':
        return 100
    else:
        raise ValueError('Unknown dataset. Please check your selection!')
