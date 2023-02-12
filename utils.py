"""

author: san9569@naver.com (Sang Jin Park)
"""

import os
import argparse
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from tabulate import tabulate

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_classifier(config):
    # Load pre-trained classifier
    norm = True
    if config.DATASET == "imagenet":
        from classifiers.imagenet.resnet_ import resnet50
        classifier = resnet50(pretrained=True, norm=norm).to(device)
    elif ("cifar" in config.DATASET) or (config.DATASET == "svhn"):
        if config.DATASET == "cifar10":
            from classifiers.cifar10.resnet import resnet50, resnet18
            from classifiers.cifar10.wide_resnet import wrnet_28_10, wrnet_34_10, wrnet_70_16
        elif config.DATASET == "cifar100":
            from classifiers.cifar100.resnet import resnet50, resnet18
            from classifiers.cifar100.wide_resnet import wrnet_28_10, wrnet_34_10, wrnet_70_16
        elif config.DATASET == "svhn":
            from classifiers.svhn.resnet import resnet50, resnet18
            from classifiers.svhn.wide_resnet import wrnet_28_10, wrnet_34_10, wrnet_70_16
        
        if config.CLASSIFIER == "resnet18":
            classifier = resnet18(pretrained=True, device=device, norm=norm).to(device)
        elif config.CLASSIFIER == "resnet50":
            classifier = resnet50(pretrained=True, device=device, norm=norm).to(device)
        elif config.CLASSIFIER == "wrnet-28-10":
            classifier = wrnet_28_10(pretrained=True, device=device, norm=norm).to(device)
        elif config.CLASSIFIER == "wrnet-34-10":
            classifier = wrnet_34_10(pretrained=True, device=device, norm=norm).to(device)
        elif config.CLASSIFIER == "wrnet-70-16":
            classifier = wrnet_70_16(pretrained=True, device=device, norm=norm).to(device)

    return classifier


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def adjust_learning_rate(optimizers, decay, number_decay, base_lr):
    lr = base_lr * (decay ** number_decay)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_loss(loss_dict):
    df = df_maker(len(loss_dict), 1, 0)
    df.columns = [k for k in loss_dict.keys()]
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            v = v.item()
        df[k] = v
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))


def df_maker(col_num, ind_num, fill):
    col = []
    ind = []
    con = []
    for i in range(0,col_num):
        col.append(fill)
    for i in range(0,ind_num):
        ind.append(fill)
    for i in range(0,ind_num):
        con.append(col)
    return pd.DataFrame(con, columns=col, index=ind)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BBLACK = '\033[90m'
    BRED = '\033[91m'
    BGREEN = '\033[92m'
    BYELLOW = '\033[93m'
    BBLUE = '\033[94m'
    BMAGENTA = '\033[95m'
    BCYAN = '\033[96m'
    BWHITE = '\033[97m'


def recover_image(x, cv=True):
    # x = (x + 1.0) * (255.0 / 2.0)
    x = x.detach().cpu().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])
        if cv:
            x = x[:, :, :, ::-1]
    elif len(x.shape) == 3:
        x = np.transpose(x, [1, 2, 0])
        if cv:
            x = x[:, :, ::-1]
        
    return x

