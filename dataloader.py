import os
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(dataset_name="imagenet", which="train", subsample=1.0, batch_size=128, config=None):
    if dataset_name == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if which == "train":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])
        dataset_path = os.path.join(config.DATASET_PATH, which)
        dataset = datasets.ImageFolder(dataset_path, transform)
    elif dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2471, 0.2435, 0.2616]
        if which == "train":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        dataset = datasets.CIFAR10(root    ='../data', train    =True if which == "train" else False,
                                   download=True,      transform=transform)
    elif dataset_name == "cifar100":
        mean = [0.5074,0.4867,0.4411]
        std  = [0.2011,0.1987,0.2025]
        if which == "train":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        dataset = datasets.CIFAR100(root      = '/shared/public/dataset/datasets_cvip', 
                                    train     = True if which == "train" else False, 
                                    download  = True,
                                    transform = transform)
    elif dataset_name == "svhn":
        mean = [0.4376821,  0.4437697,  0.47280442]
        std  = [0.19803012, 0.20101562, 0.19703614]
        if which == "train":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ])
        dataset = datasets.SVHN(root      = '../data', 
                                split     = "train" if which == "train" else "test", 
                                download  = True, 
                                transform = transform)
        
    if subsample < 1.0:
        dataset = Subset(dataset, 
                         np.linspace(start=0, stop=len(dataset), num=int(subsample * len(dataset)),
                                     endpoint=False, dtype=np.uint32))
    print("%s-%s: %d" % (dataset_name, which, len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=True if which == "train" else False,
                                             num_workers=config.NUM_WORKERS, pin_memory=True)
        
    return dataloader
