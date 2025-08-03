import copy

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch
from functools import partial
import numpy as np

from datasets.build import build_dataset
from utils.misc import return_type

def collate_fn(batch, cfg):
    data_type = return_type(cfg, 'torch')
    x, xt, y, yt = zip(*batch)
    x = torch.from_numpy(np.stack(x)).to(data_type)
    xt = torch.from_numpy(np.stack(xt)).to(data_type)
    y = torch.from_numpy(np.stack(y)).to(data_type)
    yt = torch.from_numpy(np.stack(yt)).to(data_type)
    return x, xt, y, yt

def construct_loader(cfg, split, dataset):
    if split == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
    elif split == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = cfg.VAL.SHUFFLE
        drop_last = cfg.VAL.DROP_LAST
    elif split == "test":
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = cfg.TEST.SHUFFLE
        drop_last = cfg.TEST.DROP_LAST
    else:
        raise ValueError

    datasets = copy.deepcopy(dataset)
    datasets.split = split if split != "pred" else "test"

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            datasets,
            shuffle=shuffle
        )
        shuffle = False  # Disable shuffling when using DistributedSampler
    else:
        sampler = None
        
    loader = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR,
        persistent_workers=cfg.DATA_LOADER.PERSISTENT_WORKERS,
        collate_fn=partial(collate_fn, cfg=cfg)
    )

    return loader


def get_train_dataloader(cfg, dataset):
    return construct_loader(cfg, "train", dataset)


def get_val_dataloader(cfg, dataset):
    return construct_loader(cfg, "val", dataset)


def get_test_dataloader(cfg, dataset):
    return construct_loader(cfg, "test", dataset)
