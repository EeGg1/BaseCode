import random
from typing import Union
from pathlib import Path
import os


import numpy as np

import torch
import torch.distributed as dist

def dist_is_initialize():
    return dist.is_available() and dist.is_initialized()

def reduce_sum(tensor):
    if dist_is_initialize():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def is_main_process():
    return (not dist_is_initialize()) or dist.get_rank() == 0

def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
def set_devices(devices: str):
    """
    Set cuda devices

    Args:
        devices (str): Comma separated string of device numbers. e.g., "0", "0, 1"
    """
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory

def return_type(cfg, lib):
    dtype_map = {}
    if lib == 'torch':
        dtype_map = {
            "fp64": torch.float64,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "i8": torch.int8,
            "i16": torch.int16,
            "i32": torch.int32,
            "i64": torch.int64,
        }
    elif lib == 'numpy':
        dtype_map = {
            "fp64": np.float64,
            "fp32": np.float32,
            "fp16": np.float16,
            "bf16": np.float16,  # NumPy does not have a direct bf16 type, using float16 as a placeholder
            "i8": np.int8,
            "i16": np.int16,
            "i32": np.int32,
            "i64": np.int64,
        }
    key = getattr(cfg.TRAIN, "TYPE", None)
    return dtype_map.get(key, torch.float32 if lib == 'torch' else np.float32)