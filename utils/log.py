import os
from datetime import datetime
from pytz import timezone

import wandb
from yacs.config import CfgNode as CN

from utils.misc import mkdir


def init_wandb(cfg: CN):
    wandb.init(
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        job_type=cfg.WANDB.JOB_TYPE,
        notes=cfg.WANDB.NOTES,
        dir=cfg.WANDB.DIR,
        resume="allow",
        config=cfg
    )
    formatted_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H%M%S")
    # save checkpoints and results in the wandb log directory
    #cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, 'wandb', formatted_time+'_'+wandb.run.id)))
    #cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, 'wandb', formatted_time+'_'+wandb.run.id)))
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, 'wandb', formatted_time + '_' + cfg.WANDB.NAME + '_' + wandb.run.id)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, 'wandb', formatted_time + '_' + cfg.WANDB.NAME + '_' + wandb.run.id)))

def update_wandb(cfg: CN, global_sweep_id: str) -> CN: # sweep 켜진 경우 
    wandb.init(resume="allow", dir=f"./wandb/sweep-{global_sweep_id}")
    
    wandb_dir = os.path.join('wandb', f"sweep-{global_sweep_id}")
    cfg.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.TRAIN.CHECKPOINT_DIR.split('wandb')[0], wandb_dir)
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR.split('wandb')[0], wandb_dir)
    
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, wandb.run.id)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, wandb.run.id)))
    
    for key, value in wandb.config.items():
        param_path = key.replace('-', '.')
        keys = param_path.split('.')
        cfg[keys[0]][keys[1]] = value
        
    cfg.WANDB.ENABLE = True
    
    wandb.config.update(cfg)
    
    return cfg
    
    
def set_time_to_log_dir(cfg: CN):
    formatted_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H%M%S")
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, formatted_time)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, formatted_time)))
