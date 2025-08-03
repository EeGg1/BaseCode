import os
import time

from models.build import build_model
from utils.parser import parse_args, load_config
from utils.log import init_wandb, update_wandb, set_time_to_log_dir
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices

import wandb
import torch
from datetime import datetime
from pytz import timezone
from utils.misc import *
import shutil
import torch.distributed as dist

global_sweep_id = None

def train_main(cfg):    
    # setting folders
    
    if is_main_process():
        print('wtf2')
        if cfg.SWEEP.ENABLE:
            global global_sweep_id
            cfg = update_wandb(cfg, global_sweep_id)
        elif cfg.WANDB.ENABLE:
            init_wandb(cfg) #TODO 잘 되는지 체크 필요
        else :
            set_time_to_log_dir(cfg)
        
        with open(os.path.join(cfg.RESULT_DIR, 'config.txt'), 'w') as f:
            f.write(cfg.dump())
        
        shutil.copy(os.path.join(cfg.TRAIN.DEEPSPEED_CONFIG), os.path.join(cfg.RESULT_DIR, 'deepspeed_config.json'))
    
    if not is_main_process():
        os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for non-main processes
    
    # set random seed
    #set_seeds(cfg.SEED)

    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)
    
    # build model
    model = build_model(cfg)
    
    # build trainer
    trainer = build_trainer(cfg, model)

    if cfg.TRAIN.ENABLE:
        start=time.time()
        trainer.train()
        print('Training time:', time.time()-start)
    
    if cfg.TEST.ENABLE:
        model = trainer.load_best_model()
        predictor = Predictor(cfg, model)
        predictor.predict()
        
        if cfg.TEST.VIS_ERROR or cfg.TEST.VIS_DATA:
            predictor.visualize()   


def main():
    args = parse_args() # 현재 이 파일은 args 필요 안함
    cfg, sweep_params = load_config(args)
    set_seeds(cfg.SEED)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    if (not dist.is_initialized()) and cfg.TRAIN.USE_DEEPSPEED:
        print('wtf1')
        dist.init_process_group(backend='nccl')
    
    print("RANK", os.getenv("RANK"), "LOCAL_RANK", os.getenv("LOCAL_RANK"))
    
    if cfg.SWEEP.ENABLE:
        global global_sweep_id
        if cfg.SWEEP.RESUME:
            global_sweep_id = cfg.SWEEP.SWEEP_ID            
            wandb.agent(global_sweep_id, function=lambda:train_main(cfg), count=cfg.SWEEP.COUNT, project=cfg.WANDB.PROJECT)
        else:
            sweep_config = {
                'method': cfg.SWEEP.method,
                'metric': cfg.SWEEP.metric,
                'parameters': sweep_params,
                'name': cfg.WANDB.NAME,
            }
            if hasattr(cfg.SWEEP, 'early_terminate'):
                sweep_config['early_terminate'] = {
                    'type': cfg.SWEEP.early_terminate.type,
                    'min_iter': cfg.SWEEP.early_terminate.min_iter,
                }
            
            global_sweep_id = wandb.sweep(sweep_config, project=cfg.WANDB.PROJECT)
            
            wandb.agent(global_sweep_id, function=lambda:train_main(cfg), count=cfg.SWEEP.COUNT)

    else:          
        train_main(cfg)            


if __name__ == '__main__':
    main()
