import argparse
import sys
import yaml
from yacs.config import CfgNode as CN

# from config import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="BaseCode"
    )
    # parser.add_argument(
    #     "--cfg",
    #     dest="cfg_file",
    #     help="Path to the config file",
    #     default=None,
    #     type=str,
    # )
    # parser.add_argument(
    #     "opts",
    #     help="See config.py for all options",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_known_args()


def load_config(args):
    # # Setup cfg.
    # cfg = get_cfg_defaults()
    # # Load config from cfg.
    # if args.cfg_file is not None:
    #     cfg.merge_from_file(args.cfg_file)
    # # Load config from command line, overwrite config from opts.
    # if args.opts is not None:
    #     cfg.merge_from_list(args.opts)
    
    cfg = {}
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    
    sweep_params = {}
    
    for k, v in cfg.items():
        if isinstance(v, dict):
            for key, value in v.items():        
                if isinstance(value, dict) and (('values' in value) or ('distribution' in value)):
                    sweep_params[f'{k}-{key}'] = value    
                    
    cfg = CN(cfg)
    cfg.TRAIN.BEST_METRIC_INITIAL = float(cfg.TRAIN.BEST_METRIC_INITIAL)
    return cfg, sweep_params
    # return cfg