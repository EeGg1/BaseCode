import torch
import importlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.misc import return_type

def build_model(cfg):
    try:
        # Dynamically import the module containing the model
        module = importlib.import_module(f"models.{cfg.MODEL_NAME}")

        # Retrieve the model class dynamically
        model_class = getattr(module, "Model")  # Ensure the model class is named "Model"
        model = model_class(cfg.MODEL)

        use_deepspeed = cfg.TRAIN.USE_DEEPSPEED if hasattr(cfg.TRAIN, 'USE_DEEPSPEED') else False
        
        if use_deepspeed:
            import deepspeed
        
        model = model.to(return_type(cfg, 'torch'))
        
        if use_deepspeed:

            ds_config = cfg.TRAIN.DEEPSPEED_CONFIG
            model_engine, _, _, _ = deepspeed.initialize(
                model=model,
                #optimizer=self.optimizer,
                args=None,
                model_parameters=model.parameters(),
                config=ds_config,
                #training_data=self.train_loader
            )
            model = model_engine
        
        if torch.cuda.is_available() and not use_deepspeed:
            print('noo')
            model = model.cuda()
        if cfg.NUM_GPUS > 1  and not use_deepspeed:
            model = DDP(model, device_ids=cfg.VISIBLE_DEVICES.split(','),)

        return model

    except ModuleNotFoundError:
        raise ImportError(f"Model {cfg.MODEL_NAME} not found in models/")
