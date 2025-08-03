import os
from typing import Dict

import wandb
import numpy as np
import torch
import torch.nn.functional as F
import time

from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from datasets.build import build_dataset
from trainer import prepare_inputs
from utils.misc import *
from visualize import vis_1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.losses import Losses

losses=Losses()
class Predictor:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.dtype = return_type(cfg, 'torch')

        assert hasattr(model, "forecast")
        self.model = model

        cfg.TRAIN.SHUFFLE, cfg.TRAIN.DROP_LAST = False, False
        
        self.dataset = build_dataset(self.cfg)
        
        self.train_loader = get_train_dataloader(self.cfg, self.dataset)
        self.val_loader = get_val_dataloader(self.cfg, self.dataset)
        self.test_loader = get_test_dataloader(self.cfg, self.dataset)

        #TODO 여기 둘다 건드려야 될 거 같은데? error 만 반환하는거 보니까
        if cfg.TEST.PREDICTION_ERROR_DIR:
            self.test_errors, self.train_errors = self._load_prediction_errors(cfg.TEST.PREDICTION_ERROR_DIR)
        else:
            self.test_errors, self.train_errors = self._get_test_errors(), self._get_train_errors()

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        log_dict = {}
        
        self.errors_all = {
            "test_mse": self.test_errors['mse'], 
            "test_mae": self.test_errors['mae'], 
            "train_mse": self.train_errors['mse'], 
            "train_mae": self.train_errors['mae'], 
        }
        
        if self.cfg.TEST.VIS_DATA ==  True:
            self.data_all = {
                "test_top_enc_window": self.test_errors['top_enc_window'],
                "test_top_ground_truth": self.test_errors['top_ground_truth'],
                "test_top_pred": self.test_errors['top_pred'],
                
                "test_worst_enc_window": self.test_errors['worst_enc_window'],
                "test_worst_ground_truth": self.test_errors['worst_ground_truth'],
                "test_worst_pred": self.test_errors['worst_pred'],
                
                "train_top_enc_window": self.train_errors['top_enc_window'],
                "train_top_ground_truth": self.train_errors['top_ground_truth'],
                "train_top_pred": self.train_errors['top_pred'],
                
                "train_worst_enc_window": self.train_errors['worst_enc_window'],
                "train_worst_ground_truth": self.train_errors['worst_ground_truth'],
                "train_worst_pred": self.train_errors['worst_pred'],
            }

        results = self.get_results()  # {test_mse: , test_mae:, train_mse: train_mae: }
        self.save_results(results)
        
        self.save_to_npy(**self.errors_all)

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})
        if is_main_process() and self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

    def visualize(self):
        if self.cfg.TEST.VIS_ERROR == True:
            for metric, errors in self.errors_all.items():
                assert isinstance(errors, np.ndarray)
                if self.cfg.TEST.APPLY_MOVING_AVERAGE:
                    window_size = self.cfg.TEST.MOVING_AVERAGE_WINDOW
                    errors = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
                vis_1d(errors, title=metric, save_path=os.path.join(self.cfg.RESULT_DIR, f'{metric}.png'))
                     
        if self.cfg.TEST.VIS_DATA == True:
            keys = list(self.data_all.keys())
            pbar = tqdm(total=(2*(len(self.data_all[keys[0]])+len(self.data_all[keys[3]]))), desc ='Visualizing Data')
            for i in range(4): # keys에서 test_top, test worst, train_top, train_worst 순서로 저장되어 있음
                temp = keys[3*i].split('_', 2)
                temp = '_'.join(temp[:2])
                for j in range(len(self.data_all[keys[3*i]])): # range : top, worst 개수에 따라
                    save_path = os.path.join(self.cfg.RESULT_DIR,f'{temp}_{j+1}/')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    k_range = self.data_all[keys[3*i]][j].shape[1]-self.cfg.DATA.TARGET_START_IDX #TARGET_START_IDX가 0이 아닐 경우 그만큼 빼줘야함
                    for k in range(k_range):
                        plt.clf()
                        plt.plot(self.data_all[keys[3*i]][j][:,k+self.cfg.DATA.TARGET_START_IDX], label='input', color='green')
                        plt.plot(range(len(self.data_all[keys[3*i]][j][:,k+self.cfg.DATA.TARGET_START_IDX]), len(self.data_all[keys[3*i]][j][:,k+self.cfg.DATA.TARGET_START_IDX])+len(self.data_all[keys[3*i+1]][j][:,k])),\
                            self.data_all[keys[3*i+1]][j][:,k], label='ground truth', color='red')
                        plt.plot(range(len(self.data_all[keys[3*i]][j][:,k+self.cfg.DATA.TARGET_START_IDX]), len(self.data_all[keys[3*i]][j][:,k+self.cfg.DATA.TARGET_START_IDX])+len(self.data_all[keys[3*i+2]][j][:,k])), \
                            self.data_all[keys[3*i+2]][j][:,k], label='prediction', color='blue')
                        plt.legend(loc='upper left')
                        save_path_fig = os.path.join(save_path, f'n_var_{k+1}.png')
                        plt.savefig(save_path_fig)
                    pbar.update(1)
            pbar.close()
            
    #@torch.no_grad()
    def _get_errors_from_dataloader(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()
        mse_all = []
        mae_all = []
        top_enc_window = {}
        top_ground_truth = {}
        top_pred = {}
        worst_enc_window = {}
        worst_ground_truth = {}
        worst_pred = {}
        
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='Calculating Errors'):
                enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)

                ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].to(self.dtype)
                dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).to(self.dtype)
                dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).to(self.dtype).cuda()
                # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
                model_cfg = self.cfg.MODEL
                
                if model_cfg.output_attention:
                    pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
                else:
                    pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)

                pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]

                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
                
                mse_all.append(mse) 
                mae_all.append(mae)

                #! 이 논리 맞는지 확인 필요
                if self.cfg.TEST.VIS_DATA == True:
                    for new_mae, new_enc_window, new_ground_truth, new_pred in zip(mae, enc_window, ground_truth, pred):
                        if len(top_enc_window) < self.cfg.TEST.VIS_DATA_NUM:
                            top_enc_window[new_mae.item()] = new_enc_window
                            top_ground_truth[new_mae.item()] = new_ground_truth
                            top_pred[new_mae.item()] = new_pred
                        else:
                            if new_mae < max(top_enc_window.keys()):
                                del top_enc_window[max(top_enc_window.keys())]
                                del top_ground_truth[max(top_ground_truth.keys())]
                                del top_pred[max(top_pred.keys())]
                                top_enc_window[new_mae.item()] = new_enc_window
                                top_ground_truth[new_mae.item()] = new_ground_truth
                                top_pred[new_mae.item()] = new_pred
                                
                        if len(worst_enc_window) < self.cfg.TEST.VIS_DATA_NUM:
                            worst_enc_window[new_mae.item()] = new_enc_window
                            worst_ground_truth[new_mae.item()] = new_ground_truth
                            worst_pred[new_mae.item()] = new_pred
                        else:
                            if new_mae > min(worst_enc_window.keys()):
                                del worst_enc_window[min(worst_enc_window.keys())]
                                del worst_ground_truth[min(worst_ground_truth.keys())]
                                del worst_pred[min(worst_pred.keys())]
                                worst_enc_window[new_mae.item()] = new_enc_window
                                worst_ground_truth[new_mae.item()] = new_ground_truth
                                worst_pred[new_mae.item()] = new_pred

        mse_all = torch.flatten(torch.cat(mse_all, dim=0))
        mae_all = torch.flatten(torch.cat(mae_all, dim=0))

        if dist_is_initialize():
            world_size = dist.get_world_size()
            
            mse_list = [torch.zeros_like(mse_all) for _ in range(world_size)]
            mae_list = [torch.zeros_like(mae_all) for _ in range(world_size)]
            dist.all_gather(mse_list, mse_all)
            dist.all_gather(mae_list, mae_all)
            
            mse_all = torch.cat(mse_list, dim=0).cpu().numpy()
            mae_all = torch.cat(mae_list, dim=0).cpu().numpy()
        
        else:
            mse_all = mse_all.cpu().numpy()
            mae_all = mae_all.cpu().numpy()

        if self.cfg.TEST.VIS_DATA == True:
            top_enc_window = torch.stack(list(top_enc_window.values()), axis=0).cpu().numpy()
            top_ground_truth = torch.stack(list(top_ground_truth.values()), axis=0).cpu().numpy()
            top_pred = torch.stack(list(top_pred.values()), axis=0).cpu().numpy()
            worst_enc_window = torch.stack(list(worst_enc_window.values()), axis=0).cpu().numpy()
            worst_ground_truth = torch.stack(list(worst_ground_truth.values()), axis=0).cpu().numpy()
            worst_pred = torch.stack(list(worst_pred.values()), axis=0).cpu().numpy()

            return {
                'mse': mse_all,
                'mae': mae_all,
                'top_enc_window': top_enc_window,
                'top_ground_truth': top_ground_truth,
                'top_pred': top_pred,
                'worst_enc_window': worst_enc_window,
                'worst_ground_truth': worst_ground_truth,
                'worst_pred': worst_pred
            }
        else:
            return {'mse': mse_all, 'mae': mae_all}

    def _get_train_errors(self):
        return self._get_errors_from_dataloader(self.train_loader)

    def _get_test_errors(self):
        return self._get_errors_from_dataloader(self.test_loader)

    def _load_prediction_errors(anomaly_scores_dir: str):
        test_scores_path = os.path.join(anomaly_scores_dir, 'test_scores.npy')
        test_labels_path = os.path.join(anomaly_scores_dir, 'test_labels.npy')
        train_scores_path = os.path.join(anomaly_scores_dir, 'train_scores.npy')
        
        test_scores = np.load(test_scores_path) if os.path.isfile(test_scores_path) else None
        test_labels = np.load(test_labels_path) if os.path.isfile(test_labels_path) else None
        train_scores = np.load(train_scores_path) if os.path.isfile(train_scores_path) else None
        
        return test_scores, test_labels, train_scores

    def get_results(self) -> Dict[str, float]:
        test_mse = self.test_errors['mse'].mean().astype(float)
        test_mae = self.test_errors['mae'].mean().astype(float)
        train_mse = self.train_errors['mse'].mean().astype(float)
        train_mae = self.train_errors['mae'].mean().astype(float)
        
        return {"test_mse": test_mse, "test_mae": test_mae, "train_mse": train_mse, "train_mae": train_mae}

    def save_results(self, results):
        results_string = ", ".join([f"{metric}: {value:.04f}" for metric, value in results.items()])
        print(results_string)

        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / "test.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)
