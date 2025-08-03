# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module): # Mean Absolute Percentage Error (MAPE) loss
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, trainer, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor: # float
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module): # Symmetric Mean Absolute Percentage Error (sMAPE) loss
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, trainer, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor: # float
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module): # Mean Absolute Scaled Error (MASE) loss
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, trainer, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor: # float
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
    
class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
    
    def forward(self, trainer, pred, gt):
        return F.mse_loss(pred, gt)
    
class mae_loss(nn.Module):
    def __init__(self):
        super(mae_loss, self).__init__()
    
    def forward(self, trainer, pred, gt):
        return F.l1_loss(pred, gt)  


class cross_entropy_loss(nn.Module):
    """
    Standard Cross‑Entropy loss for single‑label classification tasks.
    """
    def __init__(self):
        super(cross_entropy_loss, self).__init__()

    def forward(self, trainer, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        # target is expected to contain class indices, not one‑hot vectors
        return F.cross_entropy(pred, target)


class focal_loss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) to address class‑imbalance by down‑weighting easy examples.
    γ (gamma) controls focusing parameter, α (alpha) is a balancing factor.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, trainer, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        logpt = F.cross_entropy(pred, target, reduction="none")
        pt = t.exp(-logpt)
        loss = self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class label_smoothing_ce_loss(nn.Module):
    """
    Cross‑Entropy with label smoothing (Szegedy et al., 2016).
    """
    def __init__(self, smoothing: float = 0.1):
        super(label_smoothing_ce_loss, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, trainer, pred: t.Tensor, target: t.Tensor) -> t.Tensor:
        n_classes = pred.size(1)
        with t.no_grad():
            smooth_target = t.zeros_like(pred).fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-smooth_target * log_prob).sum(dim=1)
        return loss.mean()
        
class Losses():
    def __init__(self):
        self.losses = {
            # Regression losses
            "MSE": mse_loss(),
            "MAE": mae_loss(),
            "MAPE": mape_loss(),
            "SMAPE": smape_loss(),
            "MASE": mase_loss(),
            # Classification losses
            "CE": cross_entropy_loss(),
            "Focal": focal_loss(),
            "LabelSmoothingCE": label_smoothing_ce_loss(),
        }
        
        self.metrics = self.losses