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
import os
import math

from typing import List, Optional

# ================================================================
# Utility helpers
# ================================================================

def _pairwise_squared_l2(total: torch.Tensor) -> torch.Tensor:
    """Squared ℓ2 distance (‖x−y‖²) for every pair of rows in `total`."""
    return (total.unsqueeze(0) - total.unsqueeze(1)).pow(2).sum(-1)


def _pairwise_l1(total: torch.Tensor) -> torch.Tensor:
    """ℓ1 distance (‖x−y‖₁) for every pair of rows in `total`."""
    return (total.unsqueeze(0) - total.unsqueeze(1)).abs().sum(-1)


def _scalarize(total: torch.Tensor) -> torch.Tensor:
    """Return a 1-D tensor (N,) representing each sample by its mean value."""
    if total.ndim == 1:
        return total
    return total.mean(dim=-1)


# ================================================================
# Kernels
# ================================================================

def _gaussian_multi_from_d2(
    d2: torch.Tensor,      # (N, N) ‑ pairwise squared distances
    kernel_mul: float = 0.5,
    kernel_num: int = 5,
) -> torch.Tensor:
    """
    Multi-scale Gaussian kernel given the squared distance matrix d2.
    """
    N = d2.size(0)
    # upper-triangular(0은 제외) median → bandwidth 기준
    idx = torch.triu_indices(N, N, offset=1, device=d2.device)
    median_d2 = torch.median(d2[idx[0], idx[1]].detach()) + 1e-12
    median_d = torch.sqrt(median_d2)

    base_sigma = median_d / (kernel_mul ** (kernel_num // 2))

    kernels = [
        torch.exp(-d2 / (2.0 * (base_sigma * (kernel_mul ** i)) ** 2))
        for i in range(kernel_num)
    ]
    return sum(kernels)


def _laplacian_multi(d1: torch.Tensor, kernel_mul: float, kernel_num: int) -> torch.Tensor:
    n = d1.size(0)
    idx = torch.triu_indices(n, n, offset=1, device=d1.device)
    median_val = torch.median(d1[idx[0], idx[1]].detach()) + 1e-12
    base_b = median_val / (kernel_mul ** (kernel_num // 2))
    kernels = [
        torch.exp(-d1 / (base_b * (kernel_mul ** i)))
        for i in range(kernel_num)
    ]
    return sum(kernels)


def _brownian_kernel(x: torch.Tensor) -> torch.Tensor:
    x1 = x.unsqueeze(0)
    x2 = x.unsqueeze(1)
    return torch.minimum(x1, x2)


def _brownian_bridge_kernel(x: torch.Tensor) -> torch.Tensor:
    basic = _brownian_kernel(x)
    outer = x.unsqueeze(0) * x.unsqueeze(1)
    return basic - outer


def _sobolev_spline_kernel(x: torch.Tensor, r: int = 1) -> torch.Tensor:
    N = x.size(0)
    x1 = x.unsqueeze(0)
    x2 = x.unsqueeze(1)
    absdiff = (x1 - x2).abs()
    if r == 1:
        return (absdiff.pow(3) / 12.0) - (absdiff.pow(2) / 24.0) + 1.0 / 12.0
    if r == 2:
        return (absdiff.pow(5) / 120.0) - (absdiff.pow(3) / 72.0) + \
               (absdiff.pow(2) / 144.0) - 1.0 / 720.0
    raise ValueError("Only r=1 or r=2 implemented for Sobolev spline")


def _matern_kernel(d2: torch.Tensor, nu: float = 1.5, ell: float = 1.0) -> torch.Tensor:
    dist = torch.sqrt(torch.clamp(d2, min=1e-12))
    if nu == 0.5:
        return torch.exp(-dist / ell)
    if nu == 1.5:
        coef = math.sqrt(3.0) * dist / ell
        return (1.0 + coef) * torch.exp(-coef)
    if nu == 2.5:
        coef = math.sqrt(5.0) * dist / ell
        return (1.0 + coef + coef.pow(2) / 3.0) * torch.exp(-coef)
    from torch import special
    coef = math.sqrt(2.0 * nu) * dist / ell
    kv = special.kv(nu, torch.clamp(coef, min=1e-12))
    return (2.0 ** (1 - nu) / math.gamma(nu)) * coef.pow(nu) * kv


def _wendland_kernel(dist: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    r = dist / rho
    w = torch.clamp(1 - r, min=0.0)
    poly = 35 * (r ** 2) + 18 * r + 3
    return (w ** 6) * poly


def _bspline_kernel(dist: torch.Tensor, order: int = 3, h: float = 1.0) -> torch.Tensor:
    t = dist / h
    if order == 3:
        out = torch.zeros_like(t)
        mask1 = (t < 1)
        mask2 = (t >= 1) & (t < 2)
        out[mask1] = (2.0 / 3.0) - t[mask1]**2 + (t[mask1]**3) / 2.0
        out[mask2] = ((2.0 - t[mask2])**3) / 6.0
        return out
    raise ValueError("Only cubic B-spline (order=3) implemented")


def _fejer_kernel(dist: torch.Tensor, omega0: float = math.pi) -> torch.Tensor:
    d = torch.clamp(dist, min=1e-12)
    half = 0.5 * omega0 * d
    return (torch.sin(half) / half) ** 2


def _spectral_mixture_kernel(
    d2: torch.Tensor, sigmas: List[float],
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    assert len(sigmas) == len(weights)
    K = 0.0
    for w, s in zip(weights, sigmas):
        K = K + w * torch.exp(-d2 / (2.0 * (s ** 2)))
    return K


# ================================================================
# Multi-Kernel MMD
# ================================================================

class MultiKernelMMD(nn.Module):
    """Multi-kernel MMD supporting many kernel families."""

    def __init__(
        self,
        kernel_type: str = None,
        kernel_mul: float = 0.5,
        kernel_num: int = 5,
        degree: int = 3,
        coef0: float = 1.0,
        bandwidth: float = 1.0,
        nu: float = 1.5,
        ell: float = 1.0,
        H: float = 0.75,
        rho: float = 1.0,
        sobolev_r: int = 1,
        omega0: float = math.pi,
        sm_sigmas: Optional[List[float]] = None,
        sm_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.degree = degree
        self.coef0 = coef0
        self.bandwidth = bandwidth
        self.nu = nu
        self.ell = ell
        self.H = H
        self.rho = rho
        self.sobolev_r = sobolev_r
        self.omega0 = omega0
        self.sm_sigmas = sm_sigmas if sm_sigmas is not None else [0.05, 0.2, 1.0]
        self.sm_weights = sm_weights

    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = torch.cat([x, y], dim=0)  # (2B, C or D, ...)
        if self.kernel_type == "rbf":
            d2 = _pairwise_squared_l2(total)
            return _gaussian_multi_from_d2(d2, self.kernel_mul, self.kernel_num)
        if self.kernel_type == "laplacian":
            d1 = _pairwise_l1(total)
            return _laplacian_multi(d1, self.kernel_mul, self.kernel_num)
        if self.kernel_type == "linear":
            return total @ total.t()
        if self.kernel_type == "polynomial":
            return (total @ total.t() + self.coef0) ** self.degree

        # 나머지 커널들은 1D 입력(또는 스칼라화) 기반
        scalar = _scalarize(total)
        if self.kernel_type == "brownian":
            return _brownian_kernel(scalar)
        if self.kernel_type == "brownian_bridge":
            return _brownian_bridge_kernel(scalar)
        if self.kernel_type == "sobolev":
            return _sobolev_spline_kernel(scalar, r=self.sobolev_r)
        if self.kernel_type == "fbm":
            x1 = scalar.unsqueeze(0)
            x2 = scalar.unsqueeze(1)
            term = 0.5 * (x1.abs().pow(2 * self.H) + x2.abs().pow(2 * self.H) -
                          (x1 - x2).abs().pow(2 * self.H))
            return term

        dist = (scalar.unsqueeze(0) - scalar.unsqueeze(1)).abs()
        if self.kernel_type == "matern":
            return _matern_kernel(dist ** 2, nu=self.nu, ell=self.ell)
        if self.kernel_type == "wendland":
            return _wendland_kernel(dist, rho=self.rho)
        if self.kernel_type == "bspline":
            return _bspline_kernel(dist, order=3, h=self.bandwidth)
        if self.kernel_type == "fejer":
            return _fejer_kernel(dist, omega0=self.omega0)
        if self.kernel_type == "spectral_mix":
            return _spectral_mixture_kernel(dist ** 2, self.sm_sigmas, self.sm_weights)

        raise ValueError(f"Unknown kernel_type '{self.kernel_type}'")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        K = self._compute_kernel(x, y)
        # 블록 분할
        K_xx, K_yy = K[:B, :B], K[B:, B:]
        K_xy = K[:B, B:]
        # unbiased 2nd-order U-statistic
        sum_xx = (K_xx.sum() - K_xx.diagonal().sum()) / (B * (B - 1))
        sum_yy = (K_yy.sum() - K_yy.diagonal().sum()) / (B * (B - 1))
        mean_xy = K_xy.mean()
        return sum_xx + sum_yy - 2.0 * mean_xy


# ================================================================
# Helper : divide_no_nan
# ================================================================
def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise a/b, but NaN or ±Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0        # NaN
    result[result == float("inf")] = 0.0
    result[result == -float("inf")] = 0.0
    return result


# ================================================================
# Composite loss :   MSE  +  λ·MMD   (multi-stride / multi-group)
# ================================================================
class MSE_MMD_Loss(nn.Module):
    def __init__(
        self,
        kernel_type: str = None,
        kernel_mul: float = 0.5,
        kernel_num: int = 5,
        bandwidth: float = 1.0,
        group_size=None,          # int 또는 리스트/튜플
        stride=None,              # int 또는 리스트/튜플
        use_mse: bool = True, 
        **kwargs,
    ):
        """
        group_size, stride는 다음과 같이 입력 가능
            ① int               : 단일 값 사용
            ② [g1, g2, ...]     : stride에도 동일한 개수 지정
            ③ [(g1,s1), (g2,s2)]: 좌변 파라미터 대신 tuples 사용 (아래 예)
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.use_mse = use_mse
        # ①~③ 모든 케이스를 (G,S) 튜플 리스트로 통일
        if isinstance(group_size, (list, tuple)) and \
           len(group_size) > 0 and isinstance(group_size[0], (list, tuple)):
            self.group_cfgs = [tuple(cfg) for cfg in group_size]
        else:
            # group_size, stride 가 둘 다 스칼라 or 리스트
            gs = group_size if isinstance(group_size, (list, tuple)) else [group_size]
            st = stride      if isinstance(stride,     (list, tuple)) else [stride]
            if len(st) == 1 and len(gs) > 1:
                st = st * len(gs)
            assert len(gs) == len(st), \
                "`group_size`와 `stride`의 길이가 달라 맞춰줄 수 없습니다."
            self.group_cfgs = list(zip(gs, st))

        self.mmd = MultiKernelMMD(
            kernel_type=kernel_type,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            bandwidth=bandwidth,
            **kwargs,
        )

    # ------------------------------------------------------------
    # λ 스케줄 (예시)
    # ------------------------------------------------------------

    def _lambda_schedule(self, epoch: int) -> float:
        if epoch < 10 or 200 <= epoch <= 230:  # warm-up 기간
            return 0.0
        else:
            return 5.0

    # ------------------------------------------------------------
    # 시퀀스 그룹핑 : staticmethod 로 변환
    # ------------------------------------------------------------
    @staticmethod
    def _group_sequence(x: torch.Tensor, G: int, S: int) -> torch.Tensor:
        """
        x : (B, T) → (B * num_groups, G)
        """
        B, T = x.shape
        if T < G:                         # 더 짧으면 그룹핑 생략
            return x
        num_groups = (T - G) // S + 1
        chunks = [x[:, i * S : i * S + G] for i in range(num_groups)]
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, trainer, pred: torch.Tensor, target: torch.Tensor):
        """
        pred / target : (B, C, …)
        """
        epoch = int(getattr(trainer, "epoch", getattr(trainer, "cur_epoch", 0)))
        loss_mse = self.mse_loss(pred, target)
        lam = 5.0
        if self.use_mse: 
            lam = self._lambda_schedule(epoch)
        if lam == 0.0:
            return loss_mse

        B, C = pred.shape[:2]
        mmd_vals = []

        # 모든 채널, 모든 (group_size, stride) 조합에 대해 MMD 측정
        for c in range(C):
            pc = pred[:, c].reshape(B, -1)      # (B, T)
            tc = target[:, c].reshape(B, -1)
            for G, S in self.group_cfgs:
                pc_g = self._group_sequence(pc, G, S)
                tc_g = self._group_sequence(tc, G, S)
                mmd_vals.append(self.mmd(pc_g, tc_g))

        loss_mmd = torch.stack(mmd_vals).mean()     # channel + cfg 평균
        loss_mmd = torch.clamp(loss_mmd, min=0.0)
        loss_mmd = torch.sqrt(loss_mmd + 1e-6)
        if self.use_mse: 
            return loss_mse + lam * loss_mmd
        return  lam * loss_mmd            

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
    def __init__(self, cfg):
        self.losses = {
            # Regression losses
            "MSE": mse_loss(),
            "MAE": mae_loss(),
            "MAPE": mape_loss(),
            "SMAPE": smape_loss(),
            "MASE": mase_loss(),
            "MSE_MMD_Loss": MSE_MMD_Loss(
                kernel_type=cfg.MMD.KERNEL,
                group_size=cfg.MMD.GROUP_SIZE,
                stride=cfg.MMD.STRIDE,
                use_mse = cfg.MODEL.LOSS_USE_MSE), 
            # Classification losses
            "CE": cross_entropy_loss(),
            "Focal": focal_loss(),
            "LabelSmoothingCE": label_smoothing_ce_loss(),
        }
        
        self.metrics = self.losses