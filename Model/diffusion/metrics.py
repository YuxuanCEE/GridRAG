# -*- coding: utf-8 -*-
"""Model.diffusion.metrics

迁移自 Diffusion-TS 的 Context-FID 和 Cross-Correlation 指标。

用途：
1. 评估 diffusion 输出质量：与数据库原始解对比
2. 质量门控：指标过差时退回使用原检索解
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ===========================================================================
# Context-FID（基于 TS2Vec 表征的 FID 距离）
# ===========================================================================

def _calculate_fid(act1: np.ndarray, act2: np.ndarray) -> float:
    """计算两组激活向量之间的 FID 距离。"""
    import scipy.linalg

    # 样本数守卫：协方差矩阵至少需要 2 个样本（自由度 = N-1 >= 1）
    if act1.shape[0] < 2 or act2.shape[0] < 2:
        print(f"[Warning] FID requires >=2 samples per group, "
              f"got {act1.shape[0]} and {act2.shape[0]}")
        return float("nan")

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # 处理退化情况（样本数太少导致协方差矩阵奇异）
    if sigma1.ndim < 2:
        sigma1 = np.atleast_2d(sigma1)
    if sigma2.ndim < 2:
        sigma2 = np.atleast_2d(sigma2)

    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_context_fid(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    device: int = 0,
) -> float:
    """计算 Context-FID。

    Parameters
    ----------
    ori_data : (N, T, D) 原始数据
    generated_data : (M, T, D) 生成数据
    device : CUDA device id, -1 for CPU

    Returns
    -------
    fid : float
    """
    try:
        from models.ts2vec.ts2vec import TS2Vec
    except ImportError:
        # 如果 TS2Vec 不可用，返回一个大值但不崩溃
        print("[Warning] TS2Vec not available, Context-FID set to NaN")
        return float("nan")

    n_features = ori_data.shape[-1]
    actual_device = device if torch.cuda.is_available() and device >= 0 else "cpu"

    model = TS2Vec(
        input_dims=n_features,
        device=actual_device,
        batch_size=min(8, ori_data.shape[0]),
        lr=0.001,
        output_dims=min(320, max(64, n_features * 8)),
        max_train_length=3000,
    )
    model.fit(ori_data, verbose=False)
    ori_repr = model.encode(ori_data, encoding_window="full_series")
    gen_repr = model.encode(generated_data, encoding_window="full_series")

    # 对齐样本数
    n = min(ori_repr.shape[0], gen_repr.shape[0])
    idx = np.random.permutation(n)
    ori_repr = ori_repr[idx]
    gen_repr = gen_repr[:n][idx]

    return _calculate_fid(ori_repr, gen_repr)


# ===========================================================================
# Cross-Correlation
# ===========================================================================

def _cacf_torch(x: torch.Tensor, max_lag: int, dim: tuple = (0, 1)):
    """Cross-autocorrelation function (torch)."""
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / (x.std(dim, keepdims=True) + 1e-8)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = []
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, 1)
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def compute_cross_correlation(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
) -> float:
    """计算 Cross-Correlation Loss。

    Parameters
    ----------
    ori_data : (N, T, D) 原始数据
    generated_data : (M, T, D) 生成数据

    Returns
    -------
    corr_loss : float (越小越好)
    """
    if ori_data.shape[-1] < 2:
        # 只有 1 个特征时无法计算跨通道相关
        return 0.0

    x_real = torch.from_numpy(ori_data).float()
    x_fake = torch.from_numpy(generated_data).float()

    cross_correl_real = _cacf_torch(x_real, 1).mean(0)[0]
    cross_correl_fake = _cacf_torch(x_fake, 1).mean(0)[0]

    loss = torch.abs(cross_correl_fake - cross_correl_real).sum(0)
    return float(loss.item() / 10.0)


# ===========================================================================
# 质量门控
# ===========================================================================

def quality_check(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    fid_threshold: float = 50.0,
    corr_threshold: float = 1.0,
    enable: bool = True,
) -> Tuple[bool, Dict[str, float]]:
    """评估 diffusion 输出质量。

    Returns
    -------
    passed : bool
        True = 质量合格，可使用 refined 解
        False = 质量过差，应退回原检索解
    metrics : dict
        {"context_fid": float, "cross_correlation": float}
    """
    if not enable:
        return True, {"context_fid": float("nan"), "cross_correlation": float("nan")}

    metrics = {}

    try:
        fid = compute_context_fid(ori_data, generated_data)
        metrics["context_fid"] = fid
    except Exception as e:
        print(f"[Warning] Context-FID computation failed: {e}")
        metrics["context_fid"] = float("nan")
        fid = 0.0  # 不因计算失败而拒绝

    try:
        corr = compute_cross_correlation(ori_data, generated_data)
        metrics["cross_correlation"] = corr
    except Exception as e:
        print(f"[Warning] Cross-Correlation computation failed: {e}")
        metrics["cross_correlation"] = float("nan")
        corr = 0.0

    fid_ok = np.isnan(fid) or fid <= fid_threshold
    corr_ok = np.isnan(corr) or corr <= corr_threshold

    return (fid_ok and corr_ok), metrics
