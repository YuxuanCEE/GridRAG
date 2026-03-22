# -*- coding: utf-8 -*-
"""Model.diffusion.augmentation

对优化解时间序列做数据增强，以应对数据库样本量不足。

增强策略：
1. 加性高斯噪声：小幅度随机扰动
2. 缩放扰动：按维度微调幅度
3. 时间轴平移：roll ±1~2步（模拟微小时差）
4. 随机组合以上策略
"""

from __future__ import annotations

import numpy as np


def augment_solutions(
    data: np.ndarray,
    factor: int = 50,
    noise_std: float = 0.03,
    scale_std: float = 0.03,
    max_shift: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """对已归一化到 [-1, 1] 的解做数据增强。

    Parameters
    ----------
    data : np.ndarray, shape (N, T, D)
        原始样本（已归一化）
    factor : int
        每个样本增强出多少份（含原始）
    noise_std : float
        高斯噪声标准差
    scale_std : float
        缩放扰动的标准差
    max_shift : int
        时间轴平移最大步数
    seed : int
        随机种子

    Returns
    -------
    augmented : np.ndarray, shape (N * factor, T, D)
    """
    rng = np.random.RandomState(seed)
    N, T, D = data.shape
    augmented = [data.copy()]  # 保留原始样本

    for _ in range(factor - 1):
        # 随机选择基底样本
        base = data[rng.randint(0, N, size=N)].copy()

        # 随机选择增强策略的组合
        strategy = rng.randint(0, 4, size=N)

        for i in range(N):
            x = base[i]  # (T, D)
            s = strategy[i]

            if s == 0:
                # 纯高斯噪声
                x = x + rng.randn(T, D) * noise_std
            elif s == 1:
                # 缩放扰动
                scale = 1.0 + rng.randn(1, D) * scale_std
                x = x * scale
            elif s == 2:
                # 时间轴平移
                shift = rng.randint(-max_shift, max_shift + 1)
                x = np.roll(x, shift, axis=0)
            else:
                # 组合：噪声 + 缩放 + 平移
                scale = 1.0 + rng.randn(1, D) * scale_std * 0.5
                x = x * scale
                x = x + rng.randn(T, D) * noise_std * 0.7
                shift = rng.randint(-max_shift, max_shift + 1)
                x = np.roll(x, shift, axis=0)

            # clip 到 [-1, 1]
            base[i] = np.clip(x, -1.0, 1.0)

        augmented.append(base)

    return np.concatenate(augmented, axis=0)
