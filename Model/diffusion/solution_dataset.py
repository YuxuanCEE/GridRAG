# -*- coding: utf-8 -*-
"""Model.diffusion.solution_dataset

自定义 PyTorch Dataset：将从数据库 JSON 提取的优化解矩阵
经 MinMaxScaler → [-1,1] 归一化 → 数据增强后供 Diffusion-TS 训练。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from Model.diffusion.augmentation import augment_solutions


class SolutionDataset(Dataset):
    """优化解时间序列数据集。

    Parameters
    ----------
    solutions : np.ndarray, shape (N, T, D)
        原始单位的连续控制量矩阵
    augment_factor : int
        数据增强倍数（含原始）
    neg_one_to_one : bool
        是否映射到 [-1, 1]
    seed : int
        增强随机种子
    """

    def __init__(
        self,
        solutions: np.ndarray,
        augment_factor: int = 50,
        neg_one_to_one: bool = True,
        noise_std: float = 0.03,
        scale_std: float = 0.03,
        max_shift: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        N, T, D = solutions.shape
        self.window = T
        self.var_num = D
        self.auto_norm = neg_one_to_one

        # 1) Fit MinMaxScaler on all raw data
        flat = solutions.reshape(-1, D)
        self.scaler = MinMaxScaler()
        self.scaler.fit(flat)

        # 2) Transform → [0, 1]
        normed = self.scaler.transform(flat).reshape(N, T, D)

        # 3) → [-1, 1]
        if neg_one_to_one:
            normed = normalize_to_neg_one_to_one(normed)

        # 4) 数据增强
        if augment_factor > 1:
            self.samples = augment_solutions(
                normed,
                factor=augment_factor,
                noise_std=noise_std,
                scale_std=scale_std,
                max_shift=max_shift,
                seed=seed,
            )
        else:
            self.samples = normed

        self.sample_num = self.samples.shape[0]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """将原始单位的 (T, D) 或 (N, T, D) 归一化到 [-1,1]。"""
        shape = x.shape
        flat = x.reshape(-1, self.var_num)
        out = self.scaler.transform(flat)
        if self.auto_norm:
            out = normalize_to_neg_one_to_one(out)
        return out.reshape(shape)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """将 [-1,1] 的数据还原为原始单位。"""
        shape = x.shape
        flat = x.reshape(-1, self.var_num)
        if self.auto_norm:
            flat = unnormalize_to_zero_to_one(flat)
        out = self.scaler.inverse_transform(flat)
        return out.reshape(shape)

    def save_scaler(self, path: str):
        """保存 scaler 供推理时使用。"""
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @staticmethod
    def load_scaler(path: str) -> MinMaxScaler:
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, ind):
        x = self.samples[ind]  # (T, D)
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
