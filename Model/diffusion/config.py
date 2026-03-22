# -*- coding: utf-8 -*-
"""Model.diffusion.config

Diffusion 模块的默认超参数配置。
"""

from __future__ import annotations
from typing import Dict, Any


def get_default_diffusion_config() -> Dict[str, Any]:
    """返回默认 diffusion 配置字典。"""
    return {
        # ==== 模型 ====
        "model": {
            "seq_length": 96,
            # feature_size 会在运行时根据 (task, network) 自动确定
            "n_layer_enc": 3,
            "n_layer_dec": 2,
            "d_model": 64,
            "timesteps": 500,
            "sampling_timesteps": 500,
            "loss_type": "l2",
            "beta_schedule": "cosine",
            "n_heads": 4,
            "mlp_hidden_times": 4,
            "attn_pd": 0.0,
            "resid_pd": 0.0,
            "kernel_size": 1,
            "padding_size": 0,
        },

        # ==== 训练 ====
        "training": {
            "max_epochs": 5000,
            "batch_size": 64,
            "lr": 1.0e-4,
            "warmup_lr": 5.0e-4,
            "warmup_steps": 200,
            "gradient_accumulate_every": 2,
            "save_cycle": 1000,       # 每多少步保存一次
            "ema_decay": 0.995,
            "ema_update_every": 10,
            "scheduler": {
                "factor": 0.5,
                "patience": 1500,
                "min_lr": 1.0e-5,
                "threshold": 1.0e-1,
                "threshold_mode": "rel",
            },
        },

        # ==== 数据增强 ====
        "augmentation": {
            "factor": 100,          # 每个样本增强倍数
            "noise_std": 0.03,
            "scale_std": 0.03,
            "max_shift": 2,
            "seed": 42,
        },

        # ==== SDEdit 推理 ====
        "sdedit": {
            "noise_level_min": 0.2,   # 最小加噪比例
            "noise_level_max": 0.6,   # 最大加噪比例
            "distance_low": 0.1,      # 低于此距离 → noise_level_min
            "distance_high": 1.5,     # 高于此距离 → noise_level_max
            "n_samples": 4,           # 采样数，取平均
        },

        # ==== 质量阈值 ====
        "quality": {
            "context_fid_threshold": 50.0,
            "cross_corr_threshold": 1.0,
            "enable_quality_check": True,
        },

        # ==== 路径 ====
        "checkpoint_root": "checkpoints/diffusion",
    }


def distance_to_noise_level(
    distance: float,
    noise_min: float = 0.2,
    noise_max: float = 0.6,
    dist_low: float = 0.1,
    dist_high: float = 1.5,
) -> float:
    """将检索距离映射为 SDEdit 噪声等级。

    距离越大（越不相似） → 噪声等级越高 → 修正幅度越大。
    线性映射并 clip 到 [noise_min, noise_max]。
    """
    if distance <= dist_low:
        return noise_min
    if distance >= dist_high:
        return noise_max
    ratio = (distance - dist_low) / (dist_high - dist_low)
    return noise_min + ratio * (noise_max - noise_min)
