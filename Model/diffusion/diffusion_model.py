# -*- coding: utf-8 -*-
"""Model.diffusion.diffusion_model

GridDiffusion：统一封装 diffusion 模型的训练、checkpoint 管理、SDEdit 推理。

对外暴露的主入口是 diffusion_refine_pipeline()，可直接替换 main_online.py 中的 stub。
"""

from __future__ import annotations

import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

from Model.diffusion.solution_parser import SolutionParser, load_all_solutions
from Model.diffusion.solution_dataset import SolutionDataset
from Model.diffusion.trainer import DiffusionTrainer
from Model.diffusion.metrics import quality_check
from Model.diffusion.config import get_default_diffusion_config, distance_to_noise_level


# ===========================================================================
# GridDiffusion
# ===========================================================================

class GridDiffusion:
    """GridRAG Diffusion 封装：训练 / checkpoint / SDEdit 推理。"""

    def __init__(
        self,
        task: str,
        network: str,
        feature_size: int,
        diffusion_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.task = task.lower()
        self.network = network.lower()
        self.feature_size = feature_size

        self.cfg = get_default_diffusion_config()
        if diffusion_cfg:
            _deep_update(self.cfg, diffusion_cfg)

        # Device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model
        mcfg = self.cfg["model"]
        self.model = Diffusion_TS(
            seq_length=mcfg["seq_length"],
            feature_size=feature_size,
            n_layer_enc=mcfg["n_layer_enc"],
            n_layer_dec=mcfg["n_layer_dec"],
            d_model=mcfg["d_model"],
            timesteps=mcfg["timesteps"],
            sampling_timesteps=mcfg["sampling_timesteps"],
            loss_type=mcfg["loss_type"],
            beta_schedule=mcfg["beta_schedule"],
            n_heads=mcfg["n_heads"],
            mlp_hidden_times=mcfg["mlp_hidden_times"],
            attn_pd=mcfg["attn_pd"],
            resid_pd=mcfg["resid_pd"],
            kernel_size=mcfg["kernel_size"],
            padding_size=mcfg["padding_size"],
        ).to(self.device)

        self.scaler = None   # sklearn MinMaxScaler, set after training or loading
        self.trainer = None  # set during training
        self._ema_model = None  # shortcut to EMA model after loading
        self._train_data = None  # (N, T, D) 原始训练数据，用于 Context-FID 参考分布

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------

    @property
    def ckpt_dir(self) -> Path:
        root = Path(self.cfg["checkpoint_root"])
        return root / f"{self.task}_{self.network}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_database(
        self,
        opt_results_dir: str,
        verbose: bool = True,
    ) -> None:
        """从 opt_results 目录加载所有 JSON，训练 diffusion 模型。"""
        if verbose:
            print(f"[GridDiffusion] Loading solutions: task={self.task}, "
                  f"network={self.network}")

        all_cont, col_names = load_all_solutions(
            opt_results_dir, self.task, self.network
        )
        if verbose:
            print(f"  Found {all_cont.shape[0]} scenarios, "
                  f"shape per scenario: {all_cont.shape[1:]}")

        # 保存原始训练数据用于 Context-FID 参考分布
        self._train_data = all_cont.copy()

        # Build dataset
        aug_cfg = self.cfg["augmentation"]
        dataset = SolutionDataset(
            all_cont,
            augment_factor=aug_cfg["factor"],
            noise_std=aug_cfg["noise_std"],
            scale_std=aug_cfg["scale_std"],
            max_shift=aug_cfg["max_shift"],
            seed=aug_cfg["seed"],
        )
        self.scaler = dataset.scaler

        if verbose:
            print(f"  After augmentation: {len(dataset)} samples")

        # DataLoader
        bs = self.cfg["training"]["batch_size"]
        dl = DataLoader(
            dataset,
            batch_size=min(bs, len(dataset)),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        # Trainer
        self.trainer = DiffusionTrainer(
            self.model, dl, self.cfg, device=self.device
        )

        # Train
        ckpt_dir = self.ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.train(str(ckpt_dir), verbose=verbose)

        # Save scaler
        dataset.save_scaler(str(ckpt_dir / "scaler.pkl"))

        # Save training data for Context-FID reference
        np.save(str(ckpt_dir / "train_data.npy"), all_cont)

        # Save col_names for reference
        with open(ckpt_dir / "col_names.json", "w") as f:
            json.dump(col_names, f)

        # Save config snapshot
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(self.cfg, f, indent=2, default=str)

        self._ema_model = self.trainer.ema.ema_model

    # ------------------------------------------------------------------
    # Checkpoint load
    # ------------------------------------------------------------------

    def load_checkpoint(self, milestone: Optional[int] = None) -> bool:
        """加载 checkpoint + scaler。
        
        Returns True if successful, False if no checkpoint found.
        """
        ckpt_dir = self.ckpt_dir
        scaler_path = ckpt_dir / "scaler.pkl"

        if not ckpt_dir.exists() or not scaler_path.exists():
            return False

        # Load scaler
        self.scaler = SolutionDataset.load_scaler(str(scaler_path))

        # Find milestone
        if milestone is None:
            # 创建临时 trainer 来找最新 milestone
            dummy_dl = DataLoader(
                torch.utils.data.TensorDataset(torch.zeros(2, 96, self.feature_size)),
                batch_size=2,
            )
            self.trainer = DiffusionTrainer(
                self.model, dummy_dl, self.cfg, device=self.device
            )
            milestone = self.trainer.get_latest_milestone(str(ckpt_dir))
            if milestone is None:
                return False

        if self.trainer is None:
            dummy_dl = DataLoader(
                torch.utils.data.TensorDataset(torch.zeros(2, 96, self.feature_size)),
                batch_size=2,
            )
            self.trainer = DiffusionTrainer(
                self.model, dummy_dl, self.cfg, device=self.device
            )

        self.trainer.load(str(ckpt_dir), milestone)
        self._ema_model = self.trainer.ema.ema_model

        # 加载训练数据用于 Context-FID
        train_data_path = ckpt_dir / "train_data.npy"
        if train_data_path.exists():
            self._train_data = np.load(str(train_data_path))
        return True

    def has_checkpoint(self) -> bool:
        """检查是否存在可用的 checkpoint。"""
        ckpt_dir = self.ckpt_dir
        if not ckpt_dir.exists():
            return False
        scaler_path = ckpt_dir / "scaler.pkl"
        if not scaler_path.exists():
            return False
        # Check for at least one checkpoint file
        return any(ckpt_dir.glob("checkpoint-*.pt"))

    # ------------------------------------------------------------------
    # SDEdit 推理
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sdedit(
        self,
        x_raw: np.ndarray,
        noise_level: float,
        n_samples: int = 4,
        return_all_samples: bool = False,
    ):
        """SDEdit: 给检索解加噪 → 去噪 → refined 解。

        Parameters
        ----------
        x_raw : np.ndarray, shape (T, D)
            原始单位的检索解连续矩阵
        noise_level : float, 0~1
            加噪程度 (映射到扩散时间步 t_0 = int(noise_level * T))
        n_samples : int
            采样数，取均值作为最终输出
        return_all_samples : bool
            若 True，额外返回所有独立采样 (n_samples, T, D)，用于 FID 评估

        Returns
        -------
        refined : np.ndarray, shape (T, D)
            refined 后的连续矩阵（原始单位，多样本均值）
        all_samples : np.ndarray, shape (n_samples, T, D), only if return_all_samples=True
            所有独立采样的 refined 结果（原始单位）
        """
        assert self.scaler is not None, "Must train or load checkpoint first"
        assert self._ema_model is not None, "Must train or load checkpoint first"

        model = self._ema_model
        device = self.device
        T_diffusion = model.num_timesteps

        # 1) 归一化到 [-1, 1]
        x_normed = self.scaler.transform(x_raw.reshape(-1, x_raw.shape[-1]))
        x_normed = normalize_to_neg_one_to_one(x_normed)
        x_normed = x_normed.reshape(x_raw.shape)

        # 2) 转 tensor, 扩展为 batch
        x_t = torch.from_numpy(x_normed).float().to(device)
        x_batch = x_t.unsqueeze(0).repeat(n_samples, 1, 1)  # (n_samples, T, D)

        # 3) 确定加噪时间步 t_0
        t_0 = max(1, min(int(noise_level * T_diffusion), T_diffusion - 1))
        t_tensor = torch.full((n_samples,), t_0, device=device, dtype=torch.long)

        # 4) q_sample: 加噪到 t_0
        x_noisy = model.q_sample(x_start=x_batch, t=t_tensor)

        # 5) reverse denoise: 从 t_0 逐步去噪到 t=0
        if model.fast_sampling and model.sampling_timesteps < T_diffusion:
            # 使用 DDIM-like fast sampling
            x_refined = self._fast_sdedit(model, x_noisy, t_0, n_samples)
        else:
            # 标准 DDPM reverse
            img = x_noisy
            for t in reversed(range(0, t_0)):
                img, _ = model.p_sample(img, t)
            x_refined = img

        # 6) 反归一化所有样本
        x_all = x_refined.cpu().numpy()  # (n_samples, T, D)
        all_refined = []
        for i in range(n_samples):
            xi = unnormalize_to_zero_to_one(x_all[i])
            xi = self.scaler.inverse_transform(
                xi.reshape(-1, x_raw.shape[-1])
            ).reshape(x_raw.shape)
            xi = np.maximum(xi, np.minimum(x_raw, 0.0))
            all_refined.append(xi)

        # 7) 取均值作为最终输出
        refined = np.mean(all_refined, axis=0)  # (T, D)

        if return_all_samples:
            return refined, np.stack(all_refined, axis=0)  # (T,D), (n_samples,T,D)
        return refined

    def _fast_sdedit(
        self,
        model: Diffusion_TS,
        x_noisy: torch.Tensor,
        t_0: int,
        n_samples: int,
    ) -> torch.Tensor:
        """DDIM-style fast SDEdit: 只在 t_0 以下的子集步上做去噪。"""
        total_timesteps = model.num_timesteps
        sampling_timesteps = model.sampling_timesteps
        eta = model.eta
        device = x_noisy.device

        # 构建子采样时间步序列, 只取 <= t_0 的部分
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = [
            (t_now, t_next) for t_now, t_next in zip(times[:-1], times[1:])
            if t_now <= t_0
        ]

        img = x_noisy
        for t_now, t_next in time_pairs:
            time_cond = torch.full(
                (n_samples,), t_now, device=device, dtype=torch.long
            )
            pred_noise, x_start, *_ = model.model_predictions(
                img, time_cond, clip_x_start=True
            )
            if t_next < 0:
                img = x_start
                continue

            alpha = model.alphas_cumprod[t_now]
            alpha_next = model.alphas_cumprod[t_next]
            sigma = eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img


# ===========================================================================
# 顶层 pipeline 函数 (替换 main_online.py 中的 stub)
# ===========================================================================

def diffusion_refine_pipeline(
    retrieved_json: Dict[str, Any],
    warm_start: Dict[str, Any],
    query_features: np.ndarray,
    best_distance: float,
    task: str,
    network: str,
    s_base: float,
    opt_results_dir: str,
    diffusion_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Diffusion refine 完整 pipeline。

    Parameters
    ----------
    retrieved_json : dict
        检索到的历史 JSON 结果（原始格式）
    warm_start : dict
        已由 WarmStartExtractor 处理的 warm-start（备用，质量不过关时直接返回）
    query_features : np.ndarray
        查询场景特征向量
    best_distance : float
        检索距离
    task : str
    network : str
    s_base : float
    opt_results_dir : str
        opt_results 目录路径
    diffusion_cfg : dict, optional
        覆盖默认配置

    Returns
    -------
    refined_warm_start : dict
        refined warm-start（格式与原 warm_start 一致）
    diff_info : dict
        diffusion 相关信息（用于日志）
    """
    info = {
        "diffusion_used": False,
        "noise_level": 0.0,
        "quality_passed": False,
        "metrics": {},
        "training_time_sec": 0.0,    # 离线部分，不计入在线时间
        "inference_time_sec": 0.0,   # 在线部分（SDEdit + 质量检查）
        "time_sec": 0.0,             # 总时间（仅用于日志参考）
    }
    tic = time.perf_counter()

    # 1) 解析检索解
    parser = SolutionParser(task)
    try:
        continuous, discrete, col_names = parser.parse_json(retrieved_json)
    except Exception as e:
        print(f"[Diffusion] Failed to parse retrieved JSON: {e}")
        info["error"] = str(e)
        info["time_sec"] = time.perf_counter() - tic
        return warm_start, info

    feature_size = continuous.shape[1]
    if feature_size == 0:
        print("[Diffusion] No continuous variables to refine")
        info["time_sec"] = time.perf_counter() - tic
        return warm_start, info

    # 2) 构建 GridDiffusion
    cfg = diffusion_cfg or {}
    gd = GridDiffusion(task, network, feature_size, diffusion_cfg=cfg)

    # 3) 检查/训练 checkpoint
    t_train_start = time.perf_counter()
    if not gd.has_checkpoint():
        print(f"[Diffusion] No checkpoint found for {task}_{network}, "
              f"training from database...")
        try:
            gd.train_from_database(opt_results_dir, verbose=True)
        except Exception as e:
            print(f"[Diffusion] Training failed: {e}")
            info["error"] = str(e)
            info["time_sec"] = time.perf_counter() - tic
            return warm_start, info
    else:
        loaded = gd.load_checkpoint()
        if not loaded:
            print("[Diffusion] Failed to load checkpoint")
            info["time_sec"] = time.perf_counter() - tic
            return warm_start, info
    info["training_time_sec"] = time.perf_counter() - t_train_start

    # 4) SDEdit
    t_infer_start = time.perf_counter()
    sde_cfg = gd.cfg["sdedit"]
    noise_level = distance_to_noise_level(
        best_distance,
        noise_min=sde_cfg["noise_level_min"],
        noise_max=sde_cfg["noise_level_max"],
        dist_low=sde_cfg["distance_low"],
        dist_high=sde_cfg["distance_high"],
    )
    info["noise_level"] = noise_level

    try:
        refined_cont, all_samples = gd.sdedit(
            continuous,
            noise_level=noise_level,
            n_samples=sde_cfg["n_samples"],
            return_all_samples=True,
        )
    except Exception as e:
        print(f"[Diffusion] SDEdit failed: {e}")
        info["error"] = str(e)
        info["time_sec"] = time.perf_counter() - tic
        return warm_start, info
    info["inference_time_sec"] = time.perf_counter() - t_infer_start
    info["time_sec"] = time.perf_counter() - tic
    info["diffusion_used"] = True

    # 5) 质量检查
    #    ori_data: 数据库训练解 (N, T, D)，作为参考分布
    #    generated_data: SDEdit 所有独立采样 (n_samples, T, D)，作为生成分布
    q_cfg = gd.cfg["quality"]
    if gd._train_data is not None and gd._train_data.shape[0] >= 2:
        ori_batch = gd._train_data  # (N, T, D)
    else:
        # 回退：无训练数据时，用检索解扩展（质量检查将以 NaN 通过）
        ori_batch = continuous[np.newaxis, ...]  # (1, T, D)
    gen_batch = all_samples  # (n_samples, T, D)

    passed, metrics = quality_check(
        ori_batch,
        gen_batch,
        fid_threshold=q_cfg["context_fid_threshold"],
        corr_threshold=q_cfg["cross_corr_threshold"],
        enable=q_cfg["enable_quality_check"],
    )
    info["quality_passed"] = passed
    info["metrics"] = metrics

    if not passed:
        print(f"[Diffusion] Quality check failed: {metrics}, "
              f"falling back to retrieved solution")
        info["time_sec"] = time.perf_counter() - tic
        return warm_start, info

    # 6) 重组 warm-start
    refined_ws = parser.to_warmstart(refined_cont, discrete, col_names, s_base)
    info["time_sec"] = time.perf_counter() - tic

    print(f"[Diffusion] SDEdit done: noise_level={noise_level:.2f}, "
          f"metrics={metrics}, time={info['time_sec']:.2f}s")

    return refined_ws, info


# ===========================================================================
# Helpers
# ===========================================================================

def _deep_update(base: dict, override: dict) -> dict:
    """递归更新嵌套字典。"""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
