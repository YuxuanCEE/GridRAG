# -*- coding: utf-8 -*-
"""
RL Baseline 数据加载器 - Task A (VVC)

功能：
- 从场景CSV加载PV、WT出力
- 从优化结果JSON加载专家动作标签（OLTC档位、SC状态、PV/WT/SVC无功）
- 数据增强（高斯噪声、负荷缩放，仅对输入特征）
- 划分训练验证集
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config_networks import get_network_config, PRICE_CONFIG


class VVCDataProcessor:
    """VVC 任务数据处理器"""

    def __init__(self, network_name: str, scenario_id: str, project_root: Path = None):
        self.network_name = network_name.lower()
        self.scenario_id = scenario_id

        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root

        self.config = get_network_config(network_name, project_root)

        # 路径
        self.profiles_dir = project_root / "data" / "profiles"
        self.results_dir = project_root / "opt_results" / "vvc"

        # 网络参数
        self.n_buses = self.config["network"]["n_buses"]
        self.n_periods = self.config["optimization"]["day_ahead"]["n_periods"]  # 96

        # 设备配置
        self.pv_config = self.config["devices"]["pv"]
        self.wt_config = self.config["devices"]["wt"]
        self.sc_config = self.config["devices"]["sc"]
        self.svc_config = self.config["devices"]["svc"]
        self.oltc_config = self.config["devices"]["oltc"]

        # 设备数量
        self.n_pv = len(self.pv_config["buses"])
        self.n_wt = len(self.wt_config["buses"])
        self.n_sc = len(self.sc_config["buses"])
        self.n_svc = len(self.svc_config["buses"])

        # OLTC参数
        self.tap_min = self.oltc_config["tap_min"]
        self.tap_max = self.oltc_config["tap_max"]
        self.n_oltc_actions = self.tap_max - self.tap_min + 1  # e.g., 11 for [-5,5]

        # SC参数
        self.n_sc_stages = self.sc_config["n_stages"]  # e.g., 3

        # 连续动作归一化系数（用于将 MVar 归一化到 [-1,1]）
        self.pv_q_max = self.pv_config["capacity"] if self.n_pv > 0 else []
        self.wt_q_max = self.wt_config["capacity"] if self.n_wt > 0 else []
        self.svc_q_max = abs(self.svc_config["q_max"]) if self.n_svc > 0 else 0.4

        # 标准化器
        self.input_scaler = None

    def _get_scenario_file(self) -> Path:
        filename = f"scenario_{self.scenario_id}_{self.n_buses}.csv"
        return self.profiles_dir / filename

    def _get_result_file(self) -> Path:
        filename = f"{self.network_name}_vvc_scenario_{self.scenario_id}_results.json"
        return self.results_dir / filename

    def _get_price_profile(self) -> np.ndarray:
        price = np.zeros(self.n_periods)
        peak_hours = set(PRICE_CONFIG["peak_hours"])
        valley_hours = set(PRICE_CONFIG["valley_hours"])
        for t in range(self.n_periods):
            hour = (t * 15 // 60) + 1
            if hour > 24:
                hour -= 24
            if hour in peak_hours:
                price[t] = PRICE_CONFIG["peak_price"]
            elif hour in valley_hours:
                price[t] = PRICE_CONFIG["valley_price"]
            else:
                price[t] = PRICE_CONFIG["flat_price"]
        return price

    def _get_time_encoding(self) -> np.ndarray:
        time_features = np.zeros((self.n_periods, 4))
        for t in range(self.n_periods):
            hour = (t * 15 / 60) % 24
            time_features[t, 0] = np.sin(2 * np.pi * hour / 24)
            time_features[t, 1] = np.cos(2 * np.pi * hour / 24)
            time_features[t, 2] = t / self.n_periods
            hour_int = int(hour) + 1
            time_features[t, 3] = 1.0 if hour_int in PRICE_CONFIG["peak_hours"] else 0.0
        return time_features

    def _get_static_features(self) -> np.ndarray:
        static = []
        static.append(self.tap_min / 10.0)
        static.append(self.tap_max / 10.0)
        static.append(self.oltc_config["tap_step"] * 100)
        static.append(self.n_sc / 5.0)
        static.append(self.n_sc_stages / 5.0)
        static.append(self.n_pv / 5.0)
        static.append(self.n_wt / 5.0)
        static.append(self.n_svc / 5.0)
        static.append(self.n_buses / 130.0)
        return np.array(static, dtype=np.float32)

    def _get_load_curve(self) -> np.ndarray:
        """
        生成典型日负荷变化曲线（标准化因子）
        
        ★ 必须与 data/data_loader.py → DataLoader.get_load_curve() 完全一致
        优化器通过该曲线缩放base负荷: p_load[t] = p_load_pu * load_factor[t]
        """
        hours = np.linspace(0, 24, self.n_periods, endpoint=False)
        
        # 典型居民/商业混合负荷曲线
        load_curve = (
            0.4  # 基础负荷
            + 0.15 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 22)  # 白天高峰
            + 0.25 * np.exp(-((hours - 12) ** 2) / 8)  # 中午高峰
            + 0.2 * np.exp(-((hours - 19) ** 2) / 4)   # 晚高峰
            - 0.1 * np.exp(-((hours - 4) ** 2) / 4)    # 凌晨低谷
        )
        
        # 归一化到0.3-1.0范围
        load_curve = 0.3 + 0.7 * (load_curve - load_curve.min()) / (load_curve.max() - load_curve.min())
        
        return load_curve.astype(np.float64)

    def load_scenario_data(self) -> Dict[str, np.ndarray]:
        """加载场景输入数据（PV/WT容量因子 + 负荷因子 + 时间特征 + 电价 + 静态特征）"""
        scenario_file = self._get_scenario_file()
        if not scenario_file.exists():
            raise FileNotFoundError(f"场景文件不存在: {scenario_file}")

        df = pd.read_csv(scenario_file)

        pv_data = np.zeros((self.n_periods, self.n_pv))
        for i, col in enumerate(self.pv_config["columns"]):
            if col in df.columns:
                pv_data[:, i] = df[col].values[:self.n_periods]

        wt_data = np.zeros((self.n_periods, self.n_wt))
        for i, col in enumerate(self.wt_config["columns"]):
            if col in df.columns:
                wt_data[:, i] = df[col].values[:self.n_periods]

        # 负荷因子：优先从CSV读取，否则用程序化生成的典型日负荷曲线
        # ★ 必须与 data/data_loader.py 的 get_load_curve() 一致
        if "load_factor" in df.columns:
            load_factor = df["load_factor"].values[:self.n_periods].astype(np.float64)
        else:
            load_factor = self._get_load_curve()

        return {
            "pv": pv_data,
            "wt": wt_data,
            "load_factor": load_factor,
            "price": self._get_price_profile(),
            "time_encoding": self._get_time_encoding(),
            "static": self._get_static_features(),
        }

    def load_optimization_results(self) -> Dict[str, np.ndarray]:
        """加载优化器专家动作标签"""
        result_file = self._get_result_file()
        if not result_file.exists():
            raise FileNotFoundError(f"VVC结果文件不存在: {result_file}")

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # OLTC tap
        oltc_tap = np.array(data["oltc"]["tap"], dtype=np.float64)  # (96,)

        # SC stage
        sc_stage = np.array(data["sc"]["stage"], dtype=np.float64)  # (96, n_sc)
        if sc_stage.ndim == 1:
            sc_stage = sc_stage.reshape(-1, 1)

        # PV reactive (MVar)
        pv_q = np.array(data["pv_reactive"]["q_mvar"], dtype=np.float64)
        if pv_q.ndim == 1:
            pv_q = pv_q.reshape(-1, 1)

        # WT reactive (MVar)
        wt_q = np.array(data["wt_reactive"]["q_mvar"], dtype=np.float64)
        if wt_q.ndim == 1 and self.n_wt > 0:
            wt_q = wt_q.reshape(-1, 1)

        # SVC reactive (MVar)
        svc_q = np.array(data["svc_reactive"]["q_mvar"], dtype=np.float64)
        if svc_q.ndim == 1:
            svc_q = svc_q.reshape(-1, 1)

        # 参考指标
        voltage_values = np.array(data["voltage"]["values"], dtype=np.float64)
        loss_per_period = np.array(data["loss"]["per_period_kw"], dtype=np.float64)
        loss_total = float(data["loss"]["total_kw"])

        return {
            "oltc_tap": oltc_tap,       # (96,)
            "sc_stage": sc_stage,       # (96, n_sc)
            "pv_q_mvar": pv_q,          # (96, n_pv)
            "wt_q_mvar": wt_q,          # (96, n_wt)
            "svc_q_mvar": svc_q,        # (96, n_svc)
            "voltage": voltage_values,  # (96, n_buses)
            "loss_kw": loss_per_period,
            "loss_total_kw": loss_total,
        }

    def prepare_input_tensor(self, scenario_data: Dict) -> np.ndarray:
        """整理为模型输入: (n_periods, n_features)
        
        特征列: [PV容量因子(n_pv), WT容量因子(n_wt), 负荷因子(1), 电价(1), 时间编码(4), 静态(9)]
        """
        pv = scenario_data["pv"]
        wt = scenario_data["wt"]
        load_factor = scenario_data.get("load_factor", np.ones(self.n_periods))
        load_factor = load_factor.reshape(-1, 1)
        price = scenario_data["price"].reshape(-1, 1)
        time_enc = scenario_data["time_encoding"]
        static = scenario_data["static"]
        static_broadcast = np.tile(static, (self.n_periods, 1))

        input_tensor = np.concatenate([pv, wt, load_factor, price, time_enc, static_broadcast], axis=1)
        return input_tensor.astype(np.float32)

    def prepare_action_labels(self, opt_results: Dict) -> Dict[str, np.ndarray]:
        """
        整理专家动作标签

        Returns:
            oltc_class: (96,) 整数类别索引 [0, n_oltc_actions)
            sc_class:   (96, n_sc) 整数类别索引 [0, n_sc_stages]
            pv_q_norm:  (96, n_pv) float [-1, 1]
            wt_q_norm:  (96, n_wt) float [-1, 1]
            svc_q_norm: (96, n_svc) float [-1, 1]
        """
        # OLTC: 整数 tap → 类别索引
        oltc_tap = opt_results["oltc_tap"]
        oltc_class = np.round(oltc_tap).astype(np.int64) - self.tap_min  # e.g., -5→0, 5→10
        oltc_class = np.clip(oltc_class, 0, self.n_oltc_actions - 1)

        # SC stage: 直接使用
        sc_class = np.round(opt_results["sc_stage"]).astype(np.int64)
        sc_class = np.clip(sc_class, 0, self.n_sc_stages)

        # PV Q: 归一化到 [-1, 1]
        pv_q_norm = np.zeros_like(opt_results["pv_q_mvar"])
        for i in range(self.n_pv):
            q_max = self.pv_q_max[i] if i < len(self.pv_q_max) else 0.5
            if q_max > 1e-6:
                pv_q_norm[:, i] = opt_results["pv_q_mvar"][:, i] / q_max
        pv_q_norm = np.clip(pv_q_norm, -1.0, 1.0)

        # WT Q: 归一化
        wt_q_norm = np.zeros_like(opt_results["wt_q_mvar"])
        for i in range(self.n_wt):
            q_max = self.wt_q_max[i] if i < len(self.wt_q_max) else 0.5
            if q_max > 1e-6:
                wt_q_norm[:, i] = opt_results["wt_q_mvar"][:, i] / q_max
        wt_q_norm = np.clip(wt_q_norm, -1.0, 1.0)

        # SVC Q: 归一化
        svc_q_norm = opt_results["svc_q_mvar"] / self.svc_q_max if self.svc_q_max > 1e-6 else opt_results["svc_q_mvar"]
        svc_q_norm = np.clip(svc_q_norm, -1.0, 1.0)

        return {
            "oltc_class": oltc_class.astype(np.int64),
            "sc_class": sc_class.astype(np.int64),
            "pv_q_norm": pv_q_norm.astype(np.float32),
            "wt_q_norm": wt_q_norm.astype(np.float32),
            "svc_q_norm": svc_q_norm.astype(np.float32),
        }

    def augment_data(self, input_data: np.ndarray,
                     n_augments: int = 50,
                     noise_std: float = 0.02,
                     scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        数据增强：对PV/WT容量因子和负荷因子添加噪声和缩放

        特征列顺序: [PV(n_pv), WT(n_wt), load_factor(1), price(1), time_enc(4), static(9)]

        Returns:
            (n_augments + 1, n_periods, n_features)
        """
        augmented = [input_data]
        n_dg = self.n_pv + self.n_wt  # 前 n_dg 列是 PV/WT
        lf_col = n_dg                  # 负荷因子列索引

        for _ in range(n_augments):
            aug = input_data.copy()
            # 高斯噪声（仅DG列）
            noise = np.random.normal(0, noise_std, (self.n_periods, n_dg))
            aug[:, :n_dg] += noise
            # DG缩放
            scale = np.random.uniform(scale_range[0], scale_range[1])
            aug[:, :n_dg] *= scale
            # 保证非负且不超过1（容量因子上限）
            aug[:, :n_dg] = np.clip(aug[:, :n_dg], 0, 1.0)

            # 负荷因子扰动（独立于DG缩放，用更小的范围）
            lf_scale = np.random.uniform(0.97, 1.03)
            lf_noise = np.random.normal(0, noise_std * 0.5, self.n_periods)
            aug[:, lf_col] = aug[:, lf_col] * lf_scale + lf_noise
            aug[:, lf_col] = np.maximum(aug[:, lf_col], 0.1)  # 负荷因子不能为负

            augmented.append(aug)

        return np.array(augmented, dtype=np.float32)

    def prepare_dataset(self, n_augments: int = 50, val_ratio: float = 0.2,
                        noise_std: float = 0.02,
                        scale_range: Tuple[float, float] = (0.95, 1.05)
                        ) -> Tuple[Dict, Dict, Dict]:
        """
        准备完整训练/验证数据集

        Returns:
            train_data, val_data, meta
        """
        scenario_data = self.load_scenario_data()
        opt_results = self.load_optimization_results()

        input_tensor = self.prepare_input_tensor(scenario_data)     # (96, F)
        action_labels = self.prepare_action_labels(opt_results)

        # 数据增强（仅输入）
        augmented_input = self.augment_data(input_tensor, n_augments, noise_std, scale_range)
        n_samples = augmented_input.shape[0]

        # 标准化输入
        input_shape = augmented_input.shape  # (N, 96, F)
        input_flat = augmented_input.reshape(-1, input_shape[-1])
        self.input_scaler = StandardScaler()
        input_normalized = self.input_scaler.fit_transform(input_flat).reshape(input_shape).astype(np.float32)

        # 扩展标签到所有增强样本（标签不变）
        oltc_exp = np.tile(action_labels["oltc_class"][np.newaxis, :], (n_samples, 1))
        sc_exp = np.tile(action_labels["sc_class"][np.newaxis, :, :], (n_samples, 1, 1))
        pv_q_exp = np.tile(action_labels["pv_q_norm"][np.newaxis, :, :], (n_samples, 1, 1))
        wt_q_exp = np.tile(action_labels["wt_q_norm"][np.newaxis, :, :], (n_samples, 1, 1))
        svc_q_exp = np.tile(action_labels["svc_q_norm"][np.newaxis, :, :], (n_samples, 1, 1))

        # 划分
        n_val = max(1, int(n_samples * val_ratio))
        indices = np.random.permutation(n_samples)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        def _split(arr, idx):
            return arr[idx]

        train_data = {
            "input": _split(input_normalized, train_idx),
            "oltc_class": _split(oltc_exp, train_idx),
            "sc_class": _split(sc_exp, train_idx),
            "pv_q_norm": _split(pv_q_exp, train_idx),
            "wt_q_norm": _split(wt_q_exp, train_idx),
            "svc_q_norm": _split(svc_q_exp, train_idx),
        }
        val_data = {
            "input": _split(input_normalized, val_idx),
            "oltc_class": _split(oltc_exp, val_idx),
            "sc_class": _split(sc_exp, val_idx),
            "pv_q_norm": _split(pv_q_exp, val_idx),
            "wt_q_norm": _split(wt_q_exp, val_idx),
            "svc_q_norm": _split(svc_q_exp, val_idx),
        }
        meta = {
            "n_features": input_tensor.shape[-1],
            "n_periods": self.n_periods,
            "n_oltc_actions": self.n_oltc_actions,
            "n_sc": self.n_sc,
            "n_sc_stages": self.n_sc_stages,
            "n_pv": self.n_pv,
            "n_wt": self.n_wt,
            "n_svc": self.n_svc,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "pv_q_max": list(self.pv_q_max),
            "wt_q_max": list(self.wt_q_max),
            "svc_q_max": float(self.svc_q_max),
            "tap_min": self.tap_min,
            "tap_max": self.tap_max,
            "sc_q_per_stage": self.sc_config["q_per_stage"],
        }
        return train_data, val_data, meta


class VVCDataset(Dataset):
    """PyTorch Dataset for VVC BC"""

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.input = torch.from_numpy(data_dict["input"])
        self.oltc_class = torch.from_numpy(data_dict["oltc_class"])
        self.sc_class = torch.from_numpy(data_dict["sc_class"])
        self.pv_q = torch.from_numpy(data_dict["pv_q_norm"])
        self.wt_q = torch.from_numpy(data_dict["wt_q_norm"])
        self.svc_q = torch.from_numpy(data_dict["svc_q_norm"])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "oltc_class": self.oltc_class[idx],
            "sc_class": self.sc_class[idx],
            "pv_q": self.pv_q[idx],
            "wt_q": self.wt_q[idx],
            "svc_q": self.svc_q[idx],
        }


def create_vvc_data_loaders(processor: VVCDataProcessor,
                            batch_size: int = 16,
                            n_augments: int = 50,
                            val_ratio: float = 0.2,
                            num_workers: int = 0
                            ) -> Tuple[DataLoader, DataLoader, Dict]:
    train_data, val_data, meta = processor.prepare_dataset(n_augments=n_augments, val_ratio=val_ratio)
    train_loader = DataLoader(VVCDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(VVCDataset(val_data), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, meta