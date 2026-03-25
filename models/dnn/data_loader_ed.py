# -*- coding: utf-8 -*-
"""
DNN Baseline 数据加载器 - Task B (ED)

功能：
- 从场景CSV文件加载输入特征（PV、WT、电价等）
- 从优化结果JSON文件加载输出标签（ESS调度、购电功率、联络线状态）
- 数据增强（高斯噪声、负荷缩放）
- 数据标准化/归一化
- 划分训练验证集
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config_networks import get_network_config, PRICE_CONFIG


class EDDataProcessor:
    """ED任务数据处理器"""

    def __init__(self, network_name: str, scenario_id: str, project_root: Path = None):
        """
        初始化数据处理器

        Args:
            network_name: 网络名称 (ieee13, ieee33, ieee69, ieee123)
            scenario_id: 场景编号 (如 "004")
            project_root: 项目根目录
        """
        self.network_name = network_name.lower()
        self.scenario_id = scenario_id

        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root

        # 加载网络配置
        self.config = get_network_config(network_name, project_root)

        # 路径配置
        self.profiles_dir = project_root / "data" / "profiles"
        self.results_dir = project_root / "opt_results" / "ed"

        # 网络参数
        self.n_buses = self.config["network"]["n_buses"]
        self.n_periods = self.config["optimization"]["ed"]["n_periods"]  # 96
        self.delta_t = self.config["optimization"]["ed"]["delta_t"]  # 0.25

        # 设备配置
        self.pv_config = self.config["devices"]["pv"]
        self.wt_config = self.config["devices"]["wt"]
        self.ess_config = self.config["devices"]["ess"]
        self.tie_config = self.config["devices"]["tie_switches"]

        # 设备数量
        self.n_pv = len(self.pv_config["buses"])
        self.n_wt = len(self.wt_config["buses"])
        self.n_ess = len(self.ess_config["buses"])
        self.n_ties = len(self.tie_config.get("branches", []))

        # 标准化器
        self.input_scaler = None
        self.output_scaler = None

        # 原始数据
        self.raw_input = None
        self.raw_output = None

    def _get_scenario_file(self) -> Path:
        """获取场景数据文件路径"""
        # 格式: scenario_{scenario_id}_{n_buses}.csv
        filename = f"scenario_{self.scenario_id}_{self.n_buses}.csv"
        return self.profiles_dir / filename

    def _get_result_file(self) -> Path:
        """获取优化结果文件路径"""
        # 格式: {network}_ed_scenario_{scenario_id}_results.json
        filename = f"{self.network_name}_ed_scenario_{self.scenario_id}_results.json"
        return self.results_dir / filename

    def _get_price_profile(self) -> np.ndarray:
        """生成分时电价序列"""
        price = np.zeros(self.n_periods)

        peak_hours = set(PRICE_CONFIG["peak_hours"])
        valley_hours = set(PRICE_CONFIG["valley_hours"])

        peak_price = PRICE_CONFIG["peak_price"]
        valley_price = PRICE_CONFIG["valley_price"]
        flat_price = PRICE_CONFIG["flat_price"]

        for t in range(self.n_periods):
            # 每15分钟一个时段，计算对应小时
            hour = (t * 15 // 60) + 1
            if hour > 24:
                hour = hour - 24

            if hour in peak_hours:
                price[t] = peak_price
            elif hour in valley_hours:
                price[t] = valley_price
            else:
                price[t] = flat_price

        return price

    def _get_time_encoding(self) -> np.ndarray:
        """生成时间编码特征"""
        time_features = np.zeros((self.n_periods, 4))

        for t in range(self.n_periods):
            # 时间（小时）
            hour = (t * 15 / 60) % 24
            # 正弦/余弦编码（捕捉周期性）
            time_features[t, 0] = np.sin(2 * np.pi * hour / 24)
            time_features[t, 1] = np.cos(2 * np.pi * hour / 24)
            # 归一化时间
            time_features[t, 2] = t / self.n_periods
            # 是否为峰时段
            hour_int = int(hour) + 1
            time_features[t, 3] = 1.0 if hour_int in PRICE_CONFIG["peak_hours"] else 0.0

        return time_features

    def _get_static_features(self) -> np.ndarray:
        """获取网架静态特征"""
        static_features = []

        # ESS位置（one-hot编码，归一化到节点数）
        for bus in self.ess_config["buses"]:
            static_features.append(bus / self.n_buses)

        # ESS容量（归一化）
        max_capacity = max(self.ess_config["capacity_mwh"])
        for cap in self.ess_config["capacity_mwh"]:
            static_features.append(cap / max_capacity)

        # ESS充放电效率
        static_features.append(self.ess_config["efficiency_charge"])
        static_features.append(self.ess_config["efficiency_discharge"])

        # SOC限制
        static_features.append(self.ess_config["soc_min"])
        static_features.append(self.ess_config["soc_max"])

        # PV/WT数量（归一化）
        static_features.append(self.n_pv / 5.0)  # 假设最大5个
        static_features.append(self.n_wt / 5.0)

        # 联络线数量
        static_features.append(self.n_ties / 5.0)

        return np.array(static_features)

    def load_scenario_data(self) -> Dict[str, np.ndarray]:
        """加载场景输入数据"""
        scenario_file = self._get_scenario_file()

        if not scenario_file.exists():
            raise FileNotFoundError(f"场景文件不存在: {scenario_file}")

        df = pd.read_csv(scenario_file)

        # 提取PV出力
        pv_data = np.zeros((self.n_periods, self.n_pv))
        for i, col in enumerate(self.pv_config["columns"]):
            if col in df.columns:
                pv_data[:, i] = df[col].values[:self.n_periods]

        # 提取WT出力
        wt_data = np.zeros((self.n_periods, self.n_wt))
        for i, col in enumerate(self.wt_config["columns"]):
            if col in df.columns:
                wt_data[:, i] = df[col].values[:self.n_periods]

        # 电价序列
        price_data = self._get_price_profile()

        # 时间编码
        time_encoding = self._get_time_encoding()

        # 静态特征
        static_features = self._get_static_features()

        return {
            "pv": pv_data,  # (96, n_pv)
            "wt": wt_data,  # (96, n_wt)
            "price": price_data,  # (96,)
            "time_encoding": time_encoding,  # (96, 4)
            "static": static_features,  # (n_static,)
        }

    def load_optimization_results(self) -> Dict[str, np.ndarray]:
        """加载优化结果作为标签"""
        result_file = self._get_result_file()

        if not result_file.exists():
            raise FileNotFoundError(f"结果文件不存在: {result_file}")

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data["results"]

        # ESS充电功率 (96, n_ess)
        ess_charge = np.array(results["ess"]["charge_mw"])

        # ESS放电功率 (96, n_ess)
        ess_discharge = np.array(results["ess"]["discharge_mw"])

        # ESS模式 (96, n_ess) - 1=放电, 0=充电
        ess_mode = np.array(results["ess"]["mode"]) if results["ess"]["mode"] is not None else np.zeros_like(ess_charge)

        # 购电功率 (96,)
        p_grid = np.array(results["grid"]["power_mw"])

        # 联络线状态 (96, n_ties)
        if results.get("reconfiguration") and results["reconfiguration"].get("status"):
            tie_status = np.array(results["reconfiguration"]["status"])
        else:
            tie_status = np.zeros((self.n_periods, self.n_ties))

        return {
            "ess_charge": ess_charge,  # (96, n_ess)
            "ess_discharge": ess_discharge,  # (96, n_ess)
            "ess_mode": ess_mode,  # (96, n_ess)
            "p_grid": p_grid,  # (96,)
            "tie_status": tie_status,  # (96, n_ties)
        }

    def prepare_input_tensor(self, scenario_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将场景数据整理为模型输入张量

        输入特征结构 (每个时步):
        - PV出力: n_pv
        - WT出力: n_wt
        - 电价: 1
        - 时间编码: 4
        - 静态特征: n_static (广播到每个时步)

        Returns:
            input_tensor: shape (n_periods, n_features)
        """
        n_periods = self.n_periods

        # 时序特征
        pv = scenario_data["pv"]  # (96, n_pv)
        wt = scenario_data["wt"]  # (96, n_wt)
        price = scenario_data["price"].reshape(-1, 1)  # (96, 1)
        time_enc = scenario_data["time_encoding"]  # (96, 4)

        # 静态特征广播
        static = scenario_data["static"]  # (n_static,)
        static_broadcast = np.tile(static, (n_periods, 1))  # (96, n_static)

        # 拼接所有特征
        input_tensor = np.concatenate([
            pv, wt, price, time_enc, static_broadcast
        ], axis=1)

        return input_tensor.astype(np.float32)

    def prepare_output_tensor(self, opt_results: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将优化结果整理为模型输出张量

        Returns:
            continuous_output: 连续变量 (n_periods, n_continuous)
            binary_output: 二进制变量 (n_periods, n_binary)
        """
        # 连续变量
        ess_charge = opt_results["ess_charge"]  # (96, n_ess)
        ess_discharge = opt_results["ess_discharge"]  # (96, n_ess)
        p_grid = opt_results["p_grid"].reshape(-1, 1)  # (96, 1)

        continuous = np.concatenate([
            ess_charge, ess_discharge, p_grid
        ], axis=1).astype(np.float32)

        # 二进制变量
        ess_mode = opt_results["ess_mode"]  # (96, n_ess)
        tie_status = opt_results["tie_status"]  # (96, n_ties)

        binary = np.concatenate([
            ess_mode, tie_status
        ], axis=1).astype(np.float32)

        return continuous, binary

    def augment_data(self, input_data: np.ndarray,
                     n_augments: int = 50,
                     noise_std: float = 0.02,
                     scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        数据增强：对输入特征添加噪声和缩放

        Args:
            input_data: 原始输入 (n_periods, n_features)
            n_augments: 增强样本数量
            noise_std: 高斯噪声标准差
            scale_range: 缩放因子范围

        Returns:
            augmented_data: (n_augments + 1, n_periods, n_features)
        """
        augmented = [input_data]  # 原始数据

        for _ in range(n_augments):
            aug_data = input_data.copy()

            # 只对时序特征（PV、WT、电价）添加噪声，不改变静态特征
            n_time_features = self.n_pv + self.n_wt + 1 + 4  # PV + WT + price + time_encoding

            # 高斯噪声
            noise = np.random.normal(0, noise_std, (self.n_periods, n_time_features))
            aug_data[:, :n_time_features] += noise

            # 负荷缩放（仅对PV和WT）
            scale = np.random.uniform(scale_range[0], scale_range[1])
            aug_data[:, :self.n_pv + self.n_wt] *= scale

            # 确保非负
            aug_data[:, :self.n_pv + self.n_wt] = np.maximum(aug_data[:, :self.n_pv + self.n_wt], 0)

            augmented.append(aug_data)

        return np.array(augmented, dtype=np.float32)

    def normalize_data(self,
                       input_data: np.ndarray,
                       continuous_output: np.ndarray,
                       fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据标准化

        Args:
            input_data: 输入特征 (n_samples, n_periods, n_features) 或 (n_periods, n_features)
            continuous_output: 连续输出 (n_samples, n_periods, n_continuous) 或 (n_periods, n_continuous)
            fit: 是否拟合标准化器

        Returns:
            normalized_input, normalized_output
        """
        # 处理维度
        input_shape = input_data.shape
        output_shape = continuous_output.shape

        if len(input_shape) == 3:
            n_samples, n_periods, n_features = input_shape
            input_flat = input_data.reshape(-1, n_features)
            output_flat = continuous_output.reshape(-1, continuous_output.shape[-1])
        else:
            input_flat = input_data
            output_flat = continuous_output

        if fit:
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
            input_normalized = self.input_scaler.fit_transform(input_flat)
            output_normalized = self.output_scaler.fit_transform(output_flat)
        else:
            input_normalized = self.input_scaler.transform(input_flat)
            output_normalized = self.output_scaler.transform(output_flat)

        # 恢复形状
        if len(input_shape) == 3:
            input_normalized = input_normalized.reshape(input_shape)
            output_normalized = output_normalized.reshape(output_shape)

        return input_normalized.astype(np.float32), output_normalized.astype(np.float32)

    def denormalize_output(self, normalized_output: np.ndarray) -> np.ndarray:
        """反标准化输出"""
        if self.output_scaler is None:
            return normalized_output

        shape = normalized_output.shape
        if len(shape) == 3:
            flat = normalized_output.reshape(-1, shape[-1])
            denorm = self.output_scaler.inverse_transform(flat)
            return denorm.reshape(shape)
        else:
            return self.output_scaler.inverse_transform(normalized_output)

    def prepare_dataset(self,
                        n_augments: int = 50,
                        val_ratio: float = 0.2,
                        noise_std: float = 0.02,
                        scale_range: Tuple[float, float] = (0.95, 1.05)
                        ) -> Tuple[Dict, Dict, Dict]:
        """
        准备完整的训练和验证数据集

        Args:
            n_augments: 数据增强数量
            val_ratio: 验证集比例
            noise_std: 噪声标准差
            scale_range: 缩放范围

        Returns:
            train_data: 训练数据字典
            val_data: 验证数据字典
            meta: 元信息
        """
        # 加载原始数据
        scenario_data = self.load_scenario_data()
        opt_results = self.load_optimization_results()

        # 准备张量
        input_tensor = self.prepare_input_tensor(scenario_data)  # (96, n_features)
        continuous_output, binary_output = self.prepare_output_tensor(opt_results)

        # 保存原始数据
        self.raw_input = input_tensor
        self.raw_output = {"continuous": continuous_output, "binary": binary_output}

        # 数据增强
        augmented_input = self.augment_data(
            input_tensor, n_augments, noise_std, scale_range
        )  # (n_samples, 96, n_features)

        n_samples = augmented_input.shape[0]

        # 复制输出标签（增强数据使用相同标签）
        continuous_output_expanded = np.tile(
            continuous_output[np.newaxis, :, :], (n_samples, 1, 1)
        )
        binary_output_expanded = np.tile(
            binary_output[np.newaxis, :, :], (n_samples, 1, 1)
        )

        # 标准化
        input_normalized, continuous_normalized = self.normalize_data(
            augmented_input, continuous_output_expanded, fit=True
        )

        # 划分训练验证集
        n_val = int(n_samples * val_ratio)
        indices = np.random.permutation(n_samples)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_data = {
            "input": input_normalized[train_indices],
            "continuous": continuous_normalized[train_indices],
            "binary": binary_output_expanded[train_indices],
        }

        val_data = {
            "input": input_normalized[val_indices],
            "continuous": continuous_normalized[val_indices],
            "binary": binary_output_expanded[val_indices],
        }

        meta = {
            "n_features": input_tensor.shape[-1],
            "n_continuous": continuous_output.shape[-1],
            "n_binary": binary_output.shape[-1],
            "n_periods": self.n_periods,
            "n_ess": self.n_ess,
            "n_ties": self.n_ties,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
        }

        return train_data, val_data, meta


class EDDataset(Dataset):
    """PyTorch Dataset for ED task"""

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        """
        Args:
            data_dict: 包含 "input", "continuous", "binary" 的字典
        """
        self.input = torch.from_numpy(data_dict["input"])
        self.continuous = torch.from_numpy(data_dict["continuous"])
        self.binary = torch.from_numpy(data_dict["binary"])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "continuous": self.continuous[idx],
            "binary": self.binary[idx],
        }


def create_data_loaders(processor: EDDataProcessor,
                        batch_size: int = 16,
                        n_augments: int = 50,
                        val_ratio: float = 0.2,
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器

    Args:
        processor: 数据处理器
        batch_size: 批次大小
        n_augments: 增强数量
        val_ratio: 验证集比例
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, meta
    """
    train_data, val_data, meta = processor.prepare_dataset(
        n_augments=n_augments, val_ratio=val_ratio
    )

    train_dataset = EDDataset(train_data)
    val_dataset = EDDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, meta


if __name__ == "__main__":
    # 测试
    processor = EDDataProcessor("ieee33", "004")

    print("加载场景数据...")
    scenario_data = processor.load_scenario_data()
    print(f"  PV shape: {scenario_data['pv'].shape}")
    print(f"  WT shape: {scenario_data['wt'].shape}")
    print(f"  Price shape: {scenario_data['price'].shape}")

    print("\n加载优化结果...")
    opt_results = processor.load_optimization_results()
    print(f"  ESS charge shape: {opt_results['ess_charge'].shape}")
    print(f"  P_grid shape: {opt_results['p_grid'].shape}")

    print("\n准备数据集...")
    train_data, val_data, meta = processor.prepare_dataset(n_augments=50)
    print(f"  训练样本数: {meta['n_train']}")
    print(f"  验证样本数: {meta['n_val']}")
    print(f"  输入特征维度: {meta['n_features']}")
    print(f"  连续输出维度: {meta['n_continuous']}")
    print(f"  二进制输出维度: {meta['n_binary']}")