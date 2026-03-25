# -*- coding: utf-8 -*-
"""
DNN Baseline 预测器 - Task B (ED)

功能:
- 加载训练好的模型
- 对新输入进行预测（支持跨场景）
- 输出解码和后处理
- 结果保存
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler

from .dnn_model import EDTransformerModel
from .data_loader_ed import EDDataProcessor
from .trainer import EDTrainer


class EDPredictor:
    """ED任务DNN预测器"""

    def __init__(self,
                 network_name: str,
                 scenario_id: str,
                 model_path: Path = None,
                 model_scenario_id: str = None,
                 project_root: Path = None,
                 device: torch.device = None):
        """
        Args:
            network_name: 网络名称
            scenario_id: 测试场景编号（用于加载输入数据）
            model_path: 模型文件路径（如果为None则自动推断）
            model_scenario_id: 模型训练场景编号（用于跨场景推理）
            project_root: 项目根目录
            device: 计算设备
        """
        self.network_name = network_name.lower()
        self.scenario_id = scenario_id
        self.model_scenario_id = model_scenario_id or scenario_id

        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型路径（使用model_scenario_id确定）
        if model_path is None:
            model_path = project_root / "opt_results" / "dnn" / f"{network_name}_ed_scenario_{self.model_scenario_id}_model.pt"
        self.model_path = model_path

        # 数据处理器（使用scenario_id加载测试数据）
        self.processor = EDDataProcessor(network_name, scenario_id, project_root)

        # 模型
        self.model = None
        self.checkpoint = None

        # 标准化器（从模型checkpoint加载）
        self.input_scaler = None
        self.output_scaler = None

        # 推理时间统计
        self.inference_time = 0.0

        # 是否跨场景
        self.is_cross_scenario = self.model_scenario_id != self.scenario_id

    def load_model(self):
        """加载模型和标准化器"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model, self.checkpoint = EDTrainer.load_model(self.model_path, self.device)
        self.model.eval()

        # 从checkpoint加载标准化器参数
        meta = self.checkpoint.get("meta", {})

        if meta.get("input_scaler_mean") is not None:
            self.input_scaler = StandardScaler()
            self.input_scaler.mean_ = np.array(meta["input_scaler_mean"])
            self.input_scaler.scale_ = np.array(meta["input_scaler_scale"])
            self.input_scaler.var_ = self.input_scaler.scale_ ** 2
            self.input_scaler.n_features_in_ = len(self.input_scaler.mean_)

        if meta.get("output_scaler_mean") is not None:
            self.output_scaler = StandardScaler()
            self.output_scaler.mean_ = np.array(meta["output_scaler_mean"])
            self.output_scaler.scale_ = np.array(meta["output_scaler_scale"])
            self.output_scaler.var_ = self.output_scaler.scale_ ** 2
            self.output_scaler.n_features_in_ = len(self.output_scaler.mean_)

        if self.is_cross_scenario:
            print(f"模型已加载 (跨场景): {self.model_path}")
            print(f"  模型训练于场景: {self.model_scenario_id}")
            print(f"  测试场景: {self.scenario_id}")
        else:
            print(f"模型已加载: {self.model_path}")

    def prepare_input(self) -> torch.Tensor:
        """准备输入数据"""
        # 加载测试场景数据
        scenario_data = self.processor.load_scenario_data()

        # 准备输入张量
        input_tensor = self.processor.prepare_input_tensor(scenario_data)  # (96, n_features)

        # 使用模型训练时的标准化器
        if self.input_scaler is not None:
            input_normalized = self.input_scaler.transform(input_tensor)
        else:
            # 如果没有保存scaler，使用数据处理器重新拟合（不推荐）
            print("  警告: 使用测试数据重新拟合标准化器（可能影响跨场景性能）")
            _, _, _ = self.processor.prepare_dataset(n_augments=1)
            input_normalized = self.processor.input_scaler.transform(input_tensor)
            self.output_scaler = self.processor.output_scaler

        # 转换为tensor
        input_tensor = torch.from_numpy(input_normalized).float().unsqueeze(0)  # (1, 96, n_features)

        return input_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        执行预测

        Args:
            threshold: 二进制变量阈值

        Returns:
            预测结果字典
        """
        if self.model is None:
            self.load_model()

        # 准备输入
        input_tensor = self.prepare_input()

        # 推理
        start_time = time.time()

        self.model.eval()
        continuous_normalized, binary_logits = self.model(input_tensor)

        self.inference_time = time.time() - start_time

        # 反标准化连续输出
        continuous_normalized = continuous_normalized.cpu().numpy().squeeze(0)  # (96, n_continuous)

        if self.output_scaler is not None:
            continuous = self.output_scaler.inverse_transform(continuous_normalized)
        else:
            continuous = continuous_normalized

        # 二进制输出
        binary = (torch.sigmoid(binary_logits) > threshold).float().cpu().numpy().squeeze(0)  # (96, n_binary)

        # 解析输出
        n_ess = self.processor.n_ess
        n_ties = self.processor.n_ties

        results = {
            "ess_charge": continuous[:, :n_ess],
            "ess_discharge": continuous[:, n_ess:2 * n_ess],
            "p_grid": continuous[:, 2 * n_ess],
            "ess_mode": binary[:, :n_ess],
            "tie_status": binary[:, n_ess:n_ess + n_ties],
        }

        return results

    def decode_results(self, raw_results: Dict[str, np.ndarray]) -> Dict:
        """
        解码原始预测结果为完整格式

        Args:
            raw_results: 原始预测结果

        Returns:
            完整格式的结果字典
        """
        n_periods = self.processor.n_periods
        n_ess = self.processor.n_ess
        n_ties = self.processor.n_ties
        delta_t = self.processor.delta_t

        ess_charge = raw_results["ess_charge"].copy()
        ess_discharge = raw_results["ess_discharge"].copy()
        ess_mode = raw_results["ess_mode"]
        p_grid = raw_results["p_grid"]
        tie_status = raw_results["tie_status"]

        # 后处理：根据ess_mode修正充放电
        # mode=1时只放电，mode=0时只充电
        for t in range(n_periods):
            for k in range(n_ess):
                if ess_mode[t, k] == 1:  # 放电模式
                    ess_charge[t, k] = 0
                else:  # 充电模式
                    ess_discharge[t, k] = 0

        # 确保非负
        ess_charge = np.maximum(ess_charge, 0)
        ess_discharge = np.maximum(ess_discharge, 0)

        # 计算SOC
        ess_config = self.processor.ess_config
        capacity = ess_config["capacity_mwh"]
        eta_ch = ess_config["efficiency_charge"]
        eta_dis = ess_config["efficiency_discharge"]
        soc_init = ess_config["soc_init"]

        ess_soc = np.zeros((n_periods, n_ess))
        for k in range(n_ess):
            E_prev = soc_init * capacity[k]
            for t in range(n_periods):
                E_new = E_prev + eta_ch * ess_charge[t, k] * delta_t - ess_discharge[t, k] / eta_dis * delta_t
                ess_soc[t, k] = E_new
                E_prev = E_new

        # 构建完整结果
        results = {
            "ess": {
                "buses": self.processor.ess_config["buses"],
                "charge_mw": ess_charge.tolist(),
                "discharge_mw": ess_discharge.tolist(),
                "soc_mwh": ess_soc.tolist(),
                "mode": ess_mode.astype(int).tolist(),
            },
            "grid": {
                "power_mw": p_grid.tolist(),
                "total_purchase_mwh": float(np.sum(np.maximum(p_grid, 0)) * delta_t),
                "total_sell_mwh": float(-np.sum(np.minimum(p_grid, 0)) * delta_t),
            },
            "reconfiguration": {
                "status": tie_status.astype(int).tolist(),
                "power_mw": np.zeros((n_periods, n_ties)).tolist(),
                "changes": np.zeros((n_periods, n_ties), dtype=int).tolist(),
                "total_switches": 0,
            } if n_ties > 0 else None,
            "inference_time": self.inference_time,
            "model_scenario_id": self.model_scenario_id,
            "test_scenario_id": self.scenario_id,
            "is_cross_scenario": self.is_cross_scenario,
        }

        return results

    def save_results(self, results: Dict, output_path: Path = None):
        """保存预测结果"""
        if output_path is None:
            output_dir = self.project_root / "opt_results" / "dnn"
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.is_cross_scenario:
                # 跨场景结果文件名包含模型来源
                output_path = output_dir / f"{self.network_name}_ed_scenario_{self.scenario_id}_from_{self.model_scenario_id}_dnn_results.json"
            else:
                output_path = output_dir / f"{self.network_name}_ed_scenario_{self.scenario_id}_dnn_results.json"

        output_data = {
            "network": self.network_name,
            "scenario_id": self.scenario_id,
            "model_scenario_id": self.model_scenario_id,
            "is_cross_scenario": self.is_cross_scenario,
            "method": "dnn",
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"结果已保存至: {output_path}")
        return output_path

    def run(self, save: bool = True) -> Dict:
        """
        运行完整预测流程

        Args:
            save: 是否保存结果

        Returns:
            预测结果
        """
        print(f"\n{'=' * 60}")
        if self.is_cross_scenario:
            print(f"DNN跨场景预测 - {self.network_name.upper()}")
            print(f"  模型: 场景 {self.model_scenario_id}")
            print(f"  测试: 场景 {self.scenario_id}")
        else:
            print(f"DNN预测 - {self.network_name.upper()} - 场景 {self.scenario_id}")
        print(f"{'=' * 60}")

        # 加载模型
        self.load_model()

        # 预测
        print("\n执行预测...")
        raw_results = self.predict()

        # 解码
        print("解码结果...")
        results = self.decode_results(raw_results)

        print(f"推理时间: {self.inference_time * 1000:.2f} ms")

        # 保存
        if save:
            self.save_results(results)

        return results


if __name__ == "__main__":
    # 测试同场景
    print("=" * 70)
    print("测试同场景预测")
    print("=" * 70)
    predictor = EDPredictor("ieee33", "004")

    try:
        results = predictor.run(save=True)
        print("\n预测完成!")
        print(f"  购电总量: {results['grid']['total_purchase_mwh']:.2f} MWh")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练模型!")

    # 测试跨场景
    print("\n" + "=" * 70)
    print("测试跨场景预测 (004模型 -> 005数据)")
    print("=" * 70)
    predictor_cross = EDPredictor("ieee33", "005", model_scenario_id="004")

    try:
        results_cross = predictor_cross.run(save=True)
        print("\n跨场景预测完成!")
        print(f"  购电总量: {results_cross['grid']['total_purchase_mwh']:.2f} MWh")
    except FileNotFoundError as e:
        print(f"错误: {e}")