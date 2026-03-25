# -*- coding: utf-8 -*-
"""
评估指标模块
计算和汇总优化结果的各种评估指标
"""

import numpy as np
from typing import Dict, Any, List


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, v_min: float = 0.95, v_max: float = 1.05):
        """
        初始化
        
        Args:
            v_min: 电压下限
            v_max: 电压上限
        """
        self.v_min = v_min
        self.v_max = v_max
    
    def calculate_all_metrics(self, results: Dict, statistics: Dict) -> Dict:
        """
        计算所有评估指标
        
        Args:
            results: 优化结果
            statistics: 求解统计信息
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 时间指标
        metrics["time"] = self._calculate_time_metrics(statistics)
        
        # 电压指标
        metrics["voltage"] = self._calculate_voltage_metrics(results)
        
        # 网损指标
        metrics["loss"] = self._calculate_loss_metrics(results)
        
        # 设备动作指标
        metrics["device_actions"] = self._calculate_action_metrics(results)
        
        # 模型规模指标
        metrics["model_scale"] = self._calculate_scale_metrics(statistics)
        
        return metrics
    
    def _calculate_time_metrics(self, statistics: Dict) -> Dict:
        """计算时间相关指标"""
        return {
            "build_time_s": statistics.get("build_time", 0),
            "solve_time_s": statistics.get("solve_time", 0),
            "total_time_s": statistics.get("total_time", 0),
        }
    
    def _calculate_voltage_metrics(self, results: Dict) -> Dict:
        """计算电压相关指标"""
        voltage = results["voltage"]["values"]
        
        # 电压越限统计
        v_min_violation = np.maximum(self.v_min - voltage, 0)
        v_max_violation = np.maximum(voltage - self.v_max, 0)
        total_violation = v_min_violation + v_max_violation
        
        # 电压偏差（相对于1.0pu）
        voltage_deviation = np.abs(voltage - 1.0)
        
        return {
            "min_voltage_pu": float(voltage.min()),
            "max_voltage_pu": float(voltage.max()),
            "mean_voltage_pu": float(voltage.mean()),
            "std_voltage_pu": float(voltage.std()),
            
            # 越限指标
            "n_undervoltage_points": int(np.sum(voltage < self.v_min)),
            "n_overvoltage_points": int(np.sum(voltage > self.v_max)),
            "total_violation_pu": float(total_violation.sum()),
            "max_violation_pu": float(total_violation.max()),
            
            # 偏差指标
            "mean_deviation_pu": float(voltage_deviation.mean()),
            "max_deviation_pu": float(voltage_deviation.max()),
            
            # 是否所有电压都在安全范围内
            "all_voltages_safe": bool(np.all((voltage >= self.v_min) & (voltage <= self.v_max))),
        }
    
    def _calculate_loss_metrics(self, results: Dict) -> Dict:
        """计算网损相关指标"""
        loss_per_period = results["loss"]["per_period_kw"]
        
        return {
            "total_loss_kwh": float(results["loss"]["total_kw"] * 0.25),  # 15分钟转小时
            "average_loss_kw": float(results["loss"]["average_kw"]),
            "max_loss_kw": float(loss_per_period.max()),
            "min_loss_kw": float(loss_per_period.min()),
            "std_loss_kw": float(loss_per_period.std()),
        }
    
    def _calculate_action_metrics(self, results: Dict) -> Dict:
        """计算设备动作相关指标"""
        metrics = {}
        
        # 第一阶段指标 (OLTC, SC)
        if "oltc" in results:
            metrics["oltc_actions"] = results["oltc"]["n_actions"]
        
        if "sc" in results:
            metrics["sc_actions"] = results["sc"]["n_actions"]
            metrics["total_sc_actions"] = sum(results["sc"]["n_actions"])
        
        # 第二阶段指标 (PV, WT, SVC)
        if "pv_reactive" in results:
            pv_q = np.array(results["pv_reactive"]["q_mvar"])
            metrics["pv_q_mean"] = float(pv_q.mean())
            metrics["pv_q_range"] = [float(pv_q.min()), float(pv_q.max())]

        if "wt_reactive" in results:
            wt_q = np.array(results["wt_reactive"]["q_mvar"])
            if wt_q.size > 0:  # ✓ 添加空数组检查
                metrics["wt_q_mean"] = float(wt_q.mean())
                metrics["wt_q_range"] = [float(wt_q.min()), float(wt_q.max())]
            else:
                metrics["wt_q_mean"] = 0.0
                metrics["wt_q_range"] = [0.0, 0.0]
        
        if "svc_reactive" in results:
            svc_q = np.array(results["svc_reactive"]["q_mvar"])
            metrics["svc_q_mean"] = float(svc_q.mean())
            metrics["svc_q_range"] = [float(svc_q.min()), float(svc_q.max())]
        
        return metrics
    
    def _calculate_scale_metrics(self, statistics: Dict) -> Dict:
        """计算模型规模指标"""
        return {
            "n_variables": statistics.get("n_variables", 0),
            "n_constraints": statistics.get("n_constraints", 0),
            "n_binary_vars": statistics.get("n_binary_vars", 0),
        }
    
    def print_metrics_report(self, metrics: Dict):
        """打印评估指标报告"""
        print("\n" + "=" * 60)
        print("优化结果评估报告")
        print("=" * 60)
        
        # 时间指标
        print("\n【时间指标】")
        print(f"  模型构建时间: {metrics['time']['build_time_s']:.3f} 秒")
        print(f"  求解时间: {metrics['time']['solve_time_s']:.3f} 秒")
        print(f"  总运行时间: {metrics['time']['total_time_s']:.3f} 秒")
        
        # 电压指标
        print("\n【电压指标】")
        print(f"  最小电压: {metrics['voltage']['min_voltage_pu']:.4f} pu")
        print(f"  最大电压: {metrics['voltage']['max_voltage_pu']:.4f} pu")
        print(f"  平均电压: {metrics['voltage']['mean_voltage_pu']:.4f} pu")
        print(f"  电压标准差: {metrics['voltage']['std_voltage_pu']:.4f} pu")
        print(f"  欠压点数: {metrics['voltage']['n_undervoltage_points']}")
        print(f"  过压点数: {metrics['voltage']['n_overvoltage_points']}")
        print(f"  电压安全: {'是' if metrics['voltage']['all_voltages_safe'] else '否'}")
        
        # 网损指标
        print("\n【网损指标】")
        print(f"  总网损: {metrics['loss']['total_loss_kwh']:.2f} kWh")
        print(f"  平均网损: {metrics['loss']['average_loss_kw']:.2f} kW")
        print(f"  最大网损: {metrics['loss']['max_loss_kw']:.2f} kW")
        print(f"  最小网损: {metrics['loss']['min_loss_kw']:.2f} kW")
        
        # 设备动作指标
        print("\n【设备动作/无功输出】")
        device_actions = metrics['device_actions']
        
        # 第一阶段指标
        if 'oltc_actions' in device_actions:
            print(f"  OLTC动作次数: {device_actions['oltc_actions']}")
        if 'sc_actions' in device_actions:
            print(f"  SC动作次数: {device_actions['sc_actions']}")
            print(f"  SC总动作次数: {device_actions['total_sc_actions']}")
        
        # 第二阶段指标
        if 'pv_q_mean' in device_actions:
            print(f"  PV平均无功: {device_actions['pv_q_mean']:.4f} MVar")
            print(f"  PV无功范围: [{device_actions['pv_q_range'][0]:.4f}, {device_actions['pv_q_range'][1]:.4f}] MVar")
        if 'wt_q_mean' in device_actions:
            print(f"  WT平均无功: {device_actions['wt_q_mean']:.4f} MVar")
            print(f"  WT无功范围: [{device_actions['wt_q_range'][0]:.4f}, {device_actions['wt_q_range'][1]:.4f}] MVar")
        if 'svc_q_mean' in device_actions:
            print(f"  SVC平均无功: {device_actions['svc_q_mean']:.4f} MVar")
            print(f"  SVC无功范围: [{device_actions['svc_q_range'][0]:.4f}, {device_actions['svc_q_range'][1]:.4f}] MVar")
        
        # 模型规模
        print("\n【模型规模】")
        print(f"  变量数: {metrics['model_scale']['n_variables']}")
        print(f"  约束数: {metrics['model_scale']['n_constraints']}")
        print(f"  二元变量数: {metrics['model_scale']['n_binary_vars']}")
        
        print("\n" + "=" * 60)
    
    def to_dataframe(self, metrics: Dict) -> 'pd.DataFrame':
        """将指标转换为DataFrame格式"""
        import pandas as pd
        
        # 扁平化指标字典
        flat_metrics = {}
        for category, category_metrics in metrics.items():
            for key, value in category_metrics.items():
                flat_metrics[f"{category}_{key}"] = value
        
        return pd.DataFrame([flat_metrics])


def get_metrics_calculator(v_min: float = 0.95, v_max: float = 1.05) -> MetricsCalculator:
    """获取指标计算器实例"""
    return MetricsCalculator(v_min, v_max)


def calculate_metrics(results: Dict, statistics: Dict, 
                      v_min: float = 0.95, v_max: float = 1.05) -> Dict:
    """便捷函数：计算所有指标"""
    calculator = get_metrics_calculator(v_min, v_max)
    return calculator.calculate_all_metrics(results, statistics)
