# -*- coding: utf-8 -*-
"""
约束违反度计算模块

功能:
- 计算DNN预测结果的各项约束违反度
- 支持ESS约束、电压约束等
- 输出详细的约束违反统计
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config_networks import get_network_config


class ConstraintViolationChecker:
    """约束违反度检查器"""

    def __init__(self, network_name: str, scenario_id: str, project_root: Path = None):
        """
        Args:
            network_name: 网络名称
            scenario_id: 场景编号
            project_root: 项目根目录
        """
        self.network_name = network_name.lower()
        self.scenario_id = scenario_id

        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = project_root

        # 加载配置
        self.config = get_network_config(network_name, project_root)

        # ESS配置
        self.ess_config = self.config["devices"]["ess"]
        self.n_ess = len(self.ess_config["buses"])
        self.ess_capacity = self.ess_config["capacity_mwh"]
        self.soc_min = self.ess_config["soc_min"]
        self.soc_max = self.ess_config["soc_max"]
        self.max_charge_rate = self.ess_config["max_charge_rate"]
        self.max_discharge_rate = self.ess_config["max_discharge_rate"]
        self.eta_ch = self.ess_config["efficiency_charge"]
        self.eta_dis = self.ess_config["efficiency_discharge"]
        self.soc_init = self.ess_config["soc_init"]

        # 电压配置
        self.v_min = self.config["network"]["v_min"]
        self.v_max = self.config["network"]["v_max"]

        # 时间配置
        self.n_periods = self.config["optimization"]["ed"]["n_periods"]
        self.delta_t = self.config["optimization"]["ed"]["delta_t"]

    def load_dnn_results(self, results_path: Path = None) -> Dict:
        """加载DNN预测结果"""
        if results_path is None:
            results_path = self.project_root / "opt_results" / "dnn" / \
                           f"{self.network_name}_ed_scenario_{self.scenario_id}_dnn_results.json"

        if not results_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {results_path}")

        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data["results"]

    def load_ground_truth(self, gt_path: Path = None) -> Dict:
        """加载优化器求解的真实结果（ground truth）"""
        if gt_path is None:
            gt_path = self.project_root / "opt_results" / "ed" / \
                      f"{self.network_name}_ed_scenario_{self.scenario_id}_results.json"

        if not gt_path.exists():
            raise FileNotFoundError(f"真实结果文件不存在: {gt_path}")

        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data["results"]

    def check_ess_soc_bounds(self, results: Dict) -> Dict:
        """
        检查ESS SOC边界约束

        约束: SOC_min * capacity <= SOC <= SOC_max * capacity
        """
        ess_soc = np.array(results["ess"]["soc_mwh"])  # (96, n_ess)

        violations = {
            "min_violations": [],
            "max_violations": [],
            "total_violation_count": 0,
            "total_check_count": 0,
            "total_violation_percentage": 0.0,
        }

        total_violations = 0
        total_checks = ess_soc.size

        for k in range(self.n_ess):
            soc_min_k = self.soc_min * self.ess_capacity[k]
            soc_max_k = self.soc_max * self.ess_capacity[k]

            # 下界违反
            min_viol = np.maximum(soc_min_k - ess_soc[:, k], 0)
            min_viol_count = np.sum(min_viol > 1e-6)
            total_violations += min_viol_count

            # 上界违反
            max_viol = np.maximum(ess_soc[:, k] - soc_max_k, 0)
            max_viol_count = np.sum(max_viol > 1e-6)
            total_violations += max_viol_count

            violations["min_violations"].append({
                "ess_id": k,
                "violation_count": int(min_viol_count),
                "max_violation_mwh": float(np.max(min_viol)),
                "avg_violation_mwh": float(np.mean(min_viol[min_viol > 1e-6])) if min_viol_count > 0 else 0.0,
            })

            violations["max_violations"].append({
                "ess_id": k,
                "violation_count": int(max_viol_count),
                "max_violation_mwh": float(np.max(max_viol)),
                "avg_violation_mwh": float(np.mean(max_viol[max_viol > 1e-6])) if max_viol_count > 0 else 0.0,
            })

        violations["total_violation_count"] = int(total_violations)
        violations["total_check_count"] = int(total_checks)
        violations["total_violation_percentage"] = 100.0 * total_violations / total_checks

        return violations

    def check_ess_power_bounds(self, results: Dict) -> Dict:
        """
        检查ESS充放电功率边界约束

        约束:
        - 0 <= P_ch <= max_charge_rate * capacity
        - 0 <= P_dis <= max_discharge_rate * capacity
        """
        ess_charge = np.array(results["ess"]["charge_mw"])
        ess_discharge = np.array(results["ess"]["discharge_mw"])

        violations = {
            "charge_violations": [],
            "discharge_violations": [],
            "negative_violations": 0,
            "total_violation_count": 0,
            "total_check_count": 0,
            "total_violation_percentage": 0.0,
        }

        total_violations = 0
        total_checks = ess_charge.size + ess_discharge.size

        for k in range(self.n_ess):
            max_ch_k = self.max_charge_rate * self.ess_capacity[k]
            max_dis_k = self.max_discharge_rate * self.ess_capacity[k]

            # 充电上界违反
            ch_viol = np.maximum(ess_charge[:, k] - max_ch_k, 0)
            ch_viol_count = np.sum(ch_viol > 1e-6)
            total_violations += ch_viol_count

            # 放电上界违反
            dis_viol = np.maximum(ess_discharge[:, k] - max_dis_k, 0)
            dis_viol_count = np.sum(dis_viol > 1e-6)
            total_violations += dis_viol_count

            violations["charge_violations"].append({
                "ess_id": k,
                "violation_count": int(ch_viol_count),
                "max_violation_mw": float(np.max(ch_viol)),
            })

            violations["discharge_violations"].append({
                "ess_id": k,
                "violation_count": int(dis_viol_count),
                "max_violation_mw": float(np.max(dis_viol)),
            })

        # 负值违反
        neg_ch = np.sum(ess_charge < -1e-6)
        neg_dis = np.sum(ess_discharge < -1e-6)
        violations["negative_violations"] = int(neg_ch + neg_dis)
        total_violations += neg_ch + neg_dis

        violations["total_violation_count"] = int(total_violations)
        violations["total_check_count"] = int(total_checks)
        violations["total_violation_percentage"] = 100.0 * total_violations / total_checks

        return violations

    def check_ess_mutex(self, results: Dict) -> Dict:
        """
        检查ESS充放电互斥约束

        约束: 同一时刻不能同时充电和放电
        """
        ess_charge = np.array(results["ess"]["charge_mw"])
        ess_discharge = np.array(results["ess"]["discharge_mw"])

        # 检查同时充放电
        simultaneous = (ess_charge > 1e-6) & (ess_discharge > 1e-6)
        violation_count = np.sum(simultaneous)

        violations = {
            "violation_count": int(violation_count),
            "total_check_count": int(simultaneous.size),
            "violation_percentage": 100.0 * violation_count / simultaneous.size,
            "details": [],
        }

        for k in range(self.n_ess):
            k_violations = np.sum(simultaneous[:, k])
            if k_violations > 0:
                periods = np.where(simultaneous[:, k])[0].tolist()
                violations["details"].append({
                    "ess_id": k,
                    "violation_count": int(k_violations),
                    "violation_periods": periods[:10],  # 只显示前10个
                })

        return violations

    def check_ess_soc_dynamics(self, results: Dict) -> Dict:
        """
        检查ESS SOC动态约束

        约束: SOC[t] = SOC[t-1] + eta_ch * P_ch * dt - P_dis / eta_dis * dt
        """
        ess_charge = np.array(results["ess"]["charge_mw"])
        ess_discharge = np.array(results["ess"]["discharge_mw"])
        ess_soc = np.array(results["ess"]["soc_mwh"])

        violations = {
            "total_violation_count": 0,
            "max_deviation_mwh": 0.0,
            "avg_deviation_mwh": 0.0,
            "details": [],
        }

        all_deviations = []

        for k in range(self.n_ess):
            soc_expected = np.zeros(self.n_periods)
            E_prev = self.soc_init * self.ess_capacity[k]

            for t in range(self.n_periods):
                soc_expected[t] = E_prev + self.eta_ch * ess_charge[t, k] * self.delta_t \
                                  - ess_discharge[t, k] / self.eta_dis * self.delta_t
                E_prev = soc_expected[t]

            # 计算偏差
            deviation = np.abs(ess_soc[:, k] - soc_expected)
            violation_count = np.sum(deviation > 0.01)  # 允许1%误差

            all_deviations.extend(deviation.tolist())

            violations["details"].append({
                "ess_id": k,
                "violation_count": int(violation_count),
                "max_deviation_mwh": float(np.max(deviation)),
                "avg_deviation_mwh": float(np.mean(deviation)),
            })

        violations["total_violation_count"] = sum(d["violation_count"] for d in violations["details"])
        violations["max_deviation_mwh"] = float(np.max(all_deviations))
        violations["avg_deviation_mwh"] = float(np.mean(all_deviations))

        return violations

    def check_voltage_bounds(self, voltage_data: np.ndarray = None) -> Dict:
        """
        检查电压边界约束

        约束: V_min <= V <= V_max

        注意: 需要提供电压数据（从潮流计算获得）
        """
        if voltage_data is None:
            return {
                "status": "skipped",
                "reason": "需要运行潮流计算获取电压数据",
            }

        voltage = np.array(voltage_data)

        # 下界违反
        min_viol = np.maximum(self.v_min - voltage, 0)
        min_viol_count = np.sum(min_viol > 1e-6)

        # 上界违反
        max_viol = np.maximum(voltage - self.v_max, 0)
        max_viol_count = np.sum(max_viol > 1e-6)

        violations = {
            "min_violation_count": int(min_viol_count),
            "max_violation_count": int(max_viol_count),
            "total_violation_count": int(min_viol_count + max_viol_count),
            "total_violation_percentage": 100.0 * (min_viol_count + max_viol_count) / voltage.size,
            "voltage_min": float(np.min(voltage)),
            "voltage_max": float(np.max(voltage)),
            "voltage_mean": float(np.mean(voltage)),
            "max_undervoltage_pu": float(np.max(min_viol)),
            "max_overvoltage_pu": float(np.max(max_viol)),
        }

        return violations

    def run_power_flow_check(self, results: Dict) -> Dict:
        """
        运行潮流计算并检查电压约束

        使用简化的潮流计算（实际应用中可调用详细潮流模块）
        """
        try:
            # 尝试导入潮流计算模块
            from models.power_flow.distflow import DistFlowSolver
            from config_networks import get_network_instance

            # 获取网络实例
            network = get_network_instance(self.network_name, self.config)

            # 加载场景数据
            from data.data_loader import get_data_loader
            loader = get_data_loader(self.config)
            scenario_file = f"scenario_{self.scenario_id}_{self.config['network']['n_buses']}.csv"
            scenario_data = loader.get_scenario_data(filename=scenario_file, n_periods=self.n_periods)

            # 提取ESS功率
            ess_charge = np.array(results["ess"]["charge_mw"])
            ess_discharge = np.array(results["ess"]["discharge_mw"])
            ess_net_power = ess_discharge - ess_charge  # 净注入功率

            # 运行每个时段的潮流
            all_voltages = []

            for t in range(self.n_periods):
                # 构建注入功率
                p_inject = np.zeros(network.n_buses)
                q_inject = np.zeros(network.n_buses)

                # PV注入
                for i, bus in enumerate(self.config["devices"]["pv"]["buses"]):
                    p_inject[bus] = scenario_data["pv"][bus][t]

                # WT注入
                for i, bus in enumerate(self.config["devices"]["wt"]["buses"]):
                    p_inject[bus] = scenario_data["wt"][bus][t]

                # ESS注入
                for i, bus in enumerate(self.ess_config["buses"]):
                    p_inject[bus] += ess_net_power[t, i]

                # 负荷
                load_factor = scenario_data.get("load_factor", np.ones(self.n_periods))[t]
                p_load = network.p_load_mw * load_factor
                q_load = network.q_load_mvar * load_factor

                # 净注入
                p_net = p_inject - p_load
                q_net = q_inject - q_load

                # 简化潮流计算（使用DistFlow近似）
                solver = DistFlowSolver(network)
                voltage, _ = solver.solve(p_net, q_net)

                all_voltages.append(voltage)

            voltage_data = np.array(all_voltages)  # (96, n_buses)

            return self.check_voltage_bounds(voltage_data)

        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
            }

    def calculate_cost(self, results: Dict) -> Dict:
        """计算DNN方案的总成本"""
        from config_networks import PRICE_CONFIG

        p_grid = np.array(results["grid"]["power_mw"])
        ess_charge = np.array(results["ess"]["charge_mw"])
        ess_discharge = np.array(results["ess"]["discharge_mw"])

        # 生成电价序列
        price = np.zeros(self.n_periods)
        peak_hours = set(PRICE_CONFIG["peak_hours"])
        valley_hours = set(PRICE_CONFIG["valley_hours"])

        for t in range(self.n_periods):
            hour = (t * 15 // 60) + 1
            if hour > 24:
                hour = hour - 24

            if hour in peak_hours:
                price[t] = PRICE_CONFIG["peak_price"]
            elif hour in valley_hours:
                price[t] = PRICE_CONFIG["valley_price"]
            else:
                price[t] = PRICE_CONFIG["flat_price"]

        # 购电成本
        grid_cost = np.sum(price * p_grid * self.delta_t)

        # 储能成本
        ess_cost_rate = self.ess_config["cost_per_mwh"]
        ess_cost = np.sum(ess_charge + ess_discharge) * ess_cost_rate * self.delta_t

        # 总成本（不含网损，因为需要潮流计算）
        total_cost = grid_cost + ess_cost

        return {
            "grid_cost_yuan": float(grid_cost),
            "ess_cost_yuan": float(ess_cost),
            "total_cost_yuan": float(total_cost),
            "note": "不含网损成本（需潮流计算）",
        }

    def check_all(self, results: Dict = None,
                  include_power_flow: bool = True) -> Dict:
        """
        运行所有约束检查

        Args:
            results: DNN预测结果（如果为None则自动加载）
            include_power_flow: 是否包含潮流计算检查电压

        Returns:
            所有约束违反度的汇总
        """
        if results is None:
            results = self.load_dnn_results()

        # 各项检查
        soc_bounds = self.check_ess_soc_bounds(results)
        power_bounds = self.check_ess_power_bounds(results)
        mutex = self.check_ess_mutex(results)
        soc_dynamics = self.check_ess_soc_dynamics(results)

        # 电压约束（可选）
        if include_power_flow:
            voltage = self.run_power_flow_check(results)
        else:
            voltage = {"status": "skipped", "reason": "用户跳过"}

        # 成本计算
        cost = self.calculate_cost(results)

        # ============ 计算总违反比例 ============
        total_violation_count = 0
        total_check_count = 0

        # SOC边界
        total_violation_count += soc_bounds["total_violation_count"]
        total_check_count += soc_bounds["total_check_count"]

        # 功率边界
        total_violation_count += power_bounds["total_violation_count"]
        total_check_count += power_bounds["total_check_count"]

        # 充放电互斥
        total_violation_count += mutex["violation_count"]
        total_check_count += mutex["total_check_count"]

        # 计算总违反百分比
        total_violation_percentage = 100.0 * total_violation_count / total_check_count if total_check_count > 0 else 0.0

        # 汇总
        summary = {
            "network": self.network_name,
            "scenario_id": self.scenario_id,
            "constraints": {
                "ess_soc_bounds": soc_bounds,
                "ess_power_bounds": power_bounds,
                "ess_mutex": mutex,
                "ess_soc_dynamics": soc_dynamics,
                "voltage_bounds": voltage,
            },
            "cost": cost,
            "summary": {
                "total_violation_count": total_violation_count,
                "total_check_count": total_check_count,
                "total_violation_percentage": total_violation_percentage,
                "avg_ess_violations": (
                                              soc_bounds["total_violation_percentage"] +
                                              power_bounds["total_violation_percentage"] +
                                              mutex["violation_percentage"]
                                      ) / 3,
                "soc_dynamics_max_deviation": soc_dynamics["max_deviation_mwh"],
            },
        }

        return summary

    def compare_with_ground_truth(self, dnn_results: Dict = None,
                                  gt_results: Dict = None) -> Dict:
        """
        与优化器结果对比

        Args:
            dnn_results: DNN预测结果
            gt_results: 优化器真实结果

        Returns:
            对比结果
        """
        if dnn_results is None:
            dnn_results = self.load_dnn_results()
        if gt_results is None:
            gt_results = self.load_ground_truth()

        # 成本对比
        dnn_cost = self.calculate_cost(dnn_results)
        gt_cost = gt_results["cost"]["total_yuan"]

        cost_error = abs(dnn_cost["total_cost_yuan"] - gt_cost) / gt_cost * 100

        # ESS调度对比
        dnn_charge = np.array(dnn_results["ess"]["charge_mw"])
        dnn_discharge = np.array(dnn_results["ess"]["discharge_mw"])
        gt_charge = np.array(gt_results["ess"]["charge_mw"])
        gt_discharge = np.array(gt_results["ess"]["discharge_mw"])

        charge_rmse = np.sqrt(np.mean((dnn_charge - gt_charge) ** 2))
        discharge_rmse = np.sqrt(np.mean((dnn_discharge - gt_discharge) ** 2))

        # 购电功率对比
        dnn_pgrid = np.array(dnn_results["grid"]["power_mw"])
        gt_pgrid = np.array(gt_results["grid"]["power_mw"])
        pgrid_rmse = np.sqrt(np.mean((dnn_pgrid - gt_pgrid) ** 2))

        return {
            "cost_comparison": {
                "dnn_cost_yuan": dnn_cost["total_cost_yuan"],
                "gt_cost_yuan": gt_cost,
                "relative_error_percent": cost_error,
            },
            "ess_comparison": {
                "charge_rmse_mw": float(charge_rmse),
                "discharge_rmse_mw": float(discharge_rmse),
            },
            "grid_comparison": {
                "pgrid_rmse_mw": float(pgrid_rmse),
            },
        }


def print_violation_report(violations: Dict):
    """打印约束违反度报告"""
    print("\n" + "=" * 70)
    print("约束违反度检查报告")
    print("=" * 70)

    print(f"\n网络: {violations['network']}")
    print(f"场景: {violations['scenario_id']}")

    constraints = violations["constraints"]

    # ESS SOC边界
    print("\n[1] ESS SOC边界约束")
    soc = constraints["ess_soc_bounds"]
    print(f"    违反率: {soc['total_violation_percentage']:.2f}%")
    for v in soc["min_violations"]:
        if v["violation_count"] > 0:
            print(f"    ESS{v['ess_id']}: 下界违反 {v['violation_count']} 次, 最大 {v['max_violation_mwh']:.4f} MWh")
    for v in soc["max_violations"]:
        if v["violation_count"] > 0:
            print(f"    ESS{v['ess_id']}: 上界违反 {v['violation_count']} 次, 最大 {v['max_violation_mwh']:.4f} MWh")

    # ESS功率边界
    print("\n[2] ESS功率边界约束")
    power = constraints["ess_power_bounds"]
    print(f"    违反率: {power['total_violation_percentage']:.2f}%")
    print(f"    负值违反: {power['negative_violations']} 次")

    # ESS互斥
    print("\n[3] ESS充放电互斥约束")
    mutex = constraints["ess_mutex"]
    print(f"    违反率: {mutex['violation_percentage']:.2f}%")
    print(f"    违反次数: {mutex['violation_count']}")

    # SOC动态
    print("\n[4] ESS SOC动态约束")
    dynamics = constraints["ess_soc_dynamics"]
    print(f"    最大偏差: {dynamics['max_deviation_mwh']:.4f} MWh")
    print(f"    平均偏差: {dynamics['avg_deviation_mwh']:.4f} MWh")

    # 电压约束
    print("\n[5] 电压约束")
    voltage = constraints["voltage_bounds"]
    if "status" in voltage:
        print(f"    状态: {voltage['status']}")
        print(f"    原因: {voltage.get('reason', 'N/A')}")
    else:
        print(f"    违反率: {voltage['total_violation_percentage']:.2f}%")
        print(f"    电压范围: [{voltage['voltage_min']:.4f}, {voltage['voltage_max']:.4f}] pu")

    # 成本
    print("\n[6] 成本计算")
    cost = violations["cost"]
    print(f"    购电成本: {cost['grid_cost_yuan']:.2f} 元")
    print(f"    储能成本: {cost['ess_cost_yuan']:.2f} 元")
    print(f"    总成本: {cost['total_cost_yuan']:.2f} 元")

    # ============ 总违反比例 ============
    summary = violations["summary"]
    print("\n" + "-" * 70)
    print("【总体统计】")
    print(f"    总违反点数: {summary['total_violation_count']} / {summary['total_check_count']}")
    print(f"    总违反比例: {summary['total_violation_percentage']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    # 测试
    checker = ConstraintViolationChecker("ieee33", "004")

    try:
        violations = checker.check_all(include_power_flow=False)
        print_violation_report(violations)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行DNN预测生成结果文件!")