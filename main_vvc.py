# -*- coding: utf-8 -*-
"""
配电网多时间尺度无功优化 - 主程序 (Task A: VVC)

支持在不同IEEE测试系统上运行：
  - IEEE 13节点
  - IEEE 33节点
  - IEEE 69节点（预留）
  - IEEE 123节点（预留）

第一阶段: 日前优化 (MISOCP)
  - 优化OLTC档位和SC投切状态
  - 时间分辨率: 15分钟
  - 求解器: Gurobi

第二阶段: 实时优化 (SOCP)
  - 优化PV/WT逆变器无功和SVC
  - 时间分辨率: 15分钟 (与第一阶段对齐)
  - 接收第一阶段结果作为输入

使用方法：
    python main_vvc.py --network ieee13
    python main_vvc.py --network ieee33
"""

import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# 项目根目录添加到路径
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_networks import get_network_config, get_network_instance, get_scenario_data_file, get_result_filename
from data.data_loader import get_data_loader
from models.day_ahead.misocp_var_opt import create_day_ahead_model
from models.real_time.socp_var_opt import create_real_time_model
from utils.visualization import get_visualizer
from utils.metrics import get_metrics_calculator


def run_day_ahead_optimization(config: dict, network_name: str,
                               data_file: str = None,
                               scenario_id: int = None,
                               verbose: bool = True) -> dict:
    """
    运行日前优化（第一阶段）

    Args:
        config: 配置字典
        network_name: 网络名称 (如 "ieee13", "ieee33")
        data_file: 数据文件名（可选，默认使用配置中的文件）
        scenario_id: 场景ID（可选，如果文件包含多个场景）
        verbose: 是否显示详细信息

    Returns:
        包含结果和统计信息的字典
    """
    print("\n" + "=" * 60)
    print(f"第一阶段: 日前无功优化 (MISOCP) - {network_name.upper()}")
    print("=" * 60)

    total_start_time = time.time()

    # ====== 1. 加载网络和数据 ======
    print("\n[1/4] 加载网络和数据...")
    data_start = time.time()

    network = get_network_instance(network_name, config)
    if verbose:
        network.summary()

    loader = get_data_loader(config)
    n_periods = config["optimization"]["day_ahead"]["n_periods"]

    # 使用指定的数据文件或配置中的默认文件
    if data_file is None:
        data_file = config["data"]["dg_data_file"]

    scenario = loader.get_scenario_data(filename=data_file,
                                        scenario_id=scenario_id,
                                        n_periods=n_periods)

    data_time = time.time() - data_start
    print(f"  数据加载完成，耗时: {data_time:.3f}秒")
    print(f"  调度周期: {n_periods}个时段 (15分钟/时段)")

    # ====== 2. 构建优化模型 ======
    print("\n[2/4] 构建优化模型...")

    model = create_day_ahead_model(config)
    model.build_model(network, scenario)

    print(f"  变量数: {model.statistics['n_variables']}")
    print(f"  约束数: {model.statistics['n_constraints']}")
    print(f"  二元变量数: {model.statistics['n_binary_vars']}")

    # ====== 3. 求解模型 ======
    print("\n[3/4] 求解优化模型...")

    solver_name = config["optimization"]["solver"]["name"]
    solve_result = model.solve(solver_name=solver_name)

    # ====== 4. 获取和保存结果 ======
    print("\n[4/4] 处理优化结果...")

    results = model.get_results()

    # 更新总时间
    total_time = time.time() - total_start_time
    model.statistics["total_time"] = total_time
    model.statistics["data_load_time"] = data_time

    # 打印结果摘要
    if verbose:
        print("\n" + "-" * 40)
        print("第一阶段优化结果摘要")
        print("-" * 40)
        if results['objective'] is not None:
            print(f"  目标函数值 (网损pu): {results['objective']:.6f}")
        else:
            print(f"  目标函数值: 未获取到")
        print(f"  平均网损: {results['loss']['average_kw']:.2f} kW")
        print(f"  总网损: {results['loss']['total_kw']:.2f} kWh (等效)")
        print(f"  电压范围: [{results['voltage']['min']:.4f}, {results['voltage']['max']:.4f}] pu")
        print(f"  OLTC动作次数: {results['oltc']['n_actions']}")
        print(f"  SC动作次数: {results['sc']['n_actions']}")

    return {
        "results": results,
        "statistics": model.statistics,
        "scenario": scenario,
        "network": network,
        "config": config,
    }


def run_real_time_optimization(config: dict, day_ahead_output: dict,
                               verbose: bool = True) -> dict:
    """
    运行实时优化（第二阶段）

    Args:
        config: 配置字典
        day_ahead_output: 第一阶段输出结果
        verbose: 是否显示详细信息

    Returns:
        包含结果和统计信息的字典
    """
    print("\n" + "=" * 60)
    print("第二阶段: 实时无功优化 (SOCP)")
    print("=" * 60)

    total_start_time = time.time()

    # 从第一阶段获取数据
    network = day_ahead_output["network"]
    scenario = day_ahead_output["scenario"]
    day_ahead_results = day_ahead_output["results"]

    # ====== 1. 构建优化模型 ======
    print("\n[1/3] 构建优化模型...")

    model = create_real_time_model(config)
    model.build_model(network, scenario, day_ahead_results)

    print(f"  变量数: {model.statistics['n_variables']}")
    print(f"  约束数: {model.statistics['n_constraints']}")
    print(f"  二元变量数: {model.statistics['n_binary_vars']} (SOCP无整数变量)")

    # ====== 2. 求解模型 ======
    print("\n[2/3] 求解优化模型...")

    solver_name = config["optimization"]["solver"]["name"]
    solve_result = model.solve(solver_name=solver_name)

    # ====== 3. 获取结果 ======
    print("\n[3/3] 处理优化结果...")

    results = model.get_results()

    # 更新总时间
    total_time = time.time() - total_start_time
    model.statistics["total_time"] = total_time

    # 打印结果摘要
    if verbose:
        print("\n" + "-" * 40)
        print("第二阶段优化结果摘要")
        print("-" * 40)
        if results['objective'] is not None:
            print(f"  目标函数值: {results['objective']:.6f}")
        else:
            print(f"  目标函数值: 未获取到")
        print(f"  平均网损: {results['loss']['average_kw']:.2f} kW")
        print(f"  总网损: {results['loss']['total_kw']:.2f} kWh (等效)")
        print(f"  电压范围: [{results['voltage']['min']:.4f}, {results['voltage']['max']:.4f}] pu")
        print(f"  电压越限总量: {results['voltage']['total_violation']:.6f}")

        # DG无功统计
        pv_q = results['pv_reactive']['q_mvar']
        wt_q = results['wt_reactive']['q_mvar']
        svc_q = results['svc_reactive']['q_mvar']

        print(f"  PV无功范围: [{pv_q.min():.4f}, {pv_q.max():.4f}] MVar")
        if wt_q.size > 0:
            print(f"  WT无功范围: [{wt_q.min():.4f}, {wt_q.max():.4f}] MVar")
        else:
            print(f"  WT无功范围: 无WT配置")
        print(f"  SVC无功范围: [{svc_q.min():.4f}, {svc_q.max():.4f}] MVar")

    return {
        "results": results,
        "statistics": model.statistics,
        "scenario": scenario,
        "network": network,
        "config": config,
    }


def save_combined_results(day_ahead_output: dict, real_time_output: dict,
                          network_name: str, scenario_id: str,
                          output_dir: Path):
    """
    保存两阶段联合结果（统一格式）

    Args:
        day_ahead_output: 第一阶段输出
        real_time_output: 第二阶段输出
        network_name: 网络名称
        scenario_id: 场景ID
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_for_json(obj):
        """递归转换numpy对象为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    def safe_scalar(value, default=0, converter=float):
        """
        安全地将值转换为标量

        Args:
            value: 输入值（可能是标量、数组或None）
            default: 默认值
            converter: 转换函数（int或float）

        Returns:
            转换后的标量值
        """
        if value is None:
            return default

        # 如果是numpy数组，提取标量
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            elif value.size == 1:
                value = value.item()
            else:
                # 如果是多元素数组，取总和或平均值
                if converter == int:
                    value = int(np.sum(value))
                else:
                    value = float(np.mean(value))

        # 转换为目标类型
        try:
            return converter(value)
        except (ValueError, TypeError):
            return default

    # 提取结果
    da_results = day_ahead_output["results"]
    rt_results = real_time_output["results"]
    da_stats = day_ahead_output["statistics"]
    rt_stats = real_time_output["statistics"]

    # ===== 构造完整的结果字典 =====
    full_results = {
        "scenario_id": scenario_id,
        "network": network_name,
        "task": "vvc",
        "model_name": "Two-Stage MISOCP+SOCP VVC",

        # ===== 决策变量（可用于warm-start）=====
        "oltc": {
            "tap": convert_for_json(da_results["oltc"]["tap"]),
            "n_actions": safe_scalar(da_results["oltc"]["n_actions"], converter=int),
        },
        "sc": {
            "buses": convert_for_json(da_results["sc"]["buses"]),
            "stage": convert_for_json(da_results["sc"]["stage"]),
            "q_mvar": convert_for_json(da_results["sc"]["q_mvar"]),
            "n_actions": convert_for_json(da_results["sc"]["n_actions"]),
        },
        "pv_reactive": {
            "buses": convert_for_json(rt_results["pv_reactive"]["buses"]),
            "q_mvar": convert_for_json(rt_results["pv_reactive"]["q_mvar"]),
        },
        "wt_reactive": {
            "buses": convert_for_json(rt_results["wt_reactive"]["buses"]),
            "q_mvar": convert_for_json(rt_results["wt_reactive"]["q_mvar"]),
        },
        "svc_reactive": {
            "buses": convert_for_json(rt_results["svc_reactive"]["buses"]),
            "q_mvar": convert_for_json(rt_results["svc_reactive"]["q_mvar"]),
        },

        # ===== 性能指标（使用第二阶段结果）=====
        "voltage": {
            "values": convert_for_json(rt_results["voltage"]["values"]),
            "min": safe_scalar(rt_results["voltage"]["min"], converter=float),
            "max": safe_scalar(rt_results["voltage"]["max"], converter=float),
            "mean": safe_scalar(rt_results["voltage"]["mean"], converter=float),
            "violation": safe_scalar(rt_results["voltage"].get("violation", 0), converter=int),
            "total_violation": safe_scalar(rt_results["voltage"].get("total_violation", 0), converter=float),
        },
        "loss": {
            "per_period_kw": convert_for_json(rt_results["loss"]["per_period_kw"]),
            "total_kw": safe_scalar(rt_results["loss"]["total_kw"], converter=float),
            "average_kw": safe_scalar(rt_results["loss"]["average_kw"], converter=float),
        },

        # ===== 求解统计 =====
        "statistics": {
            "day_ahead": {
                "solve_time": safe_scalar(da_stats.get("solve_time", 0), converter=float),
                "solver_status": str(da_stats.get("solver_status", "unknown")),
                "n_variables": safe_scalar(da_stats.get("n_variables", 0), converter=int),
                "n_constraints": safe_scalar(da_stats.get("n_constraints", 0), converter=int),
                "n_binary_vars": safe_scalar(da_stats.get("n_binary_vars", 0), converter=int),
                "objective_value": convert_for_json(da_stats.get("objective_value")),
            },
            "real_time": {
                "solve_time": safe_scalar(rt_stats.get("solve_time", 0), converter=float),
                "solver_status": str(rt_stats.get("solver_status", "unknown")),
                "n_variables": safe_scalar(rt_stats.get("n_variables", 0), converter=int),
                "n_constraints": safe_scalar(rt_stats.get("n_constraints", 0), converter=int),
                "objective_value": convert_for_json(rt_stats.get("objective_value")),
            },
            "total_solve_time": safe_scalar(
                da_stats.get("solve_time", 0), converter=float
            ) + safe_scalar(
                rt_stats.get("solve_time", 0), converter=float
            ),
        },

        # ===== 其他信息 =====
        "objective": convert_for_json(rt_results.get("objective")),
        "timestamp": datetime.now().isoformat(),
    }

    # 保存文件，统一命名：{network}_vvc_scenario_{scenario_id}_results.json
    filename = f"{network_name}_vvc_scenario_{scenario_id}_results.json"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n两阶段联合结果已保存: {filepath}")

    return filepath


def print_comparison(day_ahead_output: dict, real_time_output: dict):
    """打印两阶段对比"""
    da_results = day_ahead_output["results"]
    rt_results = real_time_output["results"]
    da_stats = day_ahead_output["statistics"]
    rt_stats = real_time_output["statistics"]

    print("\n" + "=" * 60)
    print("两阶段优化结果对比")
    print("=" * 60)

    print(f"\n{'指标':<25} {'第一阶段(MISOCP)':<20} {'第二阶段(SOCP)':<20}")
    print("-" * 65)
    print(f"{'平均网损 (kW)':<25} {da_results['loss']['average_kw']:<20.2f} {rt_results['loss']['average_kw']:<20.2f}")
    print(f"{'电压最小值 (pu)':<25} {da_results['voltage']['min']:<20.4f} {rt_results['voltage']['min']:<20.4f}")
    print(f"{'电压最大值 (pu)':<25} {da_results['voltage']['max']:<20.4f} {rt_results['voltage']['max']:<20.4f}")
    print(f"{'求解时间 (s)':<25} {da_stats['solve_time']:<20.2f} {rt_stats['solve_time']:<20.2f}")
    print(f"{'变量数':<25} {da_stats['n_variables']:<20} {rt_stats['n_variables']:<20}")
    print(f"{'约束数':<25} {da_stats['n_constraints']:<20} {rt_stats['n_constraints']:<20}")

    # 网损改善
    loss_improvement = (da_results['loss']['average_kw'] - rt_results['loss']['average_kw']) / da_results['loss'][
        'average_kw'] * 100
    print(f"\n网损改善: {loss_improvement:.2f}%")
    print(f"总求解时间: {da_stats['solve_time'] + rt_stats['solve_time']:.2f} s")
    print("=" * 60)


def visualize_results(output: dict, figures_dir: Path, stage: str = "day_ahead"):
    """生成可视化图表"""
    results = output["results"]
    statistics = output["statistics"]

    visualizer = get_visualizer(figures_dir)

    if stage == "day_ahead":
        visualizer.plot_all(results, save=True)
        visualizer.create_summary_figure(results, statistics, save=True)
    else:
        # 第二阶段可视化
        visualizer.plot_voltage_profile(results, save=True)
        visualizer.plot_voltage_heatmap(results, save=True)
        visualizer.plot_power_loss(results, save=True)


def evaluate_results(output: dict, stage: str = "day_ahead") -> dict:
    """评估优化结果"""
    results = output["results"]
    statistics = output["statistics"]

    calculator = get_metrics_calculator()
    metrics = calculator.calculate_all_metrics(results, statistics)

    print(f"\n【{stage.upper()} 阶段评估报告】")
    calculator.print_metrics_report(metrics)

    return metrics


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="配电网多时间尺度无功优化 (Task A: VVC)")
    parser.add_argument("--network", "-n", type=str, default="ieee123",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"],
                        help="网络系统")
    parser.add_argument("--scenario-id", "-s", type=str, default="006",
                        help="场景ID")
    parser.add_argument("--data-file", type=str, default=None,
                        help="数据文件名")
    parser.add_argument("--stage", type=str, default="both",
                        choices=["day_ahead", "real_time", "both"],
                        help="优化阶段")
    parser.add_argument("--solver", type=str, default="gurobi",
                        help="求解器名称")
    parser.add_argument("--no-visualize", action="store_true", default=True,
                        help="不生成可视化图表")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="显示详细信息")

    args = parser.parse_args()

    # 加载配置
    network_name = args.network.lower()
    scenario_id = args.scenario_id
    config = get_network_config(network_name, PROJECT_ROOT)

    # 获取输出目录
    RESULTS_DIR = Path("opt_results") / "vvc"
    FIGURES_DIR = config["paths"]["figures"]

    # 更新配置
    config["optimization"]["solver"]["name"] = args.solver

    # 确定数据文件（优先使用命令行参数，否则根据场景ID自动生成）
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = get_scenario_data_file(network_name, scenario_id)

    print("\n" + "=" * 60)
    print("配电网多时间尺度无功优化程序 (Task A: VVC)")
    print("=" * 60)
    print(f"  网络系统: {network_name.upper()}")
    print(f"  场景ID: {scenario_id}")
    print(f"  数据文件: {data_file}")
    print(f"  优化阶段: {args.stage}")
    print(f"  求解器: {args.solver}")
    print(f"  结果目录: {RESULTS_DIR}")

    day_ahead_output = None
    real_time_output = None

    # ====== 第一阶段 ======
    if args.stage in ["day_ahead", "both"]:
        print("\n" + "=" * 60)
        print(">>> 运行第一阶段: 日前优化 (MISOCP)")
        print("=" * 60)

        day_ahead_output = run_day_ahead_optimization(
            config,
            network_name=network_name,
            data_file=data_file,
            scenario_id=None,  # 场景ID已在文件名中
            verbose=args.verbose
        )

        # 评估结果
        if args.verbose:
            print("\n>>> 评估第一阶段结果...")
            evaluate_results(day_ahead_output, stage="day_ahead")

        # 可视化
        if not args.no_visualize:
            print("\n>>> 生成第一阶段可视化图表...")
            visualize_results(day_ahead_output, FIGURES_DIR, stage="day_ahead")

    # ====== 第二阶段 ======
    if args.stage in ["real_time", "both"]:
        # 如果只运行第二阶段，需要先加载第一阶段结果
        if day_ahead_output is None:
            print("\n错误: 运行第二阶段需要先有第一阶段的结果")
            print("请使用 --stage both 或先运行 --stage day_ahead")
            return None

        print("\n" + "=" * 60)
        print(">>> 运行第二阶段: 实时优化 (SOCP)")
        print("=" * 60)

        real_time_output = run_real_time_optimization(
            config,
            day_ahead_output=day_ahead_output,
            verbose=args.verbose
        )

        # 评估结果
        if args.verbose:
            print("\n>>> 评估第二阶段结果...")
            evaluate_results(real_time_output, stage="real_time")

        # 可视化
        if not args.no_visualize:
            print("\n>>> 生成第二阶段可视化图表...")
            visualize_results(real_time_output, FIGURES_DIR, stage="real_time")

    # ====== 两阶段联合结果 ======
    if args.stage == "both" and day_ahead_output and real_time_output:
        # 保存联合结果（唯一保存点）
        print("\n>>> 保存优化结果...")
        save_combined_results(
            day_ahead_output,
            real_time_output,
            network_name,
            scenario_id,
            RESULTS_DIR
        )

        # 打印对比
        print_comparison(day_ahead_output, real_time_output)

    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)

    # 返回结果
    if args.stage == "both":
        return {"day_ahead": day_ahead_output, "real_time": real_time_output}
    elif args.stage == "day_ahead":
        return day_ahead_output
    else:
        return real_time_output


if __name__ == "__main__":
    output = main()