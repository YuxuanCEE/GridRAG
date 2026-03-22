# -*- coding: utf-8 -*-
"""
经济调度(ED)优化主程序 (Task B)

支持在不同IEEE测试系统上运行：
  - IEEE 13节点
  - IEEE 33节点
  - IEEE 69节点（预留）
  - IEEE 123节点（预留）

运行含储能和网络重构的配电网经济调度优化

使用方法：
    python main_ed.py --network ieee13
    python main_ed.py --network ieee33
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_networks import get_network_config, get_network_instance, get_scenario_data_file, get_result_filename
from data.data_loader import get_data_loader
from models.ed.socp_ed import create_ed_model


def run_ed_optimization(config: dict, network_name: str, scenario_id: str,
                        data_file: str = None, verbose: bool = True) -> dict:
    """
    运行经济调度优化（含网络重构）
    
    Args:
        config: 配置字典
        network_name: 网络名称 (如 "ieee13", "ieee33")
        scenario_id: 场景ID (如 "004", "005")
        data_file: 数据文件名（可选，默认根据网络和场景ID自动生成）
        verbose: 是否打印详细信息
    
    Returns:
        优化结果字典
    """
    # 如果未指定数据文件，自动生成
    if data_file is None:
        data_file = get_scenario_data_file(network_name, scenario_id)
    
    RESULTS_DIR = config["paths"]["results"]
    
    print(f"\n{'='*60}")
    print(f"经济调度(ED)优化 - {network_name.upper()}")
    print(f"场景ID: {scenario_id}")
    print(f"数据文件: {data_file}")
    print(f"{'='*60}")
    
    # ====== 1. 加载网络（含联络线配置）======
    print("\n[1/4] 加载网络数据...")
    
    network = get_network_instance(network_name, config)
    
    print(f"  节点数: {network.n_buses}")
    print(f"  固定支路数: {network.n_branches}")
    print(f"  联络线数: {network.n_tie_switches}")
    
    # 打印储能配置
    ess_config = config["devices"]["ess"]
    print(f"  储能节点: {[b+1 for b in ess_config['buses']]}")
    print(f"  储能容量: {ess_config['capacity_mwh']} MWh")
    
    # ====== 2. 加载场景数据 ======
    print("\n[2/4] 加载场景数据...")
    loader = get_data_loader(config)
    
    ed_config = config["optimization"]["ed"]
    n_periods = ed_config["n_periods"]
    
    scenario_data = loader.get_scenario_data(
        filename=data_file,
        n_periods=n_periods,
    )
    
    print(f"  时段数: {n_periods}")
    print(f"  PV节点: {list(scenario_data['pv'].keys())}")
    print(f"  WT节点: {list(scenario_data['wt'].keys())}")
    
    # 打印电价信息
    price_config = config["price"]
    print(f"\n  分时电价:")
    print(f"    峰时 ({price_config['peak_hours']}): {price_config['peak_price']} 元/MWh")
    print(f"    平时 ({price_config['flat_hours']}): {price_config['flat_price']} 元/MWh")
    print(f"    谷时 ({price_config['valley_hours']}): {price_config['valley_price']} 元/MWh")
    
    # ====== 3. 构建并求解ED模型 ======
    print("\n[3/4] 构建ED-SOCP模型...")
    ed_model = create_ed_model(config)
    ed_model.build_model(network, scenario_data)
    
    print(f"  变量数: {ed_model.statistics['n_variables']}")
    print(f"  二进制变量数: {ed_model.statistics['n_binary_vars']}")
    print(f"  约束数: {ed_model.statistics['n_constraints']}")
    
    print("\n[4/4] 求解优化模型...")
    solve_result = ed_model.solve(solver_name=config["optimization"]["solver"]["name"])
    
    # ====== 4. 获取并保存结果 ======
    if solve_result["termination"] in ["optimal", "feasible", "maxTimeLimit"]:
        results = ed_model.get_results()
        
        # 打印摘要
        ed_model.print_summary()
        
        # 保存结果 - 使用统一命名格式
        output_dir = RESULTS_DIR / "ed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统一命名格式: {network}_{task}_scenario_{scenario_id}_results.json
        output_file = output_dir / get_result_filename(network_name, "ed", scenario_id, "results")
        
        # 转换numpy数组为列表
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "network": network_name,
                "scenario_id": scenario_id,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "n_periods": n_periods,
                    "delta_t": ed_config["delta_t"],
                    "ess_buses": ess_config["buses"],
                    "ess_capacity": ess_config["capacity_mwh"],
                    "enable_reconfiguration": ed_config.get("enable_reconfiguration", False),
                    "n_tie_switches": network.n_tie_switches,
                },
                "results": serializable_results,
                "statistics": ed_model.statistics,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至: {output_file}")
        
        return {
            "success": True,
            "results": results,
            "statistics": ed_model.statistics,
            "output_file": str(output_file),
        }
    else:
        print(f"\n求解失败: {solve_result}")
        return {
            "success": False,
            "error": solve_result,
        }


def plot_ed_results(results: dict, output_dir: Path = None):
    """
    绘制ED优化结果图
    
    Args:
        results: 优化结果字典
        output_dir: 图片输出目录
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    if output_dir is None:
        output_dir = RESULTS_DIR / "ed" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_periods = len(results["grid"]["power_mw"])
    time_axis = np.arange(n_periods) * 0.25  # 小时
    
    # 检查是否有网络重构结果
    has_reconfig = results.get("reconfiguration") is not None
    n_plots = 5 if has_reconfig else 4
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    
    # 1. 购电功率与电价
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(time_axis, results["grid"]["power_mw"], 'b-', linewidth=2, label='购电功率')
    ax1.fill_between(time_axis, 0, results["grid"]["power_mw"], alpha=0.3)
    ax1.set_ylabel('购电功率 (MW)', color='b')
    ax1.set_xlabel('时间 (h)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    price = np.array(results["price"]["profile"])
    ax1_twin.step(time_axis, price, 'r--', where='post', linewidth=1.5, label='电价')
    ax1_twin.set_ylabel('电价 (元/MWh)', color='r')
    ax1_twin.legend(loc='upper right')
    
    ax1.set_title('购电功率与分时电价')
    
    # 2. 储能充放电
    ax2 = axes[1]
    ess_charge = np.array(results["ess"]["charge_mw"])
    ess_discharge = np.array(results["ess"]["discharge_mw"])
    n_ess = ess_charge.shape[1]
    
    colors = ['blue', 'green', 'orange']
    for k in range(n_ess):
        ax2.bar(time_axis - 0.08 + k*0.08, -ess_charge[:, k], width=0.08, 
                label=f'ESS{k+1}充电', alpha=0.7, color=colors[k])
        ax2.bar(time_axis - 0.08 + k*0.08, ess_discharge[:, k], width=0.08,
                alpha=0.7, color=colors[k], hatch='//')
    
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_ylabel('功率 (MW)')
    ax2.set_xlabel('时间 (h)')
    ax2.set_title('储能充放电功率（正:放电，负:充电）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 储能SOC
    ax3 = axes[2]
    ess_soc = np.array(results["ess"]["soc_mwh"])
    
    for k in range(n_ess):
        ax3.plot(time_axis, ess_soc[:, k], linewidth=2, label=f'ESS{k+1}')
    
    ax3.set_ylabel('SOC (MWh)')
    ax3.set_xlabel('时间 (h)')
    ax3.set_title('储能SOC状态')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 电压曲线
    ax4 = axes[3]
    voltage = np.array(results["voltage"]["values"])
    
    ax4.fill_between(time_axis, voltage.min(axis=1), voltage.max(axis=1), 
                     alpha=0.3, label='电压范围')
    ax4.plot(time_axis, voltage.mean(axis=1), 'b-', linewidth=2, label='平均电压')
    ax4.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='下限')
    ax4.axhline(y=1.05, color='r', linestyle='--', linewidth=1, label='上限')
    
    ax4.set_ylabel('电压 (pu)')
    ax4.set_xlabel('时间 (h)')
    ax4.set_title('节点电压')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.92, 1.08)
    
    # 5. 联络线开关状态（如果有）
    if has_reconfig:
        ax5 = axes[4]
        reconfig = results["reconfiguration"]
        tie_status = np.array(reconfig["status"])
        n_ties = tie_status.shape[1]
        
        colors_tie = ['red', 'green', 'blue']
        for k in range(n_ties):
            info = reconfig["tie_info"][k]
            label = f"联络线{info['id']} ({info['from']}-{info['to']})"
            ax5.step(time_axis, tie_status[:, k] + k*0.1, where='post', 
                    linewidth=2, label=label, color=colors_tie[k])
        
        ax5.set_ylabel('开关状态')
        ax5.set_xlabel('时间 (h)')
        ax5.set_title(f'联络线开关状态 (总切换次数: {reconfig["total_switches"]})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-0.1, 1.5)
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['断开', '闭合'])
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = output_dir / "ed_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存至: {fig_path}")
    
    plt.close()


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="配电网经济调度优化 (Task B: ED)")
    parser.add_argument("--network", "-n", type=str, default="ieee123",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"],
                        help="网络系统 (默认: ieee33)")
    parser.add_argument("--scenario-id", "-s", type=str, default="002",
                        help="场景ID，如 004, 005 等 (默认: 004)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="数据文件名 (默认: 根据网络和场景ID自动生成)")
    parser.add_argument("--solver", type=str, default="gurobi",
                        help="求解器名称")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="显示详细信息")
    
    args = parser.parse_args()
    
    # 加载配置
    network_name = args.network.lower()
    scenario_id = args.scenario_id
    config = get_network_config(network_name, PROJECT_ROOT)
    
    # 更新求解器配置
    config["optimization"]["solver"]["name"] = args.solver
    
    # 确定数据文件（优先使用命令行参数，否则根据场景ID自动生成）
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = get_scenario_data_file(network_name, scenario_id)
    
    print("\n" + "=" * 60)
    print("配电网经济调度优化程序 (Task B: ED)")
    print("=" * 60)
    print(f"  网络系统: {network_name.upper()}")
    print(f"  场景ID: {scenario_id}")
    print(f"  数据文件: {data_file}")
    print(f"  求解器: {args.solver}")
    
    # 运行ED优化
    result = run_ed_optimization(
        config, 
        network_name=network_name, 
        scenario_id=scenario_id,
        data_file=data_file,
        verbose=args.verbose
    )
    
    if result["success"]:
        # 尝试绘图
        try:
            plot_ed_results(result["results"])
        except Exception as e:
            print(f"绘图失败: {e}")
        
        print("\n" + "="*60)
        print("ED优化完成！")
        print("="*60)
    else:
        print("\nED优化失败！")
    
    return result


if __name__ == "__main__":
    main()
