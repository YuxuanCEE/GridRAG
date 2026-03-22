# -*- coding: utf-8 -*-
"""
Task C: 综合优化主程序 (Joint Optimization)
融合 VVC + ED + EV调度的多目标优化

支持在不同IEEE测试系统上运行：
  - IEEE 13节点
  - IEEE 33节点
  - IEEE 69节点（预留）
  - IEEE 123节点（预留）

功能:
- 加载网络（含联络线和EV充电站）
- 加载风光负荷场景数据
- 加载EV充电场景数据
- 构建并求解MISOCP优化模型
- 输出优化结果

使用方法:
    python main_joint.py --network ieee13
    python main_joint.py --network ieee33

作者: GridRAG Team
日期: 2025-01
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_networks import get_network_config, get_network_instance, get_scenario_data_file, get_result_filename, get_ev_data_file
from models.joint.socp_joint import create_joint_model


def _diagnose_power_balance(model, config):
    """
    诊断功率平衡，验证结果是否合理
    
    检查：
    1. 每个时段的供需平衡
    2. 购电量是否合理
    3. 电压水平是否正常
    4. 是否存在数值异常
    """
    results = model.get_results()
    network = model.network
    s_base = model.s_base
    
    # 计算理论负荷
    fixed_load_mw = network.total_p_load_mw
    
    # 计算EV负荷
    ev_load_avg = 0
    if "ev" in results:
        ev_power_mw = results["ev"]["power_mw"]  # (n_periods, n_stations)
        ev_load_avg = ev_power_mw.mean()
    
    # 计算DER出力
    der_avg = 0
    if "grid" in results:
        # 简化：从购电量推算DER
        p_grid_avg = results["grid"]["power_mw"].mean()
        der_avg = fixed_load_mw + ev_load_avg - p_grid_avg  # 近似
    
    print(f"\n平均功率平衡（忽略损耗）：")
    print(f"  固定负荷: {fixed_load_mw:.3f} MW")
    print(f"  EV负荷: {ev_load_avg:.3f} MW")
    print(f"  总需求: {fixed_load_mw + ev_load_avg:.3f} MW")
    print(f"  DER出力(估计): {der_avg:.3f} MW")
    print(f"  购电: {p_grid_avg:.3f} MW")
    print(f"  供需差: {abs(p_grid_avg + der_avg - fixed_load_mw - ev_load_avg):.4f} MW")
    
    # 检查异常
    warnings = []
    
    if p_grid_avg < 0.5 * fixed_load_mw:
        warnings.append(f"⚠️  购电量({p_grid_avg:.2f}MW)远小于固定负荷({fixed_load_mw:.2f}MW)，不合理")
    
    if "ev" in results and ev_load_avg < 0.01:
        warnings.append(f"⚠️  EV负荷接近0({ev_load_avg*1000:.1f}kW)，可能数据加载有问题")
    
    volt = results.get("voltage", {})
    if volt.get("min", 1.0) > 1.04:
        warnings.append(f"⚠️  电压过高(最低{volt['min']:.4f}pu)，可能约束有问题")
    
    if len(warnings) > 0:
        print(f"\n⚠️  发现 {len(warnings)} 个潜在问题：")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n✅ 功率平衡检查通过")
    
    # 详细时段检查（抽样）
    print(f"\n时段抽样检查（时段0, 48, 95）：")
    for t in [0, 48, 95]:
        if t >= len(results["grid"]["power_mw"]):
            continue
        
        p_grid_t = results["grid"]["power_mw"][t]
        p_ev_t = results["ev"]["power_mw"][t].sum() if "ev" in results else 0
        
        print(f"  t={t}: P_grid={p_grid_t:.3f}MW, P_ev={p_ev_t:.3f}MW")


def _get_load_curve(n_periods: int = 96) -> np.ndarray:
    """
    生成典型日负荷变化曲线（标准化因子）
    
    ★ 必须与 data/data_loader.py → DataLoader.get_load_curve() 完全一致
    优化器通过该曲线缩放base负荷: p_load[t] = p_load_pu * load_factor[t]
    """
    hours = np.linspace(0, 24, n_periods, endpoint=False)
    
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
    
    return load_curve


def load_dg_scenario_data(config: dict, n_periods: int = 96, data_file: str = None) -> dict:
    """
    加载分布式电源场景数据（PV、WT）
    
    Args:
        config: 配置字典
        n_periods: 时段数
        data_file: 数据文件名（可选，默认使用配置中的文件）
    
    Returns:
        场景数据字典
    """
    profiles_dir = config["paths"]["profiles"]
    
    # 如果未指定数据文件，使用配置中的默认文件
    if data_file is None:
        data_file = config["data"]["dg_data_file"]
    
    data_path = profiles_dir / data_file
    
    # ★ 默认使用时变负荷曲线，与 data/data_loader.py 和 Task A/B 一致
    scenario_data = {
        "pv": {},
        "wt": {},
        "load_factor": _get_load_curve(n_periods),
    }
    
    if not data_path.exists():
        print(f"[警告] 场景数据文件不存在: {data_path}")
        print("[警告] 使用默认零值数据")
        return scenario_data
    
    print(f"加载场景数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 确保数据长度匹配
    data_len = min(len(df), n_periods)
    
    # 加载PV数据
    pv_config = config["devices"]["pv"]
    for i, bus in enumerate(pv_config["buses"]):
        col_name = pv_config["columns"][i] if i < len(pv_config["columns"]) else f"pv_{bus}"
        if col_name in df.columns:
            # 数据类型：标幺值 -> MW
            pv_pu = df[col_name].values[:data_len]
            pv_mw = pv_pu * pv_config["capacity"][i]
            
            # 填充到n_periods
            full_data = np.zeros(n_periods)
            full_data[:data_len] = pv_mw
            scenario_data["pv"][bus] = full_data
            print(f"  PV节点{bus+1}: 峰值{pv_mw.max():.3f}MW")
        else:
            scenario_data["pv"][bus] = np.zeros(n_periods)
    
    # 加载WT数据
    wt_config = config["devices"]["wt"]
    for i, bus in enumerate(wt_config["buses"]):
        col_name = wt_config["columns"][i] if i < len(wt_config["columns"]) else f"wt_{bus}"
        if col_name in df.columns:
            wt_pu = df[col_name].values[:data_len]
            wt_mw = wt_pu * wt_config["capacity"][i]
            
            full_data = np.zeros(n_periods)
            full_data[:data_len] = wt_mw
            scenario_data["wt"][bus] = full_data
            print(f"  WT节点{bus+1}: 峰值{wt_mw.max():.3f}MW")
        else:
            scenario_data["wt"][bus] = np.zeros(n_periods)
    
    # 加载负荷因子（如果有的话）
    if "load_factor" in df.columns:
        lf = df["load_factor"].values[:data_len]
        full_lf = np.ones(n_periods)
        full_lf[:data_len] = lf
        scenario_data["load_factor"] = full_lf
    
    return scenario_data


def load_ev_data(config: dict, n_periods: int = 96, ev_file: str = None) -> dict:
    """
    加载EV充电场景数据
    
    数据格式（预期）:
    datetime, station_0_load_kw, station_0_soc, station_1_load_kw, station_1_soc, ...
    
    注意：
    - load_kw 是单车功率，会乘以车辆数得到站级功率
    - 支持快充（最高60kW）和慢充（7kW）混合模式
    - max_charge_rate_kw 仅用于设置优化变量上限，不做截断

    Args:
        config: 配置字典
        n_periods: 时段数
        ev_file: EV数据文件名（可选，默认使用配置中的文件）

    Returns:
        EV数据字典 {station_id: {"load_kw": array, "soc": array}}
    """
    ev_config = config["devices"].get("ev_stations", {})
    if not ev_config.get("enabled", False):
        return {}

    profiles_dir = config["paths"]["profiles"]

    # 如果未指定ev_file，使用配置中的默认文件
    if ev_file is None:
        ev_file = ev_config.get("data_file", "ev_profiles.csv")

    data_path = profiles_dir / ev_file

    n_stations = len(ev_config.get("buses", []))
    ev_data = {}

    if not data_path.exists():
        print(f"[警告] EV数据文件不存在: {data_path}")
        print("[警告] 生成默认EV充电曲线")

        # 生成默认充电曲线
        for k in range(n_stations):
            station = {
                "load_kw": np.zeros(n_periods),
                "soc": np.zeros(n_periods),
            }

            n_vehicles = ev_config["n_vehicles"][k] if k < len(ev_config["n_vehicles"]) else 30
            max_rate = ev_config.get("max_charge_rate_kw", 60.0)

            for t in range(n_periods):
                hour = (t * 15) / 60

                # 夜间充电高峰（18:00-06:00）
                if 18 <= hour <= 24 or 0 <= hour < 6:
                    if 18 <= hour <= 24:
                        progress = (hour - 18) / 12
                    else:
                        progress = (hour + 6) / 12

                    active_ratio = 0.3 + 0.4 * np.sin(np.pi * progress)
                    # 假设混合充电：30%快充 + 70%慢充
                    avg_power = 0.3 * max_rate + 0.7 * 7.0
                    station["load_kw"][t] = n_vehicles * avg_power * active_ratio * 0.5
                    station["soc"][t] = 0.3 + 0.6 * progress
                else:
                    station["load_kw"][t] = n_vehicles * 7.0 * 0.1
                    station["soc"][t] = 0.5
            
            ev_data[k] = station
            print(f"  EV站{k}: 生成默认曲线, 峰值{station['load_kw'].max():.1f}kW")
        
        return ev_data
    
    # 从文件加载
    print(f"加载EV数据: {data_path}")
    df = pd.read_csv(data_path)
    data_len = min(len(df), n_periods)
    
    for k in range(n_stations):
        load_col = f"station_{k}_load_kw"
        soc_col = f"station_{k}_soc"
        
        station = {
            "load_kw": np.zeros(n_periods),
            "soc": np.zeros(n_periods),
        }
        
        # ✅ 获取该站的车辆数
        n_vehicles = ev_config["n_vehicles"][k] if k < len(ev_config["n_vehicles"]) else 100
        
        if load_col in df.columns:
            # ✅ 修复：CSV中是单车功率，需要乘以车辆数量
            single_vehicle_load = df[load_col].values[:data_len]
            station["load_kw"][:data_len] = single_vehicle_load * n_vehicles
            
            # 数据合理性检查
            peak_single = single_vehicle_load.max()
            peak_station = station["load_kw"].max()
            max_rate = ev_config.get("max_charge_rate_kw", 7.0)
            
            if peak_station > n_vehicles * max_rate * 1.1:
                print(f"  ⚠️  EV站{k}: 峰值功率({peak_station:.1f}kW)超过理论最大值({n_vehicles * max_rate:.1f}kW)，已截断")
                station["load_kw"] = np.minimum(station["load_kw"], n_vehicles * max_rate * 0.98)
                peak_station = station["load_kw"].max()
        
        if soc_col in df.columns:
            station["soc"][:data_len] = df[soc_col].values[:data_len]
        
        ev_data[k] = station
        print(f"  EV站{k}: 峰值{peak_station:.1f}kW (单车{peak_single:.3f}kW × {n_vehicles}辆)")
    
    return ev_data


def main(network_name: str = "ieee33", scenario_id: str = "004"):
    """
    主函数
    
    Args:
        network_name: 网络名称 (如 "ieee13", "ieee33")
        scenario_id: 场景ID (如 "004", "005")
    """
    print("\n" + "=" * 70)
    print(f"GridRAG Task C: 综合优化 (Joint Optimization)")
    print(f"融合 VVC + ED + EV调度的多目标MISOCP优化")
    print(f"网络: {network_name.upper()}, 场景: {scenario_id}")
    print("=" * 70)
    
    # ========== 1. 加载配置 ==========
    print("\n[1] 加载配置...")
    config = get_network_config(network_name, PROJECT_ROOT)
    
    joint_config = config["optimization"]["joint"]
    n_periods = joint_config.get("n_periods", 96)
    
    print(f"  时段数: {n_periods}")
    print(f"  使用OLTC: {joint_config.get('use_oltc', True)}")
    print(f"  使用SC: {joint_config.get('use_sc', True)}")
    print(f"  使用ESS: {joint_config.get('use_ess', True)}")
    print(f"  使用联络线: {joint_config.get('use_tie_switches', True)}")
    print(f"  使用EV: {joint_config.get('use_ev', True)}")
    
    # ========== 2. 加载网络 ==========
    print(f"\n[2] 加载{network_name.upper()}节点网络...")
    
    network = get_network_instance(network_name, config)
    network.summary()
    
    # ========== 3. 加载场景数据 ==========
    print("\n[3] 加载场景数据...")
    data_file = get_scenario_data_file(network_name, scenario_id)
    print(f"  数据文件: {data_file}")
    scenario_data = load_dg_scenario_data(config, n_periods, data_file=data_file)
    
    # ========== 4. 加载EV数据 ==========
    print("\n[4] 加载EV数据...")
    ev_file = get_ev_data_file(network_name, scenario_id)
    print(f"  EV数据文件: {ev_file}")
    ev_data = load_ev_data(config, n_periods, ev_file=ev_file)
    
    # ========== 5. 创建优化模型 ==========
    print("\n[5] 创建综合优化模型...")
    model = create_joint_model(config)
    model.load_network(network)
    model.load_scenario_data(scenario_data)
    model.load_ev_data(ev_data)
    
    # ========== 6. 构建模型 ==========
    print("\n[6] 构建优化模型...")
    model.build_model()
    
    # ========== 7. 求解模型 ==========
    print("\n[7] 求解优化模型...")
    solver_config = config["optimization"]["solver"]
    
    stats = model.solve(
        solver_name=solver_config.get("name", "gurobi"),
        time_limit=solver_config.get("time_limit", 3600),
        mip_gap=solver_config.get("mip_gap", 1e-3),
        verbose=solver_config.get("verbose", True)
    )
    
    # ========== 8. 输出结果 ==========
    print("\n[8] 输出结果...")
    model.print_summary()
    
    # ✅ 功率平衡诊断
    print("\n" + "=" * 70)
    print("功率平衡诊断")
    print("=" * 70)
    _diagnose_power_balance(model, config)
    
    # 保存结果
    results = model.get_results()
    
    # results_dir = config["paths"]["results"]
    # results_file = results_dir / "joint_optimization_results.npz"
    #
    # # 转换为可保存格式
    # save_dict = {}
    # for key, value in results.items():
    #     if isinstance(value, dict):
    #         for subkey, subvalue in value.items():
    #             if isinstance(subvalue, np.ndarray):
    #                 save_dict[f"{key}_{subkey}"] = subvalue
    #             elif isinstance(subvalue, (int, float)):
    #                 save_dict[f"{key}_{subkey}"] = np.array([subvalue])
    #     elif isinstance(value, np.ndarray):
    #         save_dict[key] = value
    #
    # np.savez(results_file, **save_dict)

    import json
    from datetime import datetime

    # 创建joint子目录
    results_dir = config["paths"]["results"] / "joint"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用统一命名格式: {network}_{task}_scenario_{scenario_id}_results.json
    results_file = results_dir / get_result_filename(network_name, "joint", scenario_id, "results")

    # 转换numpy数组为列表
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    # 添加元数据
    output_data = {
        "network": network_name,
        "scenario_id": scenario_id,
        "timestamp": datetime.now().isoformat(),
        "results": convert_to_serializable(results),
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {results_file}")
    
    print("\n" + "=" * 70)
    print("Task C 综合优化完成!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="配电网综合优化 (Task C: Joint)")
    parser.add_argument("--network", "-n", type=str, default="ieee33",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"],
                        help="网络系统")
    parser.add_argument("--scenario-id", "-s", type=str, default="001",
                        help="场景ID")
    parser.add_argument("--solver", type=str, default="gurobi",
                        help="求解器名称")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="显示详细信息")
    
    args = parser.parse_args()
    
    try:
        results = main(network_name=args.network.lower(), scenario_id=args.scenario_id)
    except Exception as e:
        print(f"\n[错误] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
