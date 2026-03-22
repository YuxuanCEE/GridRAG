# -*- coding: utf-8 -*-
"""
网络配置模块 - 支持多种IEEE测试系统

本模块提供统一的配置接口，支持：
- IEEE 13节点系统
- IEEE 33节点系统
- IEEE 69节点系统
- IEEE 123节点系统

使用方法：
    from config_networks import get_network_config, get_network_instance
    
    # 获取特定网络的配置
    config = get_network_config("ieee13")
    
    # 获取网络实例
    network = get_network_instance("ieee13", config)
"""

from pathlib import Path


# ==================== IEEE 13节点系统配置 ====================
IEEE13_CONFIG = {
    "network": {
        "name": "ieee13",
        "n_buses": 13,
        "root_bus": 0,
        "v_base": 4.16,      # 基准电压 kV
        "s_base": 5.0,       # 基准功率 MVA
        "v_min": 0.95,       # 电压下限 pu
        "v_max": 1.05,       # 电压上限 pu
    },
    
    "devices": {
        # OLTC配置（IEEE 13有调压器）
        "oltc": {
            "bus": 0,
            "tap_min": -8,
            "tap_max": 8,
            "tap_step": 0.00625,     # 每档调节量 pu
            "max_daily_actions": 4,
            "v0": 1.0,
        },
        
        # SC配置（IEEE 13在675和611节点有电容器）
        "sc": {
            "buses": [11, 9],        # 0-indexed: 675, 611
            "n_stages": 2,           # 简化为2档
            "q_per_stage": 0.15,     # 每档无功 MVar
            "max_daily_actions": 5,
        },
        
        # PV配置（参考论文：634, 680, 684节点）
        "pv": {
            "buses": [3, 7, 8],      # 0-indexed: 634, 680, 684
            "capacity": [0.5, 0.5, 0.5],  # 容量 MW
            "columns": ["node_634_PV", "node_680_PV", "node_684_PV"],
        },
        
        # WT配置（IEEE 13较小，可选择不配置风电）
        "wt": {
            "buses": [],
            "capacity": [],
            "columns": [],
        },
        
        # SVC配置
        "svc": {
            "buses": [6, 11],        # 0-indexed: 671, 675
            "q_min": -0.3,
            "q_max": 0.3,
        },
        
        # ESS储能配置
        "ess": {
            "buses": [6],            # 0-indexed: 671
            "capacity_mwh": [1.0],
            "max_charge_rate": 0.2,
            "max_discharge_rate": 0.2,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "soc_init": 0.5,
            "soc_final_constraint": True,
            "cost_per_mwh": 40,
            "charge_discharge_mutex": True,
        },
        
        # EV充电站配置
        "ev_stations": {
            "enabled": True,
            "buses": [12],           # 0-indexed: 692
            "n_vehicles": [20],
            "battery_kwh": 60,
            "max_charge_rate_kw": 60.0,
            "max_cut_ratio": 0.8,    # 最大可切比例（0.8=80%）
            "charge_efficiency": 0.95,
            "target_soc": 0.9,
            "penalty_soc_shortage": 500,
            "penalty_interruption": 50,
            "interruption_threshold_kw": 1.0,
            "data_file": "ev_profiles_13.csv",
        },
        
        # 联络线开关配置（IEEE 13原始无联络线，可扩展）
        "tie_switches": {
            "enabled": False,
            "branches": [],
            "initial_status": [],
            "max_daily_actions": 4,
            "switching_cost": 50,
        },
    },
    
    "data": {
        "dg_data_file": "scenario_{scenario_id}_13.csv",  # 模板，{scenario_id}会被替换
        "ev_data_file": "ev_profiles_13.csv",
        "load_file": None,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "resolution_minutes": 15,
        "data_type": "pu",
        "default_scenario_id": "004",
    },
}


# ==================== IEEE 33节点系统配置 ====================
IEEE33_CONFIG = {
    "network": {
        "name": "ieee33",
        "n_buses": 33,
        "root_bus": 0,
        "v_base": 12.66,
        "s_base": 1.0,
        "v_min": 0.95,
        "v_max": 1.05,
    },
    
    "devices": {
        "oltc": {
            "bus": 0,
            "tap_min": -5,
            "tap_max": 5,
            "tap_step": 0.01,
            "max_daily_actions": 4,
            "v0": 1.0,
        },
        
        "sc": {
            "buses": [15, 19, 30],
            "n_stages": 3,
            "q_per_stage": 0.1,
            "max_daily_actions": 5,
        },
        
        "pv": {
            "buses": [17, 32],
            "capacity": [0.5, 0.5],
            "columns": ["node_18_PV", "node_33_PV"],
        },
        
        "wt": {
            "buses": [21, 24],
            "capacity": [0.5, 0.5],
            "columns": ["node_22_wind", "node_25_wind"],
        },
        
        "svc": {
            "buses": [10, 28],
            "q_min": -0.4,
            "q_max": 0.4,
        },
        
        "ess": {
            "buses": [9, 22, 30],
            "capacity_mwh": [2.0, 2.0, 2.0],
            "max_charge_rate": 0.15,
            "max_discharge_rate": 0.15,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "soc_init": 0.5,
            "soc_final_constraint": True,
            "cost_per_mwh": 40,
            "charge_discharge_mutex": True,
        },
        
        "ev_stations": {
            "enabled": True,
            "buses": [14, 20, 28],
            "n_vehicles": [30, 20, 35],
            "battery_kwh": 60,
            "max_charge_rate_kw": 60.0,
            "max_cut_ratio": 0.8,
            "charge_efficiency": 0.95,
            "target_soc": 0.9,
            "penalty_soc_shortage": 500,
            "penalty_interruption": 50,
            "interruption_threshold_kw": 1.0,
            "data_file": "ev_profiles_33.csv",
        },
        
        "tie_switches": {
            "enabled": True,
            "branches": [
                {"id": 33, "from": 8, "to": 21, "r_ohm": 2.0, "x_ohm": 2.0},
                {"id": 34, "from": 9, "to": 15, "r_ohm": 2.0, "x_ohm": 2.0},
                {"id": 35, "from": 12, "to": 22, "r_ohm": 2.0, "x_ohm": 2.0},
            ],
            "initial_status": [0, 0, 0],
            "max_daily_actions": 6,
            "switching_cost": 50,
        },
    },
    
    "data": {
        "dg_data_file": "scenario_{scenario_id}_33.csv",  # 模板，{scenario_id}会被替换
        "ev_data_file": "ev_profiles_33.csv",
        "load_file": None,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "resolution_minutes": 15,
        "data_type": "pu",
        "default_scenario_id": "004",
    },
}


# ==================== IEEE 69节点系统配置 ====================
# 参考文献: Baran & Wu 1989, Savier & Das 2007
# DER配置基于拓扑结构和负荷分布
IEEE69_CONFIG = {
    "network": {
        "name": "ieee69",
        "n_buses": 69,
        "root_bus": 0,
        "v_base": 12.66,      # 基准电压 kV
        "s_base": 1.0,        # 基准功率 MVA
        "v_min": 0.95,        # 电压下限 pu
        "v_max": 1.05,        # 电压上限 pu
    },
    
    "devices": {
        # OLTC配置（变电站调压器）
        "oltc": {
            "bus": 0,
            "tap_min": -5,
            "tap_max": 5,
            "tap_step": 0.01,
            "max_daily_actions": 4,
            "v0": 1.0,
        },
        
        # SC配置（电容器组）
        # 布置在负荷较重的末端节点：12, 50, 62 (0-indexed: 11, 49, 61)
        "sc": {
            "buses": [11, 49, 61],
            "n_stages": 3,
            "q_per_stage": 0.12,     # 每档无功 MVar
            "max_daily_actions": 5,
        },
        
        # PV配置
        # 布置节点: 27, 50, 62 (0-indexed: 26, 49, 61)
        "pv": {
            "buses": [26, 49, 61],
            "capacity": [0.4, 0.6, 0.5],  # 容量 MW
            "columns": ["node_27_PV", "node_50_PV", "node_62_PV"],
        },
        
        # WT配置
        # 布置节点: 18, 65 (0-indexed: 17, 64)
        "wt": {
            "buses": [17, 64],
            "capacity": [0.4, 0.5],       # 容量 MW
            "columns": ["node_18_wind", "node_65_wind"],
        },
        
        # SVC配置（静态无功补偿）
        # 布置节点: 18, 51 (0-indexed: 17, 50)
        "svc": {
            "buses": [17, 50],
            "q_min": -0.4,
            "q_max": 0.4,
        },
        
        # ESS储能配置
        # 布置节点: 12, 50 (0-indexed: 11, 49)
        "ess": {
            "buses": [11, 49],
            "capacity_mwh": [2.0, 2.5],
            "max_charge_rate": 0.15,
            "max_discharge_rate": 0.15,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "soc_init": 0.5,
            "soc_final_constraint": True,
            "cost_per_mwh": 40,
            "charge_discharge_mutex": True,
        },
        
        # EV充电站配置
        # 布置节点: 28, 51, 66 (0-indexed: 27, 50, 65)
        "ev_stations": {
            "enabled": True,
            "buses": [27, 50, 65],
            "n_vehicles": [30, 40, 30],
            "battery_kwh": 60,
            "max_charge_rate_kw": 60.0,
            "max_cut_ratio": 0.8,
            "charge_efficiency": 0.95,
            "target_soc": 0.9,
            "penalty_soc_shortage": 500,
            "penalty_interruption": 50,
            "interruption_threshold_kw": 1.0,
            "data_file": "ev_profiles_69.csv",
        },
        
        # 联络线开关配置
        # IEEE 69标准联络线配置（修正版，确保连接不同分支）
        # 分支1末端: 35, 分支2末端: 46, 分支3末端: 50, 分支5末端: 64, 分支6末端: 68
        "tie_switches": {
            "enabled": True,
            "branches": [
                {"id": 69, "from": 10, "to": 42, "r_ohm": 2.0, "x_ohm": 2.0},  # 主馈线-分支2
                {"id": 70, "from": 35, "to": 46, "r_ohm": 2.0, "x_ohm": 2.0},  # 分支1末端-分支2末端
                {"id": 71, "from": 14, "to": 45, "r_ohm": 2.0, "x_ohm": 2.0},  # 主馈线-分支2
                {"id": 72, "from": 26, "to": 64, "r_ohm": 2.0, "x_ohm": 2.0},  # 主馈线-分支5
                {"id": 73, "from": 49, "to": 58, "r_ohm": 2.0, "x_ohm": 2.0},  # 分支3-分支5
            ],
            "initial_status": [0, 0, 0, 0, 0],
            "max_daily_actions": 6,
            "switching_cost": 50,
        },
    },
    
    "data": {
        "dg_data_file": "scenario_{scenario_id}_69.csv",  # 模板，{scenario_id}会被替换
        "ev_data_file": "ev_profiles_69.csv",
        "load_file": None,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "resolution_minutes": 15,
        "data_type": "pu",
        "default_scenario_id": "004",
    },
}


# ==================== IEEE 123节点系统配置（预留） ====================
IEEE123_CONFIG = {
    "network": {
        "name": "ieee123",
        "n_buses": 123,
        "root_bus": 0,
        "v_base": 4.16,      # 基准电压 kV
        "s_base": 5.0,       # 基准功率 MVA
        "v_min": 0.95,       # 电压下限 pu
        "v_max": 1.05,       # 电压上限 pu
    },
    
    "devices": {
        # OLTC配置（变电站调压器）
        "oltc": {
            "bus": 0,
            "tap_min": -8,
            "tap_max": 8,
            "tap_step": 0.00625,
            "max_daily_actions": 6,
            "v0": 1.0,
        },
        
        # SC配置（电容器组）
        "sc": {
            "buses": [82, 87, 64],           # 0-indexed: 节点83, 88, 65
            "n_stages": 4,
            "q_per_stage": 0.12,             # 每档无功 MVar（统一值）
            "max_daily_actions": 4,
        },
        
        # SVC配置（静态无功补偿）
        "svc": {
            "buses": [49, 109, 70],          # 0-indexed: 节点50, 110, 71
            "q_min": -0.3,                   # MVar
            "q_max": 0.3,                    # MVar
        },
        
        # PV配置（基于论文Fig.5b）
        # 原始节点: 32, 50, 110, 88, 83
        "pv": {
            "buses": [31, 49, 109, 87, 82],  # 0-indexed
            "capacity": [0.6, 0.8, 0.7, 0.5, 0.5],  # MW
            "pf_min": 0.85,
            "columns": ["node_32_PV", "node_50_PV", "node_110_PV", 
                       "node_88_PV", "node_83_PV"],
        },
        
        # WT配置（选择节点41, 71, 57作为风电）
        # 原始节点: 41, 71, 57
        "wt": {
            "buses": [40, 70, 56],           # 0-indexed
            "capacity": [0.6, 0.5, 0.4],     # MW
            "pf_min": 0.85,
            "columns": ["node_41_wind", "node_71_wind", "node_57_wind"],
        },
        
        # ESS配置（储能系统）
        "ess": {
            "buses": [49, 87],               # 0-indexed: 节点50, 88
            "capacity_mwh": [2.0, 1.5],
            "max_charge_rate": 0.25,         # 最大充电功率比率 (C-rate)
            "max_discharge_rate": 0.25,      # 最大放电功率比率
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "soc_init": 0.5,
            "soc_final_constraint": True,
            "cost_per_mwh": 40,
            "charge_discharge_mutex": True,
        },
        
        # 联络线配置（网络重构）
        "tie_switches": {
            "enabled": True,
            "tie_lines": [
                {"from": 53, "to": 95, "r": 0.03, "x": 0.02, "normally_open": True},
                {"from": 75, "to": 95, "r": 0.025, "x": 0.018, "normally_open": True},
                {"from": 42, "to": 88, "r": 0.035, "x": 0.023, "normally_open": True},
            ],
        },
        
        # EV充电站配置
        # 原始节点: 48, 65, 76, 96, 114
        "ev_stations": {
            "enabled": True,
            "buses": [47, 64, 75, 95, 113],  # 0-indexed
            "n_vehicles": [30, 25, 20, 20, 25],  # 快充场景下相应减少
            "battery_kwh": 60,                # 单车电池容量
            "max_charge_rate_kw": 60.0,       # 支持快充
            "max_cut_ratio": 0.8,
            "charge_efficiency": 0.95,
            "target_soc": 0.9,
            "penalty_soc_shortage": 500,
            "penalty_interruption": 50,
            "interruption_threshold_kw": 1.0,
            "data_file": "ev_profiles_123.csv",
        },
    },
    
    "data": {
        "dg_data_file": "scenario_{scenario_id}_123.csv",
        "ev_data_file": "ev_profiles_123.csv",
        "load_file": None,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "resolution_minutes": 15,
        "data_type": "pu",
        "default_scenario_id": "004",
    },
}


# ==================== 通用优化配置 ====================
OPTIMIZATION_CONFIG = {
    "task": "multi_timescale_var_opt",
    "method": "model_based",
    
    "solver": {
        "name": "gurobi",
        "time_limit": 3600,
        "mip_gap": 5e-4,
        "verbose": True,
    },
    
    "day_ahead": {
        "enabled": True,
        "interval_minutes": 15,
        "n_periods": 96,
        "objective": "min_loss",
    },
    
    "real_time": {
        "enabled": True,
        "interval_minutes": 5,
        "n_periods_per_hour": 12,
        "objective": "min_loss_voltage",
        "voltage_penalty": 1000,
        "loss_price": 450,
    },
    
    "ed": {
        "enabled": True,
        "interval_minutes": 15,
        "n_periods": 96,
        "delta_t": 0.25,
        "objective": "min_cost",
        "allow_pv_curtailment": True,
        "pv_curtailment_cost": 700,
        "loss_cost": 500,
        "enable_reconfiguration": True,
        "fix_topology_per_hour": False,
    },
    
    "joint": {
        "enabled": True,
        "interval_minutes": 15,
        "n_periods": 96,
        "delta_t": 0.25,
        "use_oltc": True,
        "use_sc": True,
        "use_svc": True,
        "use_ess": True,
        "use_tie_switches": True,
        "use_ev": True,
        "allow_pv_curtailment": True,
        "weights": {
            "loss": [0.3, 0.2, 0.2],
            "voltage": [0.2, 0.2, 0.2],
            "renewable": [0.2, 0.2, 0.3],
            "cost": [0.3, 0.4, 0.3],
            "ev_satisfaction": [0.2, 0.2, 0.3],
        },
        "voltage_soft_constraint": True,
        "voltage_penalty_coef": 1000,
        "pv_curtailment_cost": 700,
        "loss_cost": 500,
        # EV可切负荷阶梯惩罚配置（元/MWh削减量）
        # 削减越多，边际惩罚越高，鼓励尽量少削减
        "ev_curtailment_penalty_tier1": 200,   # 0-30%削减
        "ev_curtailment_penalty_tier2": 800,   # 30-60%削减
        "ev_curtailment_penalty_tier3": 1500,  # 60-80%削减
    },
}


# ==================== 分时电价配置 ====================
PRICE_CONFIG = {
    "peak_hours": [9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
    "valley_hours": [1, 2, 3, 4, 5, 22, 23, 24],
    "flat_hours": [6, 7, 8, 14, 15],
    "peak_price": 1322,
    "flat_price": 832,
    "valley_price": 369,
}


# ==================== 网络配置索引 ====================
NETWORK_CONFIGS = {
    "ieee13": IEEE13_CONFIG,
    "ieee33": IEEE33_CONFIG,
    "ieee69": IEEE69_CONFIG,
    "ieee123": IEEE123_CONFIG,
}


def get_network_config(network_name: str, project_root: Path = None) -> dict:
    """
    获取指定网络的完整配置
    
    Args:
        network_name: 网络名称 ("ieee13", "ieee33", "ieee69", "ieee123")
        project_root: 项目根目录（可选）
    
    Returns:
        完整配置字典
    """
    network_name = network_name.lower()
    
    if network_name not in NETWORK_CONFIGS:
        raise ValueError(f"不支持的网络类型: {network_name}. "
                        f"可选: {list(NETWORK_CONFIGS.keys())}")
    
    base_config = NETWORK_CONFIGS[network_name]
    
    # 设置项目路径
    if project_root is None:
        project_root = Path(__file__).parent
    
    data_dir = project_root / "data"
    profiles_dir = data_dir / "profiles"
    results_dir = project_root / "opt_results"
    figures_dir = results_dir / "figures"
    
    # 确保目录存在
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 组合完整配置
    config = {
        "project_root": project_root,
        "network": base_config["network"],
        "devices": base_config["devices"],
        "optimization": OPTIMIZATION_CONFIG.copy(),
        "price": PRICE_CONFIG.copy(),
        "data": base_config["data"].copy(),
        "paths": {
            "data": data_dir,
            "profiles": profiles_dir,
            "results": results_dir,
            "figures": figures_dir,
        }
    }
    
    return config


def get_scenario_data_file(network_name: str, scenario_id: str) -> str:
    """
    根据网络名称和场景ID生成数据文件名
    
    Args:
        network_name: 网络名称 (如 "ieee13", "ieee33")
        scenario_id: 场景ID (如 "004", "005")
    
    Returns:
        数据文件名 (如 "scenario_004_33.csv")
    """
    network_name = network_name.lower()
    
    # 提取节点数
    if network_name == "ieee13":
        n_buses = 13
    elif network_name == "ieee33":
        n_buses = 33
    elif network_name == "ieee69":
        n_buses = 69
    elif network_name == "ieee123":
        n_buses = 123
    else:
        raise ValueError(f"不支持的网络类型: {network_name}")
    
    return f"scenario_{scenario_id}_{n_buses}.csv"


def get_ev_data_file(network_name: str, scenario_id: str) -> str:
    """
    根据网络名称和场景ID生成EV数据文件名
    
    Args:
        network_name: 网络名称 (如 "ieee13", "ieee33")
        scenario_id: 场景ID (如 "004", "005")
    
    Returns:
        EV数据文件名 (如 "ev_profiles_004_33.csv")
    """
    network_name = network_name.lower()
    buses_map = {"ieee13": 13, "ieee33": 33, "ieee69": 69, "ieee123": 123}
    n_buses = buses_map.get(network_name, 33)
    return f"ev_profiles_{scenario_id}_{n_buses}.csv"


def get_result_filename(network_name: str, task: str, scenario_id: str, 
                        suffix: str = "results") -> str:
    """
    生成统一格式的结果文件名
    
    Args:
        network_name: 网络名称 (如 "ieee13", "ieee33")
        task: 任务名称 (如 "vvc", "ed", "joint")
        scenario_id: 场景ID (如 "004")
        suffix: 文件后缀 (如 "results", "day_ahead", "real_time")
    
    Returns:
        结果文件名 (如 "ieee33_vvc_scenario_004_results.json")
    """
    network_name = network_name.lower()
    task = task.lower()
    return f"{network_name}_{task}_scenario_{scenario_id}_{suffix}.json"


def get_network_buses(network_name: str) -> int:
    """获取网络节点数"""
    network_name = network_name.lower()
    buses_map = {"ieee13": 13, "ieee33": 33, "ieee69": 69, "ieee123": 123}
    return buses_map.get(network_name, 0)


def get_network_instance(network_name: str, config: dict = None):
    """
    获取网络实例
    
    Args:
        network_name: 网络名称
        config: 配置字典（可选，如果为None则使用默认配置）
    
    Returns:
        网络实例
    """
    network_name = network_name.lower()
    
    if config is None:
        config = get_network_config(network_name)
    
    device_config = config.get("devices", {})
    tie_config = device_config.get("tie_switches", {"enabled": False})
    ev_config = device_config.get("ev_stations", {"enabled": False})
    
    if network_name == "ieee13":
        from data.network.ieee13 import get_ieee13_network
        return get_ieee13_network(tie_switch_config=tie_config,
                                  ev_station_config=ev_config)
    
    elif network_name == "ieee33":
        from data.network.ieee33 import get_ieee33_network
        return get_ieee33_network(tie_switch_config=tie_config,
                                  ev_station_config=ev_config)
    
    elif network_name == "ieee69":
        from data.network.ieee69 import get_ieee69_network
        return get_ieee69_network(tie_switch_config=tie_config,
                                  ev_station_config=ev_config)
    
    elif network_name == "ieee123":
        from data.network.ieee123 import get_ieee123_network
        return get_ieee123_network(tie_switch_config=tie_config,
                                   ev_station_config=ev_config)
    
    else:
        raise ValueError(f"不支持的网络类型: {network_name}")


def list_available_networks():
    """列出所有可用的网络"""
    print("可用的测试网络:")
    for name, cfg in NETWORK_CONFIGS.items():
        net_cfg = cfg["network"]
        print(f"  - {name}: {net_cfg['n_buses']}节点, "
              f"V_base={net_cfg['v_base']}kV, "
              f"S_base={net_cfg['s_base']}MVA")


if __name__ == "__main__":
    # 测试
    list_available_networks()
    
    print("\n测试IEEE 13节点配置:")
    config_13 = get_network_config("ieee13")
    print(f"  网络: {config_13['network']['name']}")
    print(f"  PV节点: {config_13['devices']['pv']['buses']}")
    print(f"  数据文件: {config_13['data']['dg_data_file']}")
    
    print("\n测试IEEE 33节点配置:")
    config_33 = get_network_config("ieee33")
    print(f"  网络: {config_33['network']['name']}")
    print(f"  PV节点: {config_33['devices']['pv']['buses']}")
    print(f"  数据文件: {config_33['data']['dg_data_file']}")
