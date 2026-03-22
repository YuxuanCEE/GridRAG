# -*- coding: utf-8 -*-
"""
全局配置文件
包含项目路径、网络参数、优化设置等
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================
# 项目根目录（Windows路径）
PROJECT_ROOT = Path(r"D:\GridRAG")

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PROFILES_DIR = DATA_DIR / "profiles"
NETWORK_DIR = DATA_DIR / "network"

# 结果输出目录
RESULTS_DIR = PROJECT_ROOT / "opt_results"
FIGURES_DIR = RESULTS_DIR / "figures"

# 确保输出目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 网络配置 ====================
NETWORK_CONFIG = {
    "name": "ieee33",
    "n_buses": 33,
    "root_bus": 0,  # 根节点（变电站）
    "v_base": 12.66,  # 基准电压 kV
    "s_base": 1.0,    # 基准功率 MVA
    "v_min": 0.95,    # 电压下限 pu
    "v_max": 1.05,    # 电压上限 pu
}

# ==================== 设备配置 ====================
DEVICE_CONFIG = {
    # OLTC配置
    "oltc": {
        "bus": 0,                    # 安装节点（根节点）
        "tap_min": -5,               # 最小档位
        "tap_max": 5,                # 最大档位
        "tap_step": 0.01,            # 每档调节量 pu
        "max_daily_actions": 4,      # 每日最大动作次数
        "v0": 1.0,                   # 一次侧电压 pu
    },
    
    # SC配置（分组投切电容器）
    "sc": {
        "buses": [15, 19, 30],       # 安装节点（0-indexed: 16,20,31 -> 15,19,30）
        "n_stages": 3,               # 档位数
        "q_per_stage": 0.1,          # 每档无功 MVar
        "max_daily_actions": 5,      # 每日最大动作次数
    },
    
    # PV配置
    "pv": {
        "buses": [17, 32],           # 安装节点（0-indexed: 18,33 -> 17,32）
        "capacity": [0.5, 0.5],      # 容量 MW
        "columns": ["node_18_PV", "node_33_PV"],  # CSV中的列名
    },
    
    # WT配置 (风电)
    "wt": {
        "buses": [21, 24],           # 安装节点（0-indexed: 22,25 -> 21,24）
        "capacity": [0.5, 0.5],      # 容量 MW
        "columns": ["node_22_wind", "node_25_wind"],  # CSV中的列名
    },
    
    # SVC配置（第二阶段使用）
    "svc": {
        "buses": [10, 28],           # 安装节点（0-indexed: 11,29 -> 10,28）
        "q_min": -0.4,               # 最小无功 MVar
        "q_max": 0.4,                # 最大无功 MVar
    },
    
    # ESS储能配置（参考Enhanced IEEE 33 Bus论文）
    "ess": {
        "buses": [9, 22, 30],        # 安装节点（0-indexed: 10,23,31 -> 9,22,30）
        "capacity_mwh": [2.0, 2.0, 2.0],  # 容量 MWh
        "max_charge_rate": 0.15,     # 最大充电倍率 (P_max = rate * capacity)
        "max_discharge_rate": 0.15,  # 最大放电倍率
        "efficiency_charge": 0.95,   # 充电效率
        "efficiency_discharge": 0.95, # 放电效率
        "soc_min": 0.1,              # SOC下限
        "soc_max": 0.9,              # SOC上限
        "soc_init": 0.5,             # 初始SOC
        "soc_final_constraint": True, # 是否约束终端SOC回到初始值
        "cost_per_mwh": 40,          # 充放电折旧成本 元/MWh
        "charge_discharge_mutex": True,  # 是否启用充放电互斥约束（增加二进制变量）
    },
    
    # ==================== EV充电站配置 ====================
    # 参考文献: 祁向龙《多时间尺度协同的配电网分层深度强化学习电压控制策略》
    # 充电站位置: 节点15, 21, 29 (1-indexed) -> 14, 20, 28 (0-indexed)
    # EV数据来源: emobpy (Scientific Data)
    "ev_stations": {
        "enabled": True,
        "buses": [14, 20, 28],              # 安装节点（0-indexed）
        "n_vehicles": [200, 150, 250],      # 每站EV数量
        "battery_kwh": 50,                  # 单车电池容量 kWh
        "max_charge_rate_kw": 7.0,          # 单车最大充电功率 kW
        "charge_efficiency": 0.95,          # 充电效率
        "target_soc": 0.9,                  # 离开时目标SOC
        # 用户体验惩罚参数
        "penalty_soc_shortage": 500,        # 能量缺口惩罚 元/MWh
        "penalty_interruption": 50,         # 每次充电中断惩罚 元/次
        "interruption_threshold_kw": 1.0,   # 判定中断的功率下降阈值 kW
        # 数据文件
        "data_file": "ev_profiles.csv",     # EV场景数据文件
    },
    
    # ==================== 联络线开关配置（网络重构）====================
    # 参考文献: Enhanced IEEE 33 Bus Benchmark Test System (2021)
    # Table IV: 3条可切换联络线
    "tie_switches": {
        "enabled": True,              # 是否启用网络重构（联络线切换）
        "branches": [
            # id: 支路编号（从33开始，续接原有32条支路）
            # from/to: 0-indexed节点编号
            # r_ohm/x_ohm: 参考Enhanced IEEE 33 Table IV
            {"id": 33, "from": 8, "to": 21, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 34, "from": 9, "to": 15, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 35, "from": 12, "to": 22, "r_ohm": 2.0, "x_ohm": 2.0},
        ],
        "initial_status": [0, 0, 0],  # 初始状态：0=断开，1=闭合（默认辐射状运行）
        "max_daily_actions": 6,       # 每日最大切换次数（所有联络线总计）
        "switching_cost": 50,         # 每次切换成本（元），用于抑制频繁切换
    },
}

# ==================== 优化配置 ====================
OPTIMIZATION_CONFIG = {
    # 任务类型
    "task": "multi_timescale_var_opt",
    
    # 求解方法: "model_based" 或 "rl"
    "method": "model_based",
    
    # 求解器设置
    "solver": {
        "name": "gurobi",
        "time_limit": 3600,           # 求解时间限制（秒）
        "mip_gap": 5e-4,             # MIP间隙
        "verbose": True,             # 是否显示求解过程
    },
    
    # 第一阶段：日前调度 (VVO)
    "day_ahead": {
        "enabled": True,
        "interval_minutes": 15,      # 调度间隔（分钟）
        "n_periods": 96,             # 时段数（24h × 4）
        "objective": "min_loss",     # 目标函数：最小化网损
    },
    
    # 第二阶段：实时调度 (VVO)
    "real_time": {
        "enabled": True,
        "interval_minutes": 5,       # 调度间隔（分钟）
        "n_periods_per_hour": 12,    # 每小时时段数
        "objective": "min_loss_voltage",  # 目标：最小化网损+电压越限
        "voltage_penalty": 1000,     # 电压越限惩罚系数
        "loss_price": 450,           # 边际网损系数 $/MWh
    },
    
    # 经济调度 (ED) - 含网络重构
    "ed": {
        "enabled": True,
        "interval_minutes": 15,      # 调度间隔（分钟）
        "n_periods": 96,             # 时段数（24h × 4）
        "delta_t": 0.25,             # 时间步长（小时）
        "objective": "min_cost",     # 目标函数：最小化运行成本
        "allow_pv_curtailment": True,  # 是否允许弃光
        "pv_curtailment_cost": 700,    # 弃光成本 元/MWh（补贴损失）
        "loss_cost": 500,              # 网损折算成本 元/MWh
        # 网络重构相关配置
        "enable_reconfiguration": True,  # 是否启用网络重构优化
        "fix_topology_per_hour": False,  # 是否每小时固定拓扑（减少切换）
    },
    
    # ==================== 综合优化 (Joint) - Task C ====================
    # 融合VVC + ED + EV调度的多目标优化
    # 参考文献: 陈海鹏《考虑电动汽车无功补偿与不确定性的配电网-电动汽车有功无功协同优化》
    "joint": {
        "enabled": True,
        "interval_minutes": 15,      # 调度间隔（分钟）
        "n_periods": 96,             # 时段数（24h × 4）
        "delta_t": 0.25,             # 时间步长（小时）
        
        # 可控设备开关
        "use_oltc": True,            # 使用OLTC
        "use_sc": True,              # 使用分组电容器
        "use_svc": True,             # 使用SVC
        "use_ess": True,             # 使用储能
        "use_tie_switches": True,    # 使用联络线开关
        "use_ev": True,              # 使用EV充电调度
        "allow_pv_curtailment": True,  # 允许弃光
        
        # 多目标权重（参考陈海鹏文章，按峰谷时段动态调整）
        # 格式: [峰时段, 平时段, 谷时段]
        "weights": {
            "loss": [0.3, 0.2, 0.2],           # β1: 网损权重
            "voltage": [0.2, 0.2, 0.2],        # β2: 电压偏差权重
            "renewable": [0.2, 0.2, 0.3],      # β3: 弃光惩罚权重
            "cost": [0.3, 0.4, 0.3],           # β4: 经济成本权重
            "ev_satisfaction": [0.2, 0.2, 0.3], # β5: EV用户体验权重（新增）
        },
        
        # 电压约束处理（软约束化）
        "voltage_soft_constraint": True,  # 使用软约束
        "voltage_penalty_coef": 1000,     # 电压越限惩罚系数
        
        # 成本参数
        "pv_curtailment_cost": 700,       # 弃光成本 元/MWh
        "loss_cost": 500,                 # 网损成本 元/MWh
        
        # ✅ EV可切负荷参数（新增）
        "ev_curtailment_penalty": 300,    # EV负荷削减惩罚 元/MW（削减越多，惩罚越高）
        # 说明：EV负荷最多可削减20%，削减会导致充电速度降低
        #      通过惩罚系数平衡削减成本和购电成本
    },
}

# ==================== 分时电价配置（参考中文文献Table B5）====================
PRICE_CONFIG = {
    # 峰时段（小时，1-indexed）
    "peak_hours": [9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
    # 谷时段
    "valley_hours": [1, 2, 3, 4, 5, 22, 23, 24],
    # 平时段（其余时段）
    "flat_hours": [6, 7, 8, 14, 15],
    
    # 电价（元/MWh，原文是元/kWh，这里乘1000转换）
    "peak_price": 1322,      # 1.322 元/kWh = 1322 元/MWh
    "flat_price": 832,       # 0.832 元/kWh = 832 元/MWh
    "valley_price": 369,     # 0.369 元/kWh = 369 元/MWh
}

# ==================== 数据文件配置 ====================
DATA_CONFIG = {
    # 风光数据文件（单文件包含所有节点）
    "dg_data_file": "scenario_004.csv",  # 默认数据文件
    
    # 负荷数据（如果有的话）
    "load_file": None,  # 暂时使用IEEE33标准负荷
    
    # EV数据文件
    "ev_data_file": "ev_profiles.csv",  # EV充电场景数据
    
    # 时间配置
    "date_format": "%Y-%m-%d %H:%M:%S",
    "resolution_minutes": 15,
    
    # 数据类型: "pu" (标幺值，需乘以容量) 或 "mw" (实际功率)
    "data_type": "pu",
}


def get_config():
    """获取完整配置"""
    return {
        "project_root": PROJECT_ROOT,
        "network": NETWORK_CONFIG,
        "devices": DEVICE_CONFIG,
        "optimization": OPTIMIZATION_CONFIG,
        "price": PRICE_CONFIG,
        "data": DATA_CONFIG,
        "paths": {
            "data": DATA_DIR,
            "profiles": PROFILES_DIR,
            "results": RESULTS_DIR,
            "figures": FIGURES_DIR,
        }
    }


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print("项目根目录:", config["project_root"])
    print("网络配置:", config["network"])
    print("设备配置:", config["devices"])
    
    # 打印联络线配置
    tie_config = config["devices"]["tie_switches"]
    print("\n联络线配置:")
    print(f"  启用: {tie_config['enabled']}")
    for tie in tie_config["branches"]:
        print(f"  支路{tie['id']}: 节点{tie['from']+1} <-> 节点{tie['to']+1}, "
              f"R={tie['r_ohm']}Ω, X={tie['x_ohm']}Ω")
