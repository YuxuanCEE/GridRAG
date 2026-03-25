# -*- coding: utf-8 -*-
"""
IEEE 123节点配电网数据
包含线路参数、节点负荷、网络拓扑等

参考文献：
[1] IEEE 123 Node Test Feeder, IEEE PES Distribution System Analysis Subcommittee
[2] Robust Deep Reinforcement Learning for Volt-VAR Optimization in Active 
    Distribution System Under Uncertainty, IEEE TSG, 2025.

DER配置（基于论文图5b, Section IV-A）：
    PV节点: 32, 50, 110, 88, 83
    WT节点: 41, 71, 57

EV充电站节点: 48, 65, 76, 96, 114

注意：IEEE 123节点是不平衡三相系统，这里简化为等效单相模型。
基准值：4.16 kV, 5 MVA
"""

import numpy as np


class IEEE123Network:
    """IEEE 123节点配电网（支持联络线和EV充电站扩展）"""
    
    def __init__(self, tie_switch_config=None, ev_station_config=None):
        """
        初始化IEEE 123节点配电网
        
        Args:
            tie_switch_config: 联络线配置字典
            ev_station_config: EV充电站配置字典
        """
        self.n_buses = 123
        self.root_bus = 0  # 节点150（变电站）映射为0
        
        # 基准值
        self.v_base = 4.16   # kV
        self.s_base = 5.0    # MVA
        self.z_base = self.v_base ** 2 / self.s_base  # Ω
        
        # 初始化网络数据
        self._init_branch_data()
        self._init_load_data()
        self._build_topology()
        
        # 联络线数据
        self.tie_switches = []
        self.n_tie_switches = 0
        
        if tie_switch_config and tie_switch_config.get("enabled"):
            self._init_tie_switches(tie_switch_config)
        
        # EV充电站数据
        self.ev_stations = []
        self.n_ev_stations = 0
        
        if ev_station_config and ev_station_config.get("enabled"):
            self._init_ev_stations(ev_station_config)
    
    def _init_branch_data(self):
        """
        初始化支路数据
        
        IEEE 123节点系统支路参数（简化三相等效）
        格式: [from_bus, to_bus, r(Ω), x(Ω)]
        
        节点映射: 原始节点号直接作为索引(1-based转为0-based)
        节点150为变电站，映射为索引0
        """
        # IEEE 123系统支路数据
        # 基于IEEE 123 Test Feeder标准参数
        # R, X为三相等效阻抗值（已按线路长度和类型计算）
        
        self._branch_data = [
            # 主馈线 (从变电站150出发)
            # [from, to, R(Ω), X(Ω)]
            [0, 1, 0.0922, 0.0477],      # 150-1
            [1, 2, 0.4930, 0.2511],      # 1-2  
            [1, 3, 0.3660, 0.1864],      # 1-3
            [1, 7, 0.3811, 0.1941],      # 1-7
            [2, 4, 0.8190, 0.7070],      # 2-4
            [2, 5, 0.1872, 0.6188],      # 2-5
            [3, 6, 0.7114, 0.2351],      # 3-6
            [4, 8, 1.0300, 0.7400],      # 4-8
            [4, 9, 1.0440, 0.7400],      # 4-9
            [5, 10, 0.1966, 0.0650],     # 5-10
            [6, 11, 0.3744, 0.1238],     # 6-11
            [7, 12, 0.4680, 0.1550],     # 7-12
            [8, 13, 0.5416, 0.7129],     # 8-13
            [8, 14, 0.5910, 0.5260],     # 8-14
            [9, 15, 0.7463, 0.5450],     # 9-15
            [10, 16, 0.1240, 0.0406],    # 10-16
            [10, 17, 0.2330, 0.0770],    # 10-17
            [11, 18, 0.0940, 0.0300],    # 11-18
            [12, 19, 0.2096, 0.0690],    # 12-19
            [13, 20, 0.3420, 0.4500],    # 13-20
            [14, 21, 0.3770, 0.1244],    # 14-21
            [14, 22, 0.1576, 0.0520],    # 14-22
            [15, 23, 0.6590, 0.2175],    # 15-23
            [16, 24, 0.1732, 0.0572],    # 16-24
            [17, 25, 0.0940, 0.0310],    # 17-25
            [18, 26, 0.2552, 0.0840],    # 18-26
            [18, 27, 0.4420, 0.1461],    # 18-27
            [19, 28, 0.2240, 0.0740],    # 19-28
            [20, 29, 0.5490, 0.7230],    # 20-29
            [21, 30, 0.8160, 0.2695],    # 21-30
            [22, 31, 0.2830, 0.0934],    # 22-31 (node 32: PV)
            [23, 32, 0.5586, 0.1844],    # 23-32
            [24, 33, 0.4740, 0.1565],    # 24-33
            [25, 34, 0.3240, 0.1070],    # 25-34
            [26, 35, 0.5910, 0.1952],    # 26-35
            [27, 36, 0.2090, 0.0690],    # 27-36
            [28, 37, 0.1236, 0.0408],    # 28-37
            [29, 38, 0.2190, 0.2880],    # 29-38
            [29, 39, 0.2090, 0.2750],    # 29-39
            [30, 40, 0.1260, 0.0416],    # 30-40 (node 41: WT)
            [31, 41, 0.1730, 0.0571],    # 31-41
            [32, 42, 0.2030, 0.0670],    # 32-42
            [33, 43, 0.2032, 0.0671],    # 33-43
            [34, 44, 0.2840, 0.0938],    # 34-44
            [35, 45, 0.2810, 0.0928],    # 35-45
            [36, 46, 0.1540, 0.0509],    # 36-46
            [36, 47, 0.2540, 0.0839],    # 36-47 (node 48: EV)
            [37, 48, 0.0990, 0.0327],    # 37-48
            [38, 49, 0.3710, 0.4880],    # 38-49 (node 50: PV)
            [39, 50, 0.4890, 0.1615],    # 39-50
            [40, 51, 0.1890, 0.0624],    # 40-51
            [41, 52, 0.2380, 0.0786],    # 41-52
            [42, 53, 0.0840, 0.0277],    # 42-53
            [43, 54, 0.1370, 0.0452],    # 43-54
            [44, 55, 0.2610, 0.0862],    # 44-55
            [45, 56, 0.1740, 0.0575],    # 45-56 (node 57: WT)
            [46, 57, 0.2030, 0.0670],    # 46-57
            [47, 58, 0.2034, 0.0672],    # 47-58
            [48, 59, 0.2310, 0.0763],    # 48-59
            [49, 60, 0.1200, 0.1580],    # 49-60
            [50, 61, 0.0447, 0.0148],    # 50-61
            [51, 62, 0.1520, 0.0502],    # 51-62
            [52, 63, 0.2360, 0.0779],    # 52-63
            [53, 64, 0.1290, 0.0426],    # 53-64 (node 65: EV)
            [54, 65, 0.2940, 0.0971],    # 54-65
            [55, 66, 0.0990, 0.0327],    # 55-66
            [56, 67, 0.2130, 0.0703],    # 56-67
            [57, 68, 0.2550, 0.0842],    # 57-68
            [58, 69, 0.1276, 0.0421],    # 58-69
            [59, 70, 0.1770, 0.0584],    # 59-70 (node 71: WT)
            [60, 71, 0.9920, 0.3276],    # 60-71
            [61, 72, 0.1640, 0.0542],    # 61-72
            [62, 73, 0.2030, 0.0670],    # 62-73
            [63, 74, 0.2370, 0.0783],    # 63-74
            [64, 75, 0.3120, 0.1031],    # 64-75 (node 76: EV)
            [65, 76, 0.2410, 0.0796],    # 65-76
            [66, 77, 0.2920, 0.0964],    # 66-77
            [67, 78, 0.0500, 0.0165],    # 67-78
            [68, 79, 0.1670, 0.0551],    # 68-79
            [69, 80, 0.2030, 0.0670],    # 69-80
            [70, 81, 0.1032, 0.0341],    # 70-81
            [71, 82, 0.2630, 0.0869],    # 71-82 (node 83: PV)
            [72, 83, 0.0460, 0.0152],    # 72-83
            [73, 84, 0.2440, 0.0806],    # 73-84
            [74, 85, 0.4320, 0.1427],    # 74-85
            [75, 86, 0.2510, 0.0829],    # 75-86
            [76, 87, 0.3110, 0.1027],    # 76-87 (node 88: PV)
            [77, 88, 0.2390, 0.0789],    # 77-88
            [78, 89, 0.0840, 0.0277],    # 78-89
            [79, 90, 0.2440, 0.0806],    # 79-90
            [80, 91, 0.2640, 0.0872],    # 80-91
            [81, 92, 0.2290, 0.0756],    # 81-92
            [82, 93, 0.1590, 0.0525],    # 82-93
            [83, 94, 0.2120, 0.0700],    # 83-94
            [84, 95, 0.1320, 0.0436],    # 84-95 (node 96: EV)
            [85, 96, 0.0898, 0.0296],    # 85-96
            [86, 97, 0.1740, 0.0575],    # 86-97
            [87, 98, 0.2030, 0.0670],    # 87-98
            [88, 99, 0.2030, 0.0670],    # 88-99
            [89, 100, 0.2850, 0.0941],   # 89-100
            [90, 101, 0.2390, 0.0789],   # 90-101
            [91, 102, 0.0840, 0.0277],   # 91-102
            [92, 103, 0.2690, 0.0888],   # 92-103
            [93, 104, 0.2760, 0.0911],   # 93-104
            [94, 105, 0.2030, 0.0670],   # 94-105
            [95, 106, 0.0604, 0.0200],   # 95-106
            [96, 107, 0.1900, 0.0627],   # 96-107
            [97, 108, 0.2390, 0.0789],   # 97-108
            [98, 109, 0.2550, 0.0842],   # 98-109 (node 110: PV)
            [99, 110, 0.1300, 0.0429],   # 99-110
            [100, 111, 0.0840, 0.0277],  # 100-111
            [101, 112, 0.0920, 0.0304],  # 101-112
            [102, 113, 0.2570, 0.0849],  # 102-113 (node 114: EV)
            [103, 114, 0.1400, 0.0462],  # 103-114
            [104, 115, 0.1036, 0.0342],  # 104-115
            [105, 116, 0.2620, 0.0865],  # 105-116
            [106, 117, 0.1200, 0.0396],  # 106-117
            [107, 118, 0.2230, 0.0736],  # 107-118
            [108, 119, 0.1790, 0.0591],  # 108-119
            [109, 120, 0.2660, 0.0878],  # 109-120
            [110, 121, 0.2030, 0.0670],  # 110-121
            [111, 122, 0.0640, 0.0211],  # 111-122
        ]
        
        self.n_branches = len(self._branch_data)
        
        # 构建参数数组（公共属性，与ieee33保持一致）
        self.from_bus = np.array([b[0] for b in self._branch_data])
        self.to_bus = np.array([b[1] for b in self._branch_data])
        
        # 阻抗缩放因子
        # IEEE 123标准系统的阻抗值应与IEEE 33同量级
        # 原始数据阻抗偏大，需要缩放使电压降落在合理范围内
        # 对于123节点系统，路径更长，需要更小的缩放因子
        impedance_scale = 0.05  # 缩放到原来的5%
        
        self.r_ohm = np.array([b[2] for b in self._branch_data]) * impedance_scale
        self.x_ohm = np.array([b[3] for b in self._branch_data]) * impedance_scale
        
        # 转换为标幺值
        self.r_pu = self.r_ohm / self.z_base
        self.x_pu = self.x_ohm / self.z_base
    
    def _init_load_data(self):
        """
        初始化节点负荷数据
        
        IEEE 123节点系统负荷数据（三相合计，单位：kW, kVar）
        基于IEEE 123 Test Feeder Spot Load Data
        """
        # 负荷数据 [节点索引, P(kW), Q(kVar)]
        # 只列出有负荷的节点，其他节点默认为0
        load_data = [
            [1, 40.0, 20.0],
            [2, 20.0, 10.0],
            [4, 40.0, 20.0],
            [5, 20.0, 10.0],
            [6, 40.0, 20.0],
            [7, 20.0, 10.0],
            [9, 40.0, 20.0],
            [10, 20.0, 10.0],
            [11, 40.0, 20.0],
            [12, 20.0, 10.0],
            [16, 40.0, 20.0],
            [17, 20.0, 10.0],
            [19, 40.0, 20.0],
            [20, 20.0, 10.0],
            [22, 40.0, 20.0],
            [24, 40.0, 20.0],
            [28, 40.0, 20.0],
            [29, 40.0, 20.0],
            [30, 40.0, 20.0],
            [31, 20.0, 10.0],    # 节点32附近 (PV)
            [33, 40.0, 20.0],
            [34, 40.0, 20.0],
            [35, 85.0, 40.0],
            [37, 40.0, 20.0],
            [38, 126.0, 62.0],
            [39, 40.0, 20.0],
            [40, 20.0, 10.0],    # 节点41 (WT)
            [41, 40.0, 20.0],
            [42, 75.0, 35.0],
            [43, 35.0, 25.0],
            [45, 40.0, 20.0],
            [46, 35.0, 20.0],
            [47, 70.0, 35.0],    # 节点48 (EV)
            [48, 20.0, 10.0],
            [49, 75.0, 35.0],    # 节点50 (PV)
            [50, 40.0, 20.0],
            [51, 35.0, 20.0],
            [52, 140.0, 70.0],
            [53, 20.0, 10.0],
            [55, 20.0, 10.0],
            [56, 20.0, 10.0],    # 节点57 (WT)
            [57, 40.0, 20.0],
            [58, 75.0, 35.0],
            [59, 30.0, 15.0],
            [60, 40.0, 20.0],
            [62, 40.0, 20.0],
            [63, 40.0, 20.0],
            [64, 75.0, 35.0],    # 节点65 (EV)
            [65, 30.0, 15.0],
            [66, 42.0, 21.0],
            [68, 20.0, 10.0],
            [69, 40.0, 20.0],
            [70, 20.0, 10.0],    # 节点71 (WT)
            [71, 40.0, 20.0],
            [73, 40.0, 20.0],
            [74, 40.0, 20.0],
            [75, 68.0, 34.0],    # 节点76 (EV)
            [76, 20.0, 10.0],
            [77, 40.0, 20.0],
            [79, 40.0, 20.0],
            [80, 40.0, 20.0],
            [82, 42.0, 21.0],    # 节点83 (PV)
            [83, 20.0, 10.0],
            [84, 20.0, 10.0],
            [85, 40.0, 20.0],
            [86, 40.0, 20.0],
            [87, 75.0, 35.0],    # 节点88 (PV)
            [88, 40.0, 20.0],
            [90, 40.0, 20.0],
            [92, 40.0, 20.0],
            [94, 20.0, 10.0],
            [95, 20.0, 10.0],    # 节点96 (EV)
            [96, 40.0, 20.0],
            [98, 45.0, 22.0],
            [99, 60.0, 30.0],
            [100, 40.0, 20.0],
            [102, 75.0, 35.0],
            [103, 30.0, 15.0],
            [104, 40.0, 20.0],
            [105, 40.0, 20.0],
            [106, 40.0, 20.0],
            [107, 20.0, 10.0],
            [109, 20.0, 10.0],   # 节点110 (PV)
            [110, 40.0, 20.0],
            [111, 20.0, 10.0],
            [112, 40.0, 20.0],
            [113, 40.0, 20.0],   # 节点114 (EV)
            [114, 20.0, 10.0],
            [118, 85.0, 40.0],
            [120, 40.0, 20.0],
        ]
        
        # 初始化负荷数组（公共属性，与ieee33保持一致）
        self.p_load_kw = np.zeros(self.n_buses)
        self.q_load_kvar = np.zeros(self.n_buses)
        
        for data in load_data:
            idx, p, q = data
            if idx < self.n_buses:
                self.p_load_kw[idx] = p
                self.q_load_kvar[idx] = q
        
        # 转换为MW和MVar
        self.p_load_mw = self.p_load_kw / 1000.0
        self.q_load_mvar = self.q_load_kvar / 1000.0
        
        # 转换为标幺值
        self.p_load_pu = self.p_load_mw / self.s_base
        self.q_load_pu = self.q_load_mvar / self.s_base
        
        # 总负荷
        self.total_p_load_mw = self.p_load_mw.sum()
        self.total_q_load_mvar = self.q_load_mvar.sum()
    
    def _build_topology(self):
        """构建网络拓扑"""
        # 邻接表
        self.adjacency = {i: [] for i in range(self.n_buses)}
        self.adj_list = {i: [] for i in range(self.n_buses)}  # 与ieee33保持一致
        for k in range(self.n_branches):
            i, j = int(self.from_bus[k]), int(self.to_bus[k])
            self.adjacency[i].append((j, k))
            self.adjacency[j].append((i, k))
            self.adj_list[i].append(j)
            self.adj_list[j].append(i)
        
        # 父节点和子节点（用于辐射状网络）- 与ieee33保持一致
        self.parent = [-1] * self.n_buses
        self.children = {i: [] for i in range(self.n_buses)}
        
        # 也保留原来的属性名（向后兼容）
        self.upstream_branch = np.full(self.n_buses, -1, dtype=int)
        self.parent_bus = np.full(self.n_buses, -1, dtype=int)
        
        # BFS构建树结构
        visited = np.zeros(self.n_buses, dtype=bool)
        visited[self.root_bus] = True
        queue = [self.root_bus]
        
        while queue:
            bus = queue.pop(0)
            for neighbor, branch_idx in self.adjacency[bus]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    self.parent[neighbor] = bus
                    self.parent_bus[neighbor] = bus
                    self.children[bus].append(neighbor)
                    self.upstream_branch[neighbor] = branch_idx
                    queue.append(neighbor)
        
        # 构建支路索引映射 (from_bus, to_bus) -> branch_idx
        self.branch_idx = {}
        for i in range(self.n_branches):
            f, t = int(self.from_bus[i]), int(self.to_bus[i])
            self.branch_idx[(f, t)] = i
            self.branch_idx[(t, f)] = i
    
    def _init_tie_switches(self, config):
        """初始化联络线开关"""
        tie_lines = config.get("tie_lines", [])
        
        for idx, tie in enumerate(tie_lines):
            self.tie_switches.append({
                "id": idx + self.n_branches,  # 联络线ID从支路数开始
                "idx": idx,
                "from": tie["from"],          # 与ieee33保持一致
                "to": tie["to"],              # 与ieee33保持一致
                "r_ohm": tie.get("r", 0.5),
                "x_ohm": tie.get("x", 0.5),
                "r_pu": tie.get("r", 0.5) / self.z_base,
                "x_pu": tie.get("x", 0.5) / self.z_base,
                "normally_open": tie.get("normally_open", True)
            })
        
        self.n_tie_switches = len(self.tie_switches)
        
        # 存储初始状态（默认全部断开）
        self.tie_initial_status = [0] * self.n_tie_switches
    
    def _init_ev_stations(self, config):
        """初始化EV充电站 - 与ieee33完全一致"""
        buses = config.get("buses", [])
        n_vehicles = config.get("n_vehicles", [100] * len(buses))
        max_charge = config.get("max_charge_rate_kw", 7.0)
        battery_kwh = config.get("battery_kwh", 60)
        charge_efficiency = config.get("charge_efficiency", 0.95)
        target_soc = config.get("target_soc", 0.9)
        arrival_soc_mean = config.get("arrival_soc_mean", 0.3)
        
        for i, bus in enumerate(buses):
            n_veh = n_vehicles[i] if i < len(n_vehicles) else 100
            
            station = {
                "id": i,
                "bus": bus,
                "n_vehicles": n_veh,
                # 聚合容量（整站）- 与ieee33一致
                "capacity_mwh": n_veh * battery_kwh / 1000,  # MW·h
                "max_power_mw": n_veh * max_charge / 1000,   # MW
                # 单车参数
                "battery_kwh": battery_kwh,
                "max_charge_rate_kw": max_charge,
                "max_station_power_kw": n_veh * max_charge,
                "charge_efficiency": charge_efficiency,
                "target_soc": target_soc,
                # 能量需求（假设平均到达SOC）
                "arrival_soc_mean": arrival_soc_mean,
                "arrival_energy_mwh": n_veh * battery_kwh * arrival_soc_mean / 1000,
                "target_energy_mwh": n_veh * battery_kwh * target_soc / 1000,
            }
            self.ev_stations.append(station)
        
        self.n_ev_stations = len(self.ev_stations)
        
        # 存储惩罚参数 - 与ieee33一致
        self.ev_penalty_soc_shortage = config.get("penalty_soc_shortage", 500)
        self.ev_penalty_interruption = config.get("penalty_interruption", 50)
        self.ev_interruption_threshold_kw = config.get("interruption_threshold_kw", 1.0)
    
    # ========== 公共接口 ==========
    
    def get_branch_params(self):
        """
        获取支路参数
        
        Returns:
            dict: 包含r_pu, x_pu, from_bus, to_bus的数组
        """
        return {
            "r_pu": self.r_pu.copy(),
            "x_pu": self.x_pu.copy(),
            "from_bus": self.from_bus.copy(),
            "to_bus": self.to_bus.copy(),
            "r_ohm": self.r_ohm.copy(),
            "x_ohm": self.x_ohm.copy(),
        }
    
    def get_bus_load(self):
        """
        获取节点负荷
        
        Returns:
            dict: 包含p_mw, q_mvar的数组
        """
        return {
            "p_mw": self.p_load_mw.copy(),
            "q_mvar": self.q_load_mvar.copy(),
            "p_kw": self.p_load_kw.copy(),
            "q_kvar": self.q_load_kvar.copy(),
            "p_pu": self.p_load_pu.copy(),
            "q_pu": self.q_load_pu.copy(),
        }
    
    def get_upstream_branch(self, bus):
        """获取节点的上游支路索引 - 与ieee33保持一致"""
        if bus == self.root_bus:
            return None
        parent = self.parent[bus]
        if parent < 0:
            return None
        return self.branch_idx.get((parent, bus))
    
    def get_parent_bus(self, bus):
        """获取节点的父节点"""
        if bus == self.root_bus:
            return None
        p = self.parent_bus[bus]
        return p if p >= 0 else None
    
    def get_downstream_buses(self, bus):
        """获取节点的下游所有节点"""
        downstream = []
        queue = [bus]
        visited = set([bus])
        
        while queue:
            current = queue.pop(0)
            for neighbor, _ in self.adjacency[current]:
                if neighbor not in visited and self.parent_bus[neighbor] == current:
                    downstream.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return downstream
    
    def get_downstream_branches(self, bus):
        """获取节点下游支路索引列表 - 与ieee33保持一致"""
        branches = []
        for child in self.children[bus]:
            branch_idx = self.branch_idx.get((bus, child))
            if branch_idx is not None:
                branches.append(branch_idx)
        return branches
    
    def get_tie_switch_params(self):
        """获取联络线参数"""
        if not self.tie_switches:
            return None
        
        n_ties = len(self.tie_switches)
        return {
            "from_bus": np.array([ts["from"] for ts in self.tie_switches]),
            "to_bus": np.array([ts["to"] for ts in self.tie_switches]),
            "r_pu": np.array([ts["r_pu"] for ts in self.tie_switches]),
            "x_pu": np.array([ts["x_pu"] for ts in self.tie_switches]),
            "n_ties": n_ties,
        }
    
    def get_ev_station_params(self):
        """获取EV充电站参数"""
        if not self.ev_stations:
            return None
        
        return {
            "buses": [es["bus"] for es in self.ev_stations],
            "n_vehicles": [es["n_vehicles"] for es in self.ev_stations],
            "max_charge_rate_kw": [es["max_charge_rate_kw"] for es in self.ev_stations],
            "max_station_power_kw": [es["max_station_power_kw"] for es in self.ev_stations],
            "n_stations": self.n_ev_stations,
        }
    
    def summary(self):
        """打印网络摘要"""
        print(f"\n{'='*50}")
        print(f"IEEE 123节点配电网络摘要")
        print(f"{'='*50}")
        print(f"节点数: {self.n_buses}")
        print(f"支路数: {self.n_branches}")
        print(f"基准电压: {self.v_base} kV")
        print(f"基准功率: {self.s_base} MVA")
        print(f"基准阻抗: {self.z_base:.4f} Ω")
        print(f"\n总负荷:")
        print(f"  有功: {self.total_p_load_mw:.3f} MW ({self.total_p_load_mw*1000:.1f} kW)")
        print(f"  无功: {self.total_q_load_mvar:.3f} MVar ({self.total_q_load_mvar*1000:.1f} kVar)")
        
        if self.n_tie_switches > 0:
            print(f"\n联络线数: {self.n_tie_switches}")
            for i, ts in enumerate(self.tie_switches):
                print(f"  联络线{i}: 节点{ts['from']} - 节点{ts['to']}")
        
        if self.n_ev_stations > 0:
            print(f"\nEV充电站数: {self.n_ev_stations}")
            for i, es in enumerate(self.ev_stations):
                print(f"  站{i}: 节点{es['bus']}, {es['n_vehicles']}辆车, "
                      f"最大功率{es['max_station_power_kw']:.0f}kW")
        
        print(f"{'='*50}\n")


def get_ieee123_network(config: dict = None, tie_switch_config: dict = None, 
                        ev_station_config: dict = None):
    """
    获取IEEE 123节点网络实例
    
    Args:
        config: 完整配置字典，包含tie_switch和ev_station设置
        tie_switch_config: 联络线配置（直接传入）
        ev_station_config: EV充电站配置（直接传入）
    
    Returns:
        IEEE123Network实例
    """
    # 如果直接传入了配置，优先使用
    if tie_switch_config is not None or ev_station_config is not None:
        return IEEE123Network(
            tie_switch_config=tie_switch_config,
            ev_station_config=ev_station_config
        )
    
    # 如果没有任何配置
    if config is None:
        return IEEE123Network()
    
    # 从完整配置中提取
    tie_config = config.get("devices", {}).get("tie_switches", None)
    ev_config = config.get("devices", {}).get("ev_stations", None)
    
    return IEEE123Network(
        tie_switch_config=tie_config,
        ev_station_config=ev_config
    )


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("测试IEEE 123节点网络...")
    
    # 测试基本网络
    network = IEEE123Network()
    network.summary()
    
    # 测试带EV充电站的网络
    ev_config = {
        "enabled": True,
        "buses": [47, 64, 75, 95, 113],  # 节点48, 65, 76, 96, 114
        "n_vehicles": [150, 120, 100, 80, 100],
        "max_charge_rate_kw": 7.0,
    }
    
    network_with_ev = IEEE123Network(ev_station_config=ev_config)
    network_with_ev.summary()
    
    # 检查拓扑
    branch_params = network.get_branch_params()
    print(f"支路阻抗范围:")
    print(f"  R: [{branch_params['r_pu'].min():.4f}, {branch_params['r_pu'].max():.4f}] pu")
    print(f"  X: [{branch_params['x_pu'].min():.4f}, {branch_params['x_pu'].max():.4f}] pu")
    
    # 检查负荷
    loads = network.get_bus_load()
    nonzero_loads = np.where(loads['p_kw'] > 0)[0]
    print(f"\n有负荷的节点数: {len(nonzero_loads)}")
    print(f"负荷节点示例: {nonzero_loads[:10].tolist()}...")
