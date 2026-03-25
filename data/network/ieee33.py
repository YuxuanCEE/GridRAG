# -*- coding: utf-8 -*-
"""
IEEE 33节点配电网数据
包含线路参数、节点负荷、网络拓扑等

参考文献：
[1] Baran M E, Wu F F. Network reconfiguration in distribution systems for loss 
    reduction and load balancing[J]. IEEE Trans on Power Delivery, 1989, 4(2): 1401-1407.
[2] Enhanced IEEE 33 Bus Benchmark Test System for Distribution System Studies, 
    IEEE Trans on Power Systems, 2021.
"""

import numpy as np


class IEEE33Network:
    """IEEE 33节点配电网（支持联络线扩展）"""
    
    def __init__(self, tie_switch_config=None, ev_station_config=None):
        """
        初始化IEEE 33节点配电网
        
        Args:
            tie_switch_config: 联络线配置字典，如果为None则使用默认辐射状网络
            ev_station_config: EV充电站配置字典
        """
        self.n_buses = 33
        self.n_branches = 32  # 固定支路数
        self.root_bus = 0
        
        # 基准值
        self.v_base = 12.66  # kV
        self.s_base = 1.0    # MVA (可根据需要调整)
        self.z_base = self.v_base ** 2 / self.s_base  # Ω
        
        # 初始化网络数据
        self._init_branch_data()
        self._init_load_data()
        self._build_topology()
        
        # 联络线数据（用于网络重构）
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
        格式: [from_bus, to_bus, r(Ω), x(Ω)]
        注意: 节点编号从0开始 (0-32)
        """
        # 原始数据 (节点编号1-33，这里转换为0-32)
        self.branch_data = np.array([
            [0, 1, 0.0922, 0.0470],
            [1, 2, 0.4930, 0.2511],
            [2, 3, 0.3660, 0.1864],
            [3, 4, 0.3811, 0.1941],
            [4, 5, 0.8190, 0.7070],
            [5, 6, 0.1872, 0.6188],
            [6, 7, 0.7114, 0.2351],
            [7, 8, 1.0300, 0.7400],
            [8, 9, 1.0440, 0.7400],
            [9, 10, 0.1966, 0.0650],
            [10, 11, 0.3744, 0.1238],
            [11, 12, 1.4680, 1.1550],
            [12, 13, 0.5416, 0.7129],
            [13, 14, 0.5910, 0.5260],
            [14, 15, 0.7463, 0.5450],
            [15, 16, 1.2890, 1.7210],
            [16, 17, 0.7320, 0.5740],
            [1, 18, 0.1640, 0.1565],
            [18, 19, 1.5042, 1.3554],
            [19, 20, 0.4095, 0.4784],
            [20, 21, 0.7089, 0.9373],
            [2, 22, 0.4512, 0.3083],
            [22, 23, 0.8980, 0.7091],
            [23, 24, 0.8960, 0.7011],
            [5, 25, 0.2030, 0.1034],
            [25, 26, 0.2842, 0.1447],
            [26, 27, 1.0590, 0.9337],
            [27, 28, 0.8042, 0.7006],
            [28, 29, 0.5075, 0.2585],
            [29, 30, 0.9744, 0.9630],
            [30, 31, 0.3105, 0.3619],
            [31, 32, 0.3410, 0.5302],
        ], dtype=np.float64)
        
        # 提取支路信息
        self.from_bus = self.branch_data[:, 0].astype(int)
        self.to_bus = self.branch_data[:, 1].astype(int)
        self.r_ohm = self.branch_data[:, 2]  # 电阻 Ω
        self.x_ohm = self.branch_data[:, 3]  # 电抗 Ω
        
        # 转换为标幺值
        self.r_pu = self.r_ohm / self.z_base
        self.x_pu = self.x_ohm / self.z_base
    
    def _init_load_data(self):
        """
        初始化节点负荷数据
        格式: [bus, P_load(kW), Q_load(kVar)]
        注意: 节点0为根节点，无负荷
        """
        # 原始负荷数据 (kW, kVar)
        load_kw_kvar = np.array([
            [0, 0, 0],        # 节点0（根节点）
            [1, 100, 60],
            [2, 90, 40],
            [3, 120, 80],
            [4, 60, 30],
            [5, 60, 20],
            [6, 200, 100],
            [7, 200, 100],
            [8, 60, 20],
            [9, 60, 20],
            [10, 45, 30],
            [11, 60, 35],
            [12, 60, 35],
            [13, 120, 80],
            [14, 60, 10],
            [15, 60, 20],
            [16, 60, 20],
            [17, 90, 40],
            [18, 90, 40],
            [19, 90, 40],
            [20, 90, 40],
            [21, 90, 40],
            [22, 90, 50],
            [23, 420, 200],
            [24, 420, 200],
            [25, 60, 25],
            [26, 60, 25],
            [27, 60, 20],
            [28, 120, 70],
            [29, 200, 600],
            [30, 150, 70],
            [31, 210, 100],
            [32, 60, 40],
        ], dtype=np.float64)
        
        self.bus_ids = load_kw_kvar[:, 0].astype(int)
        self.p_load_kw = load_kw_kvar[:, 1]
        self.q_load_kvar = load_kw_kvar[:, 2]
        
        # 转换为MW和MVar
        self.p_load_mw = self.p_load_kw / 1000
        self.q_load_mvar = self.q_load_kvar / 1000
        
        # 转换为标幺值
        self.p_load_pu = self.p_load_mw / self.s_base
        self.q_load_pu = self.q_load_mvar / self.s_base
        
        # 总负荷
        self.total_p_load_mw = np.sum(self.p_load_mw)
        self.total_q_load_mvar = np.sum(self.q_load_mvar)
    
    def _build_topology(self):
        """构建网络拓扑"""
        # 邻接表
        self.adj_list = {i: [] for i in range(self.n_buses)}
        for i in range(self.n_branches):
            f, t = int(self.from_bus[i]), int(self.to_bus[i])
            self.adj_list[f].append(t)
            self.adj_list[t].append(f)
        
        # 父节点和子节点（用于辐射状网络）
        self.parent = [-1] * self.n_buses
        self.children = {i: [] for i in range(self.n_buses)}
        
        # BFS构建树结构
        visited = [False] * self.n_buses
        queue = [self.root_bus]
        visited[self.root_bus] = True
        
        while queue:
            node = queue.pop(0)
            for neighbor in self.adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    self.parent[neighbor] = node
                    self.children[node].append(neighbor)
                    queue.append(neighbor)
        
        # 构建支路索引映射 (from_bus, to_bus) -> branch_idx
        self.branch_idx = {}
        for i in range(self.n_branches):
            f, t = int(self.from_bus[i]), int(self.to_bus[i])
            self.branch_idx[(f, t)] = i
            self.branch_idx[(t, f)] = i
    
    def _init_tie_switches(self, config):
        """
        初始化联络线数据
        
        参考Enhanced IEEE 33 Bus Benchmark Test System论文:
        - 支路33: 节点9 (8) <-> 节点22 (21)
        - 支路34: 节点10 (9) <-> 节点16 (15)  
        - 支路35: 节点13 (12) <-> 节点23 (22)
        
        Args:
            config: 联络线配置字典
        """
        tie_data = config.get("branches", [])
        self.n_tie_switches = len(tie_data)
        
        for idx, tie in enumerate(tie_data):
            tie_info = {
                "id": tie.get("id", 33 + idx),
                "idx": idx,  # 在联络线列表中的索引
                "from": tie["from"],
                "to": tie["to"],
                "r_ohm": tie["r_ohm"],
                "x_ohm": tie["x_ohm"],
                "r_pu": tie["r_ohm"] / self.z_base,
                "x_pu": tie["x_ohm"] / self.z_base,
            }
            self.tie_switches.append(tie_info)
        
        # 存储初始状态
        self.tie_initial_status = config.get("initial_status", [0] * self.n_tie_switches)
        
        print(f"  已加载 {self.n_tie_switches} 条联络线:")
        for tie in self.tie_switches:
            print(f"    支路{tie['id']}: 节点{tie['from']+1} <-> 节点{tie['to']+1}, "
                  f"R={tie['r_pu']:.6f}pu, X={tie['x_pu']:.6f}pu")
    
    def _init_ev_stations(self, config):
        """
        初始化EV充电站数据
        
        参考文献:
        - 祁向龙《多时间尺度协同的配电网分层深度强化学习电压控制策略》
        - emobpy (Scientific Data): An open tool for creating BEV time series
        
        Args:
            config: EV充电站配置字典
        """
        buses = config.get("buses", [])
        n_vehicles = config.get("n_vehicles", [100] * len(buses))
        battery_kwh = config.get("battery_kwh", 50)
        max_charge_rate_kw = config.get("max_charge_rate_kw", 7.0)
        charge_efficiency = config.get("charge_efficiency", 0.95)
        target_soc = config.get("target_soc", 0.9)
        
        self.n_ev_stations = len(buses)
        
        for idx, bus in enumerate(buses):
            n_ev = n_vehicles[idx] if idx < len(n_vehicles) else n_vehicles[-1]
            
            station = {
                "id": idx,
                "bus": bus,
                "n_vehicles": n_ev,
                # 聚合容量（整站）
                "capacity_mwh": n_ev * battery_kwh / 1000,  # MW·h
                "max_power_mw": n_ev * max_charge_rate_kw / 1000,  # MW
                # 单车参数
                "battery_kwh": battery_kwh,
                "max_charge_rate_kw": max_charge_rate_kw,
                "charge_efficiency": charge_efficiency,
                "target_soc": target_soc,
                # 能量需求（假设平均到达SOC为0.3）
                "arrival_soc_mean": 0.3,
                "arrival_energy_mwh": n_ev * battery_kwh * 0.3 / 1000,
                "target_energy_mwh": n_ev * battery_kwh * target_soc / 1000,
            }
            self.ev_stations.append(station)
        
        # 存储惩罚参数
        self.ev_penalty_soc_shortage = config.get("penalty_soc_shortage", 500)
        self.ev_penalty_interruption = config.get("penalty_interruption", 50)
        self.ev_interruption_threshold_kw = config.get("interruption_threshold_kw", 1.0)
        
        print(f"  已加载 {self.n_ev_stations} 个EV充电站:")
        for station in self.ev_stations:
            print(f"    充电站{station['id']}: 节点{station['bus']+1}, "
                  f"{station['n_vehicles']}辆EV, "
                  f"总容量{station['capacity_mwh']:.2f}MWh, "
                  f"最大功率{station['max_power_mw']:.2f}MW")
    
    def get_upstream_branch(self, bus):
        """获取节点上游支路索引"""
        if bus == self.root_bus:
            return None
        parent = self.parent[bus]
        return self.branch_idx.get((parent, bus))
    
    def get_downstream_branches(self, bus):
        """获取节点下游支路索引列表"""
        branches = []
        for child in self.children[bus]:
            branch_idx = self.branch_idx.get((bus, child))
            if branch_idx is not None:
                branches.append(branch_idx)
        return branches
    
    def get_branch_params(self, branch_idx):
        """获取支路参数"""
        return {
            "from_bus": int(self.from_bus[branch_idx]),
            "to_bus": int(self.to_bus[branch_idx]),
            "r_pu": self.r_pu[branch_idx],
            "x_pu": self.x_pu[branch_idx],
            "r_ohm": self.r_ohm[branch_idx],
            "x_ohm": self.x_ohm[branch_idx],
        }
    
    def get_tie_switch_params(self, tie_idx):
        """获取联络线参数"""
        if tie_idx < 0 or tie_idx >= self.n_tie_switches:
            return None
        return self.tie_switches[tie_idx]
    
    def get_bus_load(self, bus, p_factor=1.0, q_factor=1.0):
        """
        获取节点负荷
        
        Args:
            bus: 节点编号
            p_factor: 有功负荷因子（用于时变负荷）
            q_factor: 无功负荷因子
        
        Returns:
            (p_load_pu, q_load_pu)
        """
        return (
            self.p_load_pu[bus] * p_factor,
            self.q_load_pu[bus] * q_factor
        )
    
    def get_all_branches_for_reconfiguration(self):
        """
        获取用于网络重构的所有支路信息
        
        Returns:
            dict: {
                "fixed": 固定支路列表（32条）,
                "tie": 联络线列表,
                "total": 总支路数
            }
        """
        fixed_branches = []
        for i in range(self.n_branches):
            fixed_branches.append({
                "idx": i,
                "from": int(self.from_bus[i]),
                "to": int(self.to_bus[i]),
                "r_pu": self.r_pu[i],
                "x_pu": self.x_pu[i],
                "is_tie": False,
            })
        
        tie_branches = []
        for tie in self.tie_switches:
            tie_branches.append({
                "idx": tie["idx"],
                "from": tie["from"],
                "to": tie["to"],
                "r_pu": tie["r_pu"],
                "x_pu": tie["x_pu"],
                "is_tie": True,
            })
        
        return {
            "fixed": fixed_branches,
            "tie": tie_branches,
            "total": self.n_branches + self.n_tie_switches,
        }
    
    def summary(self):
        """打印网络摘要"""
        print("=" * 50)
        print("IEEE 33节点配电网")
        print("=" * 50)
        print(f"节点数: {self.n_buses}")
        print(f"固定支路数: {self.n_branches}")
        print(f"联络线数: {self.n_tie_switches}")
        print(f"EV充电站数: {self.n_ev_stations}")
        print(f"基准电压: {self.v_base} kV")
        print(f"基准功率: {self.s_base} MVA")
        print(f"总有功负荷: {self.total_p_load_mw:.3f} MW")
        print(f"总无功负荷: {self.total_q_load_mvar:.3f} MVar")
        
        if self.n_tie_switches > 0:
            print("\n联络线信息:")
            for tie in self.tie_switches:
                print(f"  支路{tie['id']}: 节点{tie['from']+1} <-> 节点{tie['to']+1}")
        
        if self.n_ev_stations > 0:
            print("\nEV充电站信息:")
            for station in self.ev_stations:
                print(f"  充电站{station['id']}: 节点{station['bus']+1}, "
                      f"{station['n_vehicles']}辆EV, "
                      f"容量{station['capacity_mwh']:.2f}MWh")
        
        print("=" * 50)


# 全局网络实例（方便导入使用）
def get_ieee33_network(tie_switch_config=None, ev_station_config=None):
    """
    获取IEEE33网络实例
    
    Args:
        tie_switch_config: 联络线配置字典，如果为None则使用默认辐射状网络
        ev_station_config: EV充电站配置字典
    
    Returns:
        IEEE33Network实例
    """
    return IEEE33Network(tie_switch_config=tie_switch_config, 
                         ev_station_config=ev_station_config)


if __name__ == "__main__":
    # 测试基本网络
    print("=== 测试基本辐射状网络 ===")
    net = get_ieee33_network()
    net.summary()
    
    # 打印部分支路参数
    print("\n前5条支路参数:")
    for i in range(5):
        params = net.get_branch_params(i)
        print(f"  支路{i}: {params['from_bus']}->{params['to_bus']}, "
              f"r={params['r_pu']:.6f} pu, x={params['x_pu']:.6f} pu")
    
    # 打印拓扑结构
    print(f"\n节点5的子节点: {net.children[5]}")
    print(f"节点10的父节点: {net.parent[10]}")
    print(f"节点10的上游支路: {net.get_upstream_branch(10)}")
    
    # 测试带联络线的网络
    print("\n\n=== 测试带联络线的网络 ===")
    tie_config = {
        "enabled": True,
        "branches": [
            {"id": 33, "from": 8, "to": 21, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 34, "from": 9, "to": 15, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 35, "from": 12, "to": 22, "r_ohm": 2.0, "x_ohm": 2.0},
        ],
        "initial_status": [0, 0, 0],
    }
    net_with_ties = get_ieee33_network(tie_switch_config=tie_config)
    net_with_ties.summary()
    
    # 获取重构用支路信息
    all_branches = net_with_ties.get_all_branches_for_reconfiguration()
    print(f"\n总支路数（含联络线）: {all_branches['total']}")
    print(f"固定支路: {len(all_branches['fixed'])}条")
    print(f"联络线: {len(all_branches['tie'])}条")
