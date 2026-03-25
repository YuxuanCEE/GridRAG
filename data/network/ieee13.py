# -*- coding: utf-8 -*-
"""
IEEE 13节点配电网数据
包含线路参数、节点负荷、网络拓扑等

参考文献：
[1] IEEE 13 Node Test Feeder, IEEE PES Distribution System Analysis Subcommittee
[2] Robust Deep Reinforcement Learning for Volt-VAR Optimization in Active 
    Distribution System Under Uncertainty, IEEE TSG, 2025.

节点编号映射（原始 -> 0-indexed）：
    650 -> 0  (变电站/slack bus)
    632 -> 1
    633 -> 2
    634 -> 3  (PV)
    645 -> 4
    646 -> 5
    671 -> 6
    680 -> 7  (PV)
    684 -> 8  (PV)
    611 -> 9
    652 -> 10
    675 -> 11
    692 -> 12 (EV充电站)

注意：IEEE 13节点是不平衡三相系统，这里简化为等效单相模型。
"""

import numpy as np


class IEEE13Network:
    """IEEE 13节点配电网（支持联络线和EV充电站扩展）"""
    
    # 节点名称映射（便于调试和可视化）
    NODE_NAMES = {
        0: "650", 1: "632", 2: "633", 3: "634",
        4: "645", 5: "646", 6: "671", 7: "680",
        8: "684", 9: "611", 10: "652", 11: "675", 12: "692"
    }
    
    # 反向映射：原始节点号 -> 索引
    NODE_INDEX = {
        650: 0, 632: 1, 633: 2, 634: 3,
        645: 4, 646: 5, 671: 6, 680: 7,
        684: 8, 611: 9, 652: 10, 675: 11, 692: 12
    }
    
    def __init__(self, tie_switch_config=None, ev_station_config=None):
        """
        初始化IEEE 13节点配电网
        
        Args:
            tie_switch_config: 联络线配置字典，如果为None则使用默认辐射状网络
            ev_station_config: EV充电站配置字典
        """
        self.n_buses = 13
        self.n_branches = 12  # 固定支路数
        self.root_bus = 0
        
        # 基准值（IEEE 13标准）
        self.v_base = 4.16   # kV (主馈线电压等级)
        self.s_base = 5.0    # MVA
        self.z_base = self.v_base ** 2 / self.s_base  # Ω = 3.46 Ω
        
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
        
        IEEE 13节点线路参数（三相等效）：
        - 线路类型601: 主干线, 3相, R=0.3465 Ω/mi, X=1.0179 Ω/mi
        - 线路类型602: 支线, 3相
        - 线路类型603: 2相线路
        - 线路类型604: 单相线路
        - 线路类型605: 单相线路
        - 线路类型606: 3相线路
        - 线路类型607: 单相线路
        
        注意：阻抗值已按三相等效进行简化处理。
        """
        # 支路数据 [from_bus, to_bus, r(Ω), x(Ω)]
        # 基于IEEE 13 Node Test Feeder数据，转换为等效阻抗
        # 1 mile = 5280 ft, 阻抗按长度线性计算
        
        self.branch_data = np.array([
            # from, to, R(Ω), X(Ω)
            # Branch 0: 650-632 (调压器，近似为小阻抗)
            [0, 1, 0.0010, 0.0010],
            
            # Branch 1: 632-633 (Line 601, 500 ft = 0.0947 mi)
            [1, 2, 0.0328, 0.0964],
            
            # Branch 2: 633-634 (变压器XFM-1，简化为等效阻抗)
            [2, 3, 0.0050, 0.0200],
            
            # Branch 3: 632-645 (Line 603, 500 ft = 0.0947 mi, 2相)
            [1, 4, 0.1251, 0.1293],
            
            # Branch 4: 645-646 (Line 603, 300 ft = 0.0568 mi)
            [4, 5, 0.0751, 0.0776],
            
            # Branch 5: 632-671 (Line 601, 2000 ft = 0.3788 mi)
            [1, 6, 0.1312, 0.3856],
            
            # Branch 6: 671-680 (Line 601, 1000 ft = 0.1894 mi)
            [6, 7, 0.0656, 0.1928],
            
            # Branch 7: 671-684 (Line 604, 300 ft = 0.0568 mi, 单相)
            [6, 8, 0.0755, 0.0770],
            
            # Branch 8: 684-611 (Line 605, 300 ft = 0.0568 mi, 单相)
            [8, 9, 0.0755, 0.0770],
            
            # Branch 9: 684-652 (Line 607, 800 ft = 0.1515 mi, 单相)
            [8, 10, 0.2012, 0.0785],
            
            # Branch 10: 671-675 (Line 606, 500 ft = 0.0947 mi)
            [6, 11, 0.0551, 0.0338],
            
            # Branch 11: 671-692 (开关，闭合状态，近似为小阻抗)
            [6, 12, 0.0010, 0.0010],
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
        
        IEEE 13节点负荷数据（三相总和简化）
        格式: [bus, P_load(kW), Q_load(kVar)]
        
        原始数据来源：IEEE 13 Node Test Feeder - Spot Load Data
        """
        # 负荷数据 [bus, P(kW), Q(kVar)]
        # 注意：IEEE 13原始数据是分相的，这里取三相总和
        load_kw_kvar = np.array([
            [0, 0, 0],         # 节点0 (650): 变电站，无负荷
            [1, 0, 0],         # 节点1 (632): 母线节点
            [2, 0, 0],         # 节点2 (633): 母线节点
            [3, 400, 290],     # 节点3 (634): 三相负荷 160+120+120 kW (Wye-connected)
            [4, 170, 125],     # 节点4 (645): 相B负荷
            [5, 230, 132],     # 节点5 (646): 相B负荷 (Delta-connected)
            [6, 1155, 660],    # 节点6 (671): 大负荷节点 385×3 kW
            [7, 0, 0],         # 节点7 (680): 母线节点（DER接入点）
            [8, 0, 0],         # 节点8 (684): 母线节点（DER接入点）
            [9, 170, 80],      # 节点9 (611): 相C负荷
            [10, 128, 86],     # 节点10 (652): 相A负荷
            [11, 843, 462],    # 节点11 (675): 三相负荷 485+68+290 kW
            [12, 170, 151],    # 节点12 (692): 相C负荷 (Delta-connected)
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
        
        IEEE 13节点原始系统无联络线，但可以添加用于网络重构研究
        
        Args:
            config: 联络线配置字典
        """
        tie_data = config.get("branches", [])
        self.n_tie_switches = len(tie_data)
        
        for idx, tie in enumerate(tie_data):
            tie_info = {
                "id": tie.get("id", 13 + idx),
                "idx": idx,
                "from": tie["from"],
                "to": tie["to"],
                "r_ohm": tie["r_ohm"],
                "x_ohm": tie["x_ohm"],
                "r_pu": tie["r_ohm"] / self.z_base,
                "x_pu": tie["x_ohm"] / self.z_base,
            }
            self.tie_switches.append(tie_info)
        
        self.tie_initial_status = config.get("initial_status", [0] * self.n_tie_switches)
        
        if self.n_tie_switches > 0:
            print(f"  已加载 {self.n_tie_switches} 条联络线:")
            for tie in self.tie_switches:
                print(f"    支路{tie['id']}: 节点{self.NODE_NAMES[tie['from']]} <-> "
                      f"节点{self.NODE_NAMES[tie['to']]}, "
                      f"R={tie['r_pu']:.6f}pu, X={tie['x_pu']:.6f}pu")
    
    def _init_ev_stations(self, config):
        """
        初始化EV充电站数据
        
        Args:
            config: EV充电站配置字典
        """
        buses = config.get("buses", [])
        n_vehicles = config.get("n_vehicles", [50] * len(buses))
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
                "capacity_mwh": n_ev * battery_kwh / 1000,
                "max_power_mw": n_ev * max_charge_rate_kw / 1000,
                "battery_kwh": battery_kwh,
                "max_charge_rate_kw": max_charge_rate_kw,
                "charge_efficiency": charge_efficiency,
                "target_soc": target_soc,
                "arrival_soc_mean": 0.3,
                "arrival_energy_mwh": n_ev * battery_kwh * 0.3 / 1000,
                "target_energy_mwh": n_ev * battery_kwh * target_soc / 1000,
            }
            self.ev_stations.append(station)
        
        self.ev_penalty_soc_shortage = config.get("penalty_soc_shortage", 500)
        self.ev_penalty_interruption = config.get("penalty_interruption", 50)
        self.ev_interruption_threshold_kw = config.get("interruption_threshold_kw", 1.0)
        
        if self.n_ev_stations > 0:
            print(f"  已加载 {self.n_ev_stations} 个EV充电站:")
            for station in self.ev_stations:
                print(f"    充电站{station['id']}: 节点{self.NODE_NAMES[station['bus']]}, "
                      f"{station['n_vehicles']}辆EV, "
                      f"总容量{station['capacity_mwh']:.2f}MWh, "
                      f"最大功率{station['max_power_mw']:.2f}MW")
    
    def get_node_name(self, bus_idx):
        """获取节点原始名称"""
        return self.NODE_NAMES.get(bus_idx, str(bus_idx))
    
    def get_node_index(self, node_name):
        """根据原始节点号获取索引"""
        if isinstance(node_name, int):
            return self.NODE_INDEX.get(node_name, node_name)
        return int(node_name)
    
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
            p_factor: 有功负荷因子
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
            dict: {"fixed": 固定支路列表, "tie": 联络线列表, "total": 总支路数}
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
        print("IEEE 13节点配电网")
        print("=" * 50)
        print(f"节点数: {self.n_buses}")
        print(f"固定支路数: {self.n_branches}")
        print(f"联络线数: {self.n_tie_switches}")
        print(f"EV充电站数: {self.n_ev_stations}")
        print(f"基准电压: {self.v_base} kV")
        print(f"基准功率: {self.s_base} MVA")
        print(f"总有功负荷: {self.total_p_load_mw:.3f} MW")
        print(f"总无功负荷: {self.total_q_load_mvar:.3f} MVar")
        
        print("\n节点映射 (index -> 原始编号):")
        for idx, name in self.NODE_NAMES.items():
            load_info = f"P={self.p_load_kw[idx]:.0f}kW" if self.p_load_kw[idx] > 0 else "无负荷"
            print(f"  {idx} -> {name}: {load_info}")
        
        if self.n_tie_switches > 0:
            print("\n联络线信息:")
            for tie in self.tie_switches:
                print(f"  支路{tie['id']}: 节点{self.NODE_NAMES[tie['from']]} <-> "
                      f"节点{self.NODE_NAMES[tie['to']]}")
        
        if self.n_ev_stations > 0:
            print("\nEV充电站信息:")
            for station in self.ev_stations:
                print(f"  充电站{station['id']}: 节点{self.NODE_NAMES[station['bus']]}, "
                      f"{station['n_vehicles']}辆EV, "
                      f"容量{station['capacity_mwh']:.2f}MWh")
        
        print("=" * 50)


def get_ieee13_network(tie_switch_config=None, ev_station_config=None):
    """
    获取IEEE13网络实例
    
    Args:
        tie_switch_config: 联络线配置字典
        ev_station_config: EV充电站配置字典
    
    Returns:
        IEEE13Network实例
    """
    return IEEE13Network(tie_switch_config=tie_switch_config,
                         ev_station_config=ev_station_config)


if __name__ == "__main__":
    # 测试基本网络
    print("=== 测试基本辐射状网络 ===")
    net = get_ieee13_network()
    net.summary()
    
    # 打印支路参数
    print("\n支路参数:")
    for i in range(net.n_branches):
        params = net.get_branch_params(i)
        from_name = net.NODE_NAMES[params['from_bus']]
        to_name = net.NODE_NAMES[params['to_bus']]
        print(f"  支路{i}: {from_name}->{to_name}, "
              f"r={params['r_pu']:.6f} pu, x={params['x_pu']:.6f} pu")
    
    # 测试带EV充电站的网络
    print("\n\n=== 测试带EV充电站的网络 ===")
    ev_config = {
        "enabled": True,
        "buses": [12],  # 节点692
        "n_vehicles": [50],
        "battery_kwh": 50,
        "max_charge_rate_kw": 7.0,
    }
    net_with_ev = get_ieee13_network(ev_station_config=ev_config)
    net_with_ev.summary()
