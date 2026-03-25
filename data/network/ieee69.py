# -*- coding: utf-8 -*-
"""
IEEE 69节点配电网数据
包含线路参数、节点负荷、网络拓扑等

参考文献：
[1] Baran M E, Wu F F. Network reconfiguration in distribution systems for loss 
    reduction and load balancing[J]. IEEE Trans on Power Delivery, 1989, 4(2): 1401-1407.
[2] Savier J S, Das D. Impact of network reconfiguration on loss allocation of radial 
    distribution systems[J]. IEEE Trans on Power Delivery, 2007, 22(4): 2473-2480.

DER配置方案（基于拓扑结构和负荷分布）：
    PV节点: 27, 50, 62 (0-indexed: 26, 49, 61)
    WT节点: 18, 65 (0-indexed: 17, 64)
    SC节点: 12, 50, 62 (0-indexed: 11, 49, 61)
    SVC节点: 18, 51 (0-indexed: 17, 50)
    ESS节点: 12, 50 (0-indexed: 11, 49)
    EV充电站节点: 28, 51, 66 (0-indexed: 27, 50, 65)

基准值：12.66 kV, 1.0 MVA
总负荷：约 3.80 MW + 2.69 MVar
"""

import numpy as np


class IEEE69Network:
    """IEEE 69节点配电网（支持联络线和EV充电站扩展）"""
    
    def __init__(self, tie_switch_config=None, ev_station_config=None):
        """
        初始化IEEE 69节点配电网
        
        Args:
            tie_switch_config: 联络线配置字典，如果为None则使用默认辐射状网络
            ev_station_config: EV充电站配置字典
        """
        self.n_buses = 69
        self.n_branches = 68  # 固定支路数
        self.root_bus = 0
        
        # 基准值
        self.v_base = 12.66  # kV
        self.s_base = 1.0    # MVA
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
        
        IEEE 69节点系统支路参数（来自Baran & Wu 1989）
        格式: [from_bus, to_bus, r(Ω), x(Ω)]
        注意: 节点编号从0开始 (0-68)，节点0为变电站
        """
        # IEEE 69节点系统完整支路数据
        # 数据来源: Baran & Wu 1989, Savier & Das 2007
        self.branch_data = np.array([
            # 主馈线1: 0->1->...->27
            [0, 1, 0.0005, 0.0012],
            [1, 2, 0.0005, 0.0012],
            [2, 3, 0.0015, 0.0036],
            [3, 4, 0.0251, 0.0294],
            [4, 5, 0.3660, 0.1864],
            [5, 6, 0.3811, 0.1941],
            [6, 7, 0.0922, 0.0470],
            [7, 8, 0.0493, 0.0251],
            [8, 9, 0.8190, 0.2707],
            [9, 10, 0.1872, 0.0619],
            [10, 11, 0.7114, 0.2351],
            [11, 12, 1.0300, 0.3400],
            [12, 13, 1.0440, 0.3450],
            [13, 14, 1.0580, 0.3496],
            [14, 15, 0.1966, 0.0650],
            [15, 16, 0.3744, 0.1238],
            [16, 17, 0.0047, 0.0016],
            [17, 18, 0.3276, 0.1083],
            [18, 19, 0.2106, 0.0690],
            [19, 20, 0.3416, 0.1129],
            [20, 21, 0.0140, 0.0046],
            [21, 22, 0.1591, 0.0526],
            [22, 23, 0.3463, 0.1145],
            [23, 24, 0.7488, 0.2475],
            [24, 25, 0.3089, 0.1021],
            [25, 26, 0.1732, 0.0572],
            [26, 27, 0.0044, 0.0108],
            # 分支1: 3->28->...->35
            [3, 28, 0.0640, 0.1565],
            [28, 29, 0.3978, 0.1315],
            [29, 30, 0.0702, 0.0232],
            [30, 31, 0.3510, 0.1160],
            [31, 32, 0.8390, 0.2816],
            [32, 33, 1.7080, 0.5646],
            [33, 34, 1.4740, 0.4873],
            [34, 35, 0.5490, 0.1816],
            # 分支2: 4->36->...->46
            [4, 36, 0.5460, 0.1802],
            [36, 37, 0.5460, 0.1802],
            [37, 38, 0.5460, 0.1802],
            [38, 39, 0.5460, 0.1802],
            [39, 40, 0.5460, 0.1802],
            [40, 41, 0.1246, 0.0412],
            [41, 42, 0.0298, 0.0098],
            [42, 43, 0.0298, 0.0098],
            [43, 44, 0.0298, 0.0098],
            [44, 45, 0.0298, 0.0098],
            [45, 46, 0.0298, 0.0098],
            # 分支3: 7->47->...->50
            [7, 47, 0.0910, 0.0470],
            [47, 48, 0.1089, 0.0528],
            [48, 49, 0.0009, 0.0012],
            [49, 50, 0.0034, 0.0084],
            # 分支4: 8->51->...->52
            [8, 51, 0.0034, 0.0084],
            [51, 52, 0.0851, 0.2083],
            # 分支5: 11->53->...->64
            [11, 53, 0.2898, 0.0709],
            [53, 54, 0.0822, 0.2011],
            [54, 55, 0.0928, 0.0473],
            [55, 56, 0.3319, 0.1114],
            [56, 57, 0.1740, 0.0886],
            [57, 58, 0.2030, 0.1034],
            [58, 59, 0.2842, 0.1447],
            [59, 60, 0.2813, 0.1433],
            [60, 61, 0.5900, 0.0526],
            [61, 62, 0.5549, 0.0546],
            [62, 63, 0.5598, 0.0441],
            [63, 64, 0.5573, 0.0430],
            # 分支6: 12->65->...->68 (原文献为64->65，这里修正为12->65以保持树结构)
            [12, 65, 0.0898, 0.0709],
            [65, 66, 0.5028, 0.4388],
            [66, 67, 0.5478, 0.4467],
            [67, 68, 0.0898, 0.0709],
        ], dtype=np.float64)
        
        # 提取支路信息
        self.from_bus = self.branch_data[:, 0].astype(int)
        self.to_bus = self.branch_data[:, 1].astype(int)
        self.r_ohm_raw = self.branch_data[:, 2].copy()  # 保存原始值
        self.x_ohm_raw = self.branch_data[:, 3].copy()
        
        # ====== 阻抗缩放处理 ======
        # IEEE 69系统的阻抗数据来自多个文献，可能存在量纲差异
        # 为保证数值稳定性和电压在合理范围内，添加缩放因子
        # 参考：IEEE 123系统使用了impedance_scale = 0.05
        impedance_scale = 0.20  # 缩放到原来的12%
        
        self.r_ohm = self.branch_data[:, 2] * impedance_scale
        self.x_ohm = self.branch_data[:, 3] * impedance_scale
        
        # 转换为标幺值
        self.r_pu = self.r_ohm / self.z_base
        self.x_pu = self.x_ohm / self.z_base
        
        # ====== 阻抗下限处理 ======
        # 设置最小阻抗下限，避免数值问题
        r_min_pu = 0.0001  # 最小电阻标幺值
        x_min_pu = 0.0001  # 最小电抗标幺值
        
        # 对过小的阻抗进行下限截断
        self.r_pu = np.maximum(self.r_pu, r_min_pu)
        self.x_pu = np.maximum(self.x_pu, x_min_pu)
        
        # 同步更新欧姆值（保持一致性）
        self.r_ohm = self.r_pu * self.z_base
        self.x_ohm = self.x_pu * self.z_base
    
    def _init_load_data(self):
        """
        初始化节点负荷数据
        
        IEEE 69节点系统节点负荷（来自Baran & Wu 1989）
        格式: [bus, P_load(kW), Q_load(kVar)]
        注意: 节点0为根节点（变电站），无负荷
        """
        # 完整的69节点负荷数据
        load_kw_kvar = np.array([
            [0, 0, 0],         # 节点0（根节点/变电站）
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [5, 2.6, 2.2],
            [6, 40.4, 30],
            [7, 75, 54],
            [8, 30, 22],
            [9, 28, 19],
            [10, 145, 104],
            [11, 145, 104],
            [12, 8, 5.5],
            [13, 8, 5.5],
            [14, 0, 0],
            [15, 45.5, 30],
            [16, 60, 35],
            [17, 60, 35],
            [18, 0, 0],
            [19, 1, 0.6],
            [20, 114, 81],
            [21, 5.3, 3.5],
            [22, 0, 0],
            [23, 28, 20],
            [24, 0, 0],
            [25, 14, 10],
            [26, 14, 10],
            [27, 26, 18.6],
            [28, 26, 18.6],
            [29, 0, 0],
            [30, 0, 0],
            [31, 0, 0],
            [32, 0, 0],
            [33, 14, 10],
            [34, 19.5, 14],
            [35, 6, 4],
            [36, 26, 18.55],
            [37, 26, 18.55],
            [38, 0, 0],
            [39, 24, 17],
            [40, 24, 17],
            [41, 1.2, 1],
            [42, 0, 0],
            [43, 6, 4.3],
            [44, 0, 0],
            [45, 39.22, 26.3],
            [46, 39.22, 26.3],
            [47, 0, 0],
            [48, 79, 56.4],
            [49, 384.7, 274.5],
            [50, 384.7, 274.5],
            [51, 40.5, 28.3],
            [52, 3.6, 2.7],
            [53, 4.35, 3.5],
            [54, 26.4, 19],
            [55, 24, 17.2],
            [56, 0, 0],
            [57, 0, 0],
            [58, 0, 0],
            [59, 100, 72],
            [60, 0, 0],
            [61, 1244, 888],
            [62, 32, 23],
            [63, 0, 0],
            [64, 227, 162],
            [65, 59, 42],
            [66, 18, 13],
            [67, 18, 13],
            [68, 28, 20],
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
        
        # 邻接表（带支路索引，与ieee123保持一致）
        self.adjacency = {i: [] for i in range(self.n_buses)}
        for i in range(self.n_branches):
            f, t = int(self.from_bus[i]), int(self.to_bus[i])
            self.adjacency[f].append((t, i))
            self.adjacency[t].append((f, i))
        
        # 父节点和子节点（用于辐射状网络）
        self.parent = [-1] * self.n_buses
        self.parent_bus = np.full(self.n_buses, -1, dtype=int)  # 与ieee123保持一致
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
                    self.parent_bus[neighbor] = node
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
        
        IEEE 69节点标准联络线（来自Baran & Wu 1989）:
        - 联络线1: 节点11 - 节点43 (tie switch 69)
        - 联络线2: 节点13 - 节点21 (tie switch 70)
        - 联络线3: 节点15 - 节点46 (tie switch 71)
        - 联络线4: 节点27 - 节点65 (tie switch 72)
        - 联络线5: 节点50 - 节点59 (tie switch 73)
        
        Args:
            config: 联络线配置字典
        """
        tie_data = config.get("branches", [])
        self.n_tie_switches = len(tie_data)
        
        for idx, tie in enumerate(tie_data):
            tie_info = {
                "id": tie.get("id", 69 + idx),
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
        
        Args:
            config: EV充电站配置字典
        """
        buses = config.get("buses", [])
        n_vehicles = config.get("n_vehicles", [100] * len(buses))
        battery_kwh = config.get("battery_kwh", 50)
        max_charge = config.get("max_charge_rate_kw", 7.0)
        charge_efficiency = config.get("charge_efficiency", 0.95)
        target_soc = config.get("target_soc", 0.9)
        arrival_soc_mean = config.get("arrival_soc_mean", 0.3)
        
        for idx, bus in enumerate(buses):
            n_veh = n_vehicles[idx] if idx < len(n_vehicles) else n_vehicles[-1]
            
            station = {
                "id": idx,
                "bus": bus,
                "n_vehicles": n_veh,
                # 聚合容量（整站）
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
        
        print(f"  已加载 {self.n_ev_stations} 个EV充电站:")
        for station in self.ev_stations:
            print(f"    充电站{station['id']}: 节点{station['bus']+1}, "
                  f"{station['n_vehicles']}辆EV, "
                  f"总容量{station['capacity_mwh']:.2f}MWh, "
                  f"最大功率{station['max_power_mw']:.2f}MW")
    
    # ========== 公共接口（与ieee33/ieee123保持一致） ==========
    
    def get_upstream_branch(self, bus):
        """获取节点上游支路索引"""
        if bus == self.root_bus:
            return None
        parent = self.parent[bus]
        if parent < 0:
            return None
        return self.branch_idx.get((parent, bus))
    
    def get_downstream_branches(self, bus):
        """获取节点下游支路索引列表"""
        branches = []
        for child in self.children[bus]:
            branch_idx = self.branch_idx.get((bus, child))
            if branch_idx is not None:
                branches.append(branch_idx)
        return branches
    
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
    
    def get_branch_params(self, branch_idx=None):
        """
        获取支路参数
        
        Args:
            branch_idx: 支路索引，如果为None则返回所有支路参数
        
        Returns:
            dict: 包含r_pu, x_pu, from_bus, to_bus的数组或单值
        """
        if branch_idx is not None:
            return {
                "from_bus": int(self.from_bus[branch_idx]),
                "to_bus": int(self.to_bus[branch_idx]),
                "r_pu": self.r_pu[branch_idx],
                "x_pu": self.x_pu[branch_idx],
                "r_ohm": self.r_ohm[branch_idx],
                "x_ohm": self.x_ohm[branch_idx],
            }
        else:
            return {
                "r_pu": self.r_pu.copy(),
                "x_pu": self.x_pu.copy(),
                "from_bus": self.from_bus.copy(),
                "to_bus": self.to_bus.copy(),
                "r_ohm": self.r_ohm.copy(),
                "x_ohm": self.x_ohm.copy(),
            }
    
    def get_tie_switch_params(self, tie_idx=None):
        """
        获取联络线参数
        
        Args:
            tie_idx: 联络线索引，如果为None则返回所有联络线参数
        """
        if not self.tie_switches:
            return None
        
        if tie_idx is not None:
            if tie_idx < 0 or tie_idx >= self.n_tie_switches:
                return None
            return self.tie_switches[tie_idx]
        else:
            n_ties = len(self.tie_switches)
            return {
                "from_bus": np.array([ts["from"] for ts in self.tie_switches]),
                "to_bus": np.array([ts["to"] for ts in self.tie_switches]),
                "r_pu": np.array([ts["r_pu"] for ts in self.tie_switches]),
                "x_pu": np.array([ts["x_pu"] for ts in self.tie_switches]),
                "n_ties": n_ties,
            }
    
    def get_bus_load(self, bus=None, p_factor=1.0, q_factor=1.0):
        """
        获取节点负荷
        
        Args:
            bus: 节点编号，如果为None则返回所有节点负荷
            p_factor: 有功负荷因子（用于时变负荷）
            q_factor: 无功负荷因子
        
        Returns:
            dict 或 tuple: 负荷数据
        """
        if bus is not None:
            return (
                self.p_load_pu[bus] * p_factor,
                self.q_load_pu[bus] * q_factor
            )
        else:
            return {
                "p_mw": self.p_load_mw.copy(),
                "q_mvar": self.q_load_mvar.copy(),
                "p_kw": self.p_load_kw.copy(),
                "q_kvar": self.q_load_kvar.copy(),
                "p_pu": self.p_load_pu.copy(),
                "q_pu": self.q_load_pu.copy(),
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
    
    def get_all_branches_for_reconfiguration(self):
        """
        获取用于网络重构的所有支路信息
        
        Returns:
            dict: {
                "fixed": 固定支路列表（68条）,
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
        print("IEEE 69节点配电网")
        print("=" * 50)
        print(f"节点数: {self.n_buses}")
        print(f"固定支路数: {self.n_branches}")
        print(f"联络线数: {self.n_tie_switches}")
        print(f"EV充电站数: {self.n_ev_stations}")
        print(f"基准电压: {self.v_base} kV")
        print(f"基准功率: {self.s_base} MVA")
        print(f"基准阻抗: {self.z_base:.4f} Ω")
        print(f"\n总负荷:")
        print(f"  有功: {self.total_p_load_mw:.3f} MW ({self.total_p_load_mw*1000:.1f} kW)")
        print(f"  无功: {self.total_q_load_mvar:.3f} MVar ({self.total_q_load_mvar*1000:.1f} kVar)")
        
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
def get_ieee69_network(config: dict = None, tie_switch_config: dict = None, 
                       ev_station_config: dict = None):
    """
    获取IEEE69网络实例
    
    Args:
        config: 完整配置字典，包含tie_switch和ev_station设置
        tie_switch_config: 联络线配置字典，如果为None则使用默认辐射状网络
        ev_station_config: EV充电站配置字典
    
    Returns:
        IEEE69Network实例
    """
    # 如果直接传入了配置，优先使用
    if tie_switch_config is not None or ev_station_config is not None:
        return IEEE69Network(
            tie_switch_config=tie_switch_config,
            ev_station_config=ev_station_config
        )
    
    # 如果没有任何配置
    if config is None:
        return IEEE69Network()
    
    # 从完整配置中提取
    tie_config = config.get("devices", {}).get("tie_switches", None)
    ev_config = config.get("devices", {}).get("ev_stations", None)
    
    return IEEE69Network(
        tie_switch_config=tie_config,
        ev_station_config=ev_config
    )


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试基本网络
    print("=== 测试基本辐射状网络 ===")
    net = get_ieee69_network()
    net.summary()
    
    # 打印部分支路参数
    print("\n前10条支路参数:")
    for i in range(10):
        params = net.get_branch_params(i)
        print(f"  支路{i}: {params['from_bus']}->{params['to_bus']}, "
              f"r={params['r_pu']:.6f} pu, x={params['x_pu']:.6f} pu")
    
    # 打印拓扑结构
    print(f"\n节点5的子节点: {net.children[5]}")
    print(f"节点10的父节点: {net.parent[10]}")
    print(f"节点10的上游支路: {net.get_upstream_branch(10)}")
    
    # 检查支路阻抗范围
    branch_params = net.get_branch_params()
    print(f"\n支路阻抗范围:")
    print(f"  R: [{branch_params['r_pu'].min():.6f}, {branch_params['r_pu'].max():.6f}] pu")
    print(f"  X: [{branch_params['x_pu'].min():.6f}, {branch_params['x_pu'].max():.6f}] pu")
    
    # 检查负荷
    loads = net.get_bus_load()
    nonzero_loads = np.where(loads['p_kw'] > 0)[0]
    print(f"\n有负荷的节点数: {len(nonzero_loads)}")
    print(f"负荷节点示例: {nonzero_loads[:10].tolist()}...")
    
    # 测试带联络线的网络
    print("\n\n=== 测试带联络线的网络 ===")
    tie_config = {
        "enabled": True,
        "branches": [
            {"id": 69, "from": 10, "to": 42, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 70, "from": 12, "to": 20, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 71, "from": 14, "to": 45, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 72, "from": 26, "to": 64, "r_ohm": 2.0, "x_ohm": 2.0},
            {"id": 73, "from": 49, "to": 58, "r_ohm": 2.0, "x_ohm": 2.0},
        ],
        "initial_status": [0, 0, 0, 0, 0],
    }
    net_with_ties = get_ieee69_network(tie_switch_config=tie_config)
    net_with_ties.summary()
    
    # 获取重构用支路信息
    all_branches = net_with_ties.get_all_branches_for_reconfiguration()
    print(f"\n总支路数（含联络线）: {all_branches['total']}")
    print(f"固定支路: {len(all_branches['fixed'])}条")
    print(f"联络线: {len(all_branches['tie'])}条")
    
    # 测试带EV充电站的网络
    print("\n\n=== 测试带EV充电站的网络 ===")
    ev_config = {
        "enabled": True,
        "buses": [27, 50, 65],
        "n_vehicles": [150, 200, 180],
        "max_charge_rate_kw": 7.0,
    }
    net_with_ev = get_ieee69_network(ev_station_config=ev_config)
    net_with_ev.summary()
