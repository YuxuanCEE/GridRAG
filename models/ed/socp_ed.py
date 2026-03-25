# -*- coding: utf-8 -*-
"""
含储能和网络重构的经济调度SOCP模型 (Economic Dispatch with ESS and Reconfiguration)

优化变量:
- 连续变量: 储能充放电功率, 光伏削减, DG无功输出, 潮流变量
- 离散变量: 联络线开关状态 (Binary)

目标函数:
    Min F = C_grid*P_grid + C_loss*P_loss + C_ESS*(P_ch+P_dis) + C_cut*P_cut + C_sw*N_switch

约束:
- 潮流方程 (SOCP) - 含联络线条件化约束
- 储能SOC动态约束
- 辐射状拓扑约束 (单商品流)
- 电压限制
- 开关动作次数限制

参考文献:
[1] Enhanced IEEE 33 Bus Benchmark Test System for Distribution System Studies
[2] Network reconfiguration in distribution systems for loss reduction and load balancing
"""

import time
import numpy as np
import pyomo.environ as pyo
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base_model import BaseOptimizationModel
from data.network.ieee33 import IEEE33Network


class EDOptModel(BaseOptimizationModel):
    """含储能和网络重构的经济调度SOCP模型"""
    
    def __init__(self, config: dict):
        super().__init__(name="ED_SOCP_Reconfig", config=config)
        
        self.network = None
        self.scenario_data = None
        self.price_data = None
        
        # ED配置
        ed_config = config["optimization"]["ed"]
        self.n_periods = ed_config["n_periods"]
        self.delta_t = ed_config["delta_t"]  # 时间步长（小时）
        self.allow_curtailment = ed_config.get("allow_pv_curtailment", True)
        self.curtailment_cost = ed_config.get("pv_curtailment_cost", 700)  # 元/MWh
        self.loss_cost = ed_config.get("loss_cost", 500)  # 元/MWh
        
        # 网络重构配置
        self.enable_reconfiguration = ed_config.get("enable_reconfiguration", True)
        
        # 设备配置
        self.pv_config = config["devices"]["pv"]
        self.wt_config = config["devices"]["wt"]
        self.ess_config = config["devices"]["ess"]
        
        # 联络线配置
        self.tie_config = config["devices"].get("tie_switches", {"enabled": False})
        self.tie_enabled = self.tie_config.get("enabled", False) and self.enable_reconfiguration
        self.switching_cost = self.tie_config.get("switching_cost", 50)  # 元/次
        
        # 储能充放电互斥配置
        self.ess_mutex_enabled = self.ess_config.get("charge_discharge_mutex", True)
        
        # 电价配置
        self.price_config = config["price"]
        
        # 网络配置
        self.v_min = config["network"]["v_min"]
        self.v_max = config["network"]["v_max"]
        self.s_base = config["network"]["s_base"]
    
    def _get_price_profile(self) -> np.ndarray:
        """根据配置生成分时电价序列"""
        price = np.zeros(self.n_periods)
        
        peak_hours = set(self.price_config["peak_hours"])
        valley_hours = set(self.price_config["valley_hours"])
        
        peak_price = self.price_config["peak_price"]
        valley_price = self.price_config["valley_price"]
        flat_price = self.price_config["flat_price"]
        
        for t in range(self.n_periods):
            hour = (t * 15 // 60) + 1
            if hour > 24:
                hour = hour - 24
            
            if hour in peak_hours:
                price[t] = peak_price
            elif hour in valley_hours:
                price[t] = valley_price
            else:
                price[t] = flat_price
        
        return price
    
    def build_model(self, network: IEEE33Network, scenario_data: Dict, **kwargs):
        """构建ED-SOCP优化模型（含网络重构）"""
        self.network = network
        self.scenario_data = scenario_data
        self.price_data = self._get_price_profile()
        
        # 创建模型
        model = pyo.ConcreteModel(name="ED_SOCP_Reconfig")
        
        # ====== 1. 基本集合和潮流变量 ======
        self._add_basic_sets_and_variables(model)
        
        # ====== 2. 储能变量 ======
        self._add_ess_variables(model)
        
        # ====== 3. DG变量（有功削减、无功输出）======
        self._add_dg_variables(model)
        
        # ====== 4. 购电功率变量 ======
        self._add_grid_variables(model)
        
        # ====== 5. 联络线开关变量（如果启用重构）======
        if self.tie_enabled and network.n_tie_switches > 0:
            self._add_tie_switch_variables(model)
        
        # ====== 6. 潮流约束 ======
        self._add_power_flow_constraints(model)
        
        # ====== 7. 联络线约束（如果启用）======
        if self.tie_enabled and network.n_tie_switches > 0:
            self._add_tie_line_constraints(model)
            self._add_radiality_constraints(model)
        
        # ====== 8. 储能约束 ======
        self._add_ess_constraints(model)
        
        # ====== 9. DG约束 ======
        self._add_dg_constraints(model)
        
        # ====== 10. 电压约束 ======
        self._add_voltage_constraints(model)
        
        # ====== 11. 根节点约束 ======
        self._add_root_voltage_constraint(model)
        
        # ====== 12. 目标函数 ======
        self._add_objective(model)
        
        self.model = model
        
        # 更新统计信息
        self._update_model_statistics()
        
        return model
    
    def _add_basic_sets_and_variables(self, model):
        """添加基本集合和潮流变量"""
        network = self.network
        n_periods = self.n_periods
        
        # 集合定义
        model.T = pyo.RangeSet(0, n_periods - 1)
        model.N = pyo.RangeSet(0, network.n_buses - 1)
        model.L = pyo.RangeSet(0, network.n_branches - 1)
        
        # 支路索引集合（固定支路）
        branch_set = []
        for l in range(network.n_branches):
            f = int(network.from_bus[l])
            t = int(network.to_bus[l])
            branch_set.append((f, t))
        model.Branch = pyo.Set(initialize=branch_set)
        
        # 有功功率流
        model.P = pyo.Var(model.T, model.Branch, within=pyo.Reals,
                          bounds=(-10, 10), initialize=0)
        
        # 无功功率流
        model.Q = pyo.Var(model.T, model.Branch, within=pyo.Reals,
                          bounds=(-10, 10), initialize=0)
        
        # 电流幅值平方
        model.l = pyo.Var(model.T, model.Branch, within=pyo.NonNegativeReals,
                          bounds=(0, 100), initialize=0)
        
        # 电压幅值平方
        model.u = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals,
                          bounds=(0.8, 1.2), initialize=1.0)
    
    def _add_tie_switch_variables(self, model):
        """添加联络线开关变量和潮流变量"""
        network = self.network
        n_ties = network.n_tie_switches
        
        if n_ties == 0:
            return
        
        # 联络线集合
        model.TieSwitch_Set = pyo.RangeSet(0, n_ties - 1)
        
        # 联络线索引集合（用于潮流）
        tie_branch_set = []
        for k, tie in enumerate(network.tie_switches):
            tie_branch_set.append((tie["from"], tie["to"]))
        model.TieBranch = pyo.Set(initialize=tie_branch_set)
        
        # 1. 联络线开关状态 sw ∈ {0,1}
        # 0 = 断开（默认），1 = 闭合
        model.sw_tie = pyo.Var(model.T, model.TieSwitch_Set, 
                               within=pyo.Binary, initialize=0)
        
        # 2. 联络线有功潮流
        model.P_tie = pyo.Var(model.T, model.TieSwitch_Set, 
                              within=pyo.Reals, bounds=(-10, 10), initialize=0)
        
        # 3. 联络线无功潮流
        model.Q_tie = pyo.Var(model.T, model.TieSwitch_Set,
                              within=pyo.Reals, bounds=(-10, 10), initialize=0)
        
        # 4. 联络线电流平方
        model.l_tie = pyo.Var(model.T, model.TieSwitch_Set,
                              within=pyo.NonNegativeReals, bounds=(0, 100), initialize=0)
        
        # 5. 开关状态变化指示变量（用于计算切换次数）
        model.sw_change = pyo.Var(model.T, model.TieSwitch_Set,
                                  within=pyo.Binary, initialize=0)
        
        # 存储联络线信息供后续使用
        model._tie_switches = network.tie_switches
        model._n_ties = n_ties
        
        print(f"  添加联络线变量: {n_ties}条联络线 × {self.n_periods}时段")
        print(f"  二进制变量数: {n_ties * self.n_periods * 2} (开关状态+变化指示)")
    
    def _add_ess_variables(self, model):
        """添加储能相关变量（含充放电互斥二进制变量）"""
        ess_buses = self.ess_config["buses"]
        capacity = self.ess_config["capacity_mwh"]
        max_ch_rate = self.ess_config["max_charge_rate"]
        max_dis_rate = self.ess_config["max_discharge_rate"]
        soc_min = self.ess_config["soc_min"]
        soc_max = self.ess_config["soc_max"]
        
        n_ess = len(ess_buses)
        
        model.ESS_Set = pyo.RangeSet(0, n_ess - 1)
        
        def p_ch_bounds(m, t, k):
            return (0, capacity[k] * max_ch_rate)
        model.P_ch = pyo.Var(model.T, model.ESS_Set, within=pyo.NonNegativeReals,
                             bounds=p_ch_bounds, initialize=0)
        
        def p_dis_bounds(m, t, k):
            return (0, capacity[k] * max_dis_rate)
        model.P_dis = pyo.Var(model.T, model.ESS_Set, within=pyo.NonNegativeReals,
                              bounds=p_dis_bounds, initialize=0)
        
        def e_soc_bounds(m, t, k):
            return (capacity[k] * soc_min, capacity[k] * soc_max)
        model.E_soc = pyo.Var(model.T, model.ESS_Set, within=pyo.NonNegativeReals,
                              bounds=e_soc_bounds)
        
        # 充放电互斥二进制变量
        # ess_mode = 1: 放电模式（允许放电，禁止充电）
        # ess_mode = 0: 充电模式（允许充电，禁止放电）
        if self.ess_mutex_enabled:
            model.ess_mode = pyo.Var(model.T, model.ESS_Set, within=pyo.Binary, initialize=0)
            print(f"  添加储能互斥变量: {n_ess}台ESS × {self.n_periods}时段 = {n_ess * self.n_periods}个二进制变量")
        
        model._ess_buses = ess_buses
        model._ess_capacity = capacity
        model._ess_max_ch = [capacity[k] * max_ch_rate for k in range(n_ess)]
        model._ess_max_dis = [capacity[k] * max_dis_rate for k in range(n_ess)]
    
    def _add_dg_variables(self, model):
        """添加DG相关变量"""
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        pv_capacity = self.pv_config["capacity"]
        
        n_pv = len(pv_buses)
        n_wt = len(wt_buses)
        
        model.PV_Set = pyo.RangeSet(0, n_pv - 1)
        model.WT_Set = pyo.RangeSet(0, n_wt - 1)
        
        if self.allow_curtailment:
            def p_cut_bounds(m, t, j):
                return (0, pv_capacity[j])
            model.P_cut = pyo.Var(model.T, model.PV_Set, within=pyo.NonNegativeReals,
                                   bounds=p_cut_bounds, initialize=0)
        
        model.Q_pv = pyo.Var(model.T, model.PV_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)
        
        model.Q_wt = pyo.Var(model.T, model.WT_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)
        
        model._pv_buses = pv_buses
        model._wt_buses = wt_buses
    
    def _add_grid_variables(self, model):
        """添加购电功率变量"""
        model.P_grid = pyo.Var(model.T, within=pyo.Reals,
                               bounds=(-5, 10), initialize=1)
    
    def _add_power_flow_constraints(self, model):
        """添加潮流约束（固定支路部分）"""
        network = self.network
        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        load_factor = self.scenario_data.get("load_factor", np.ones(self.n_periods))
        
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        ess_buses = self.ess_config["buses"]
        s_base = self.s_base
        
        # 获取联络线端点（如果有）
        tie_endpoints = set()
        if self.tie_enabled and network.n_tie_switches > 0:
            for tie in network.tie_switches:
                tie_endpoints.add(tie["from"])
                tie_endpoints.add(tie["to"])
        
        def active_power_balance_rule(m, t, j):
            """有功功率平衡（含联络线贡献）"""
            # 负荷
            p_load = network.p_load_pu[j] * load_factor[t]
            
            # PV有功注入
            p_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                p_pv_raw = pv_data[j][t] / s_base
                if self.allow_curtailment:
                    p_pv = p_pv_raw - m.P_cut[t, pv_idx] / s_base
                else:
                    p_pv = p_pv_raw
            
            # WT有功注入
            p_wt = 0
            if j in wt_buses:
                p_wt = wt_data[j][t] / s_base
            
            # ESS净注入
            p_ess = 0
            if j in ess_buses:
                ess_idx = ess_buses.index(j)
                p_ess = (m.P_dis[t, ess_idx] - m.P_ch[t, ess_idx]) / s_base
            
            p_inject = p_pv + p_wt + p_ess - p_load
            
            # 从父节点流入（固定支路）
            parent = network.parent[j]
            if parent >= 0:
                branch_idx = network.branch_idx[(parent, j)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                r = network.r_pu[branch_idx]
                p_in = m.P[t, (f, to)] - m.l[t, (f, to)] * r
            else:
                p_in = 0
            
            # 流向子节点（固定支路）
            p_out = 0
            for child in network.children[j]:
                branch_idx = network.branch_idx[(j, child)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                p_out += m.P[t, (f, to)]
            
            # 联络线贡献（如果启用且该节点是联络线端点）
            p_tie_net = 0
            if self.tie_enabled and network.n_tie_switches > 0 and j in tie_endpoints:
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        # 功率从j流出（正向定义为from->to）
                        p_tie_net -= m.P_tie[t, k] / s_base
                    elif j == tie["to"]:
                        # 功率流入j（需扣除损耗）
                        r_tie = tie["r_pu"]
                        p_tie_net += (m.P_tie[t, k] - m.l_tie[t, k] * r_tie) / s_base
            
            return p_inject == p_out - p_in + p_tie_net
        
        def reactive_power_balance_rule(m, t, j):
            """无功功率平衡（含联络线贡献）"""
            q_load = network.q_load_pu[j] * load_factor[t]
            
            q_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                q_pv = m.Q_pv[t, pv_idx] / s_base
            
            q_wt = 0
            if j in wt_buses:
                wt_idx = wt_buses.index(j)
                q_wt = m.Q_wt[t, wt_idx] / s_base
            
            q_inject = q_pv + q_wt - q_load
            
            parent = network.parent[j]
            if parent >= 0:
                branch_idx = network.branch_idx[(parent, j)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                x = network.x_pu[branch_idx]
                q_in = m.Q[t, (f, to)] - m.l[t, (f, to)] * x
            else:
                q_in = 0
            
            q_out = 0
            for child in network.children[j]:
                branch_idx = network.branch_idx[(j, child)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                q_out += m.Q[t, (f, to)]
            
            # 联络线贡献
            q_tie_net = 0
            if self.tie_enabled and network.n_tie_switches > 0 and j in tie_endpoints:
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        q_tie_net -= m.Q_tie[t, k] / s_base
                    elif j == tie["to"]:
                        x_tie = tie["x_pu"]
                        q_tie_net += (m.Q_tie[t, k] - m.l_tie[t, k] * x_tie) / s_base
            
            return q_inject == q_out - q_in + q_tie_net
        
        def voltage_drop_rule(m, t, f, to):
            branch_idx = network.branch_idx[(f, to)]
            r = network.r_pu[branch_idx]
            x = network.x_pu[branch_idx]
            return m.u[t, to] == m.u[t, f] - 2*(r*m.P[t,(f,to)] + x*m.Q[t,(f,to)]) + (r**2 + x**2)*m.l[t,(f,to)]
        
        def soc_rule(m, t, f, to):
            return m.P[t,(f,to)]**2 + m.Q[t,(f,to)]**2 <= m.l[t,(f,to)] * m.u[t, f]
        
        model.ActivePowerBalance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1),
            rule=active_power_balance_rule
        )
        
        model.ReactivePowerBalance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1),
            rule=reactive_power_balance_rule
        )
        
        model.VoltageDrop = pyo.Constraint(model.T, model.Branch, rule=voltage_drop_rule)
        model.SOC = pyo.Constraint(model.T, model.Branch, rule=soc_rule)
        
        def root_power_rule(m, t):
            root = network.root_bus
            p_out = 0
            for child in network.children[root]:
                branch_idx = network.branch_idx[(root, child)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                p_out += m.P[t, (f, to)]
            return m.P_grid[t] == p_out * s_base
        
        model.RootPower = pyo.Constraint(model.T, rule=root_power_rule)
    
    def _add_tie_line_constraints(self, model):
        """添加联络线相关约束（Big-M条件化）"""
        network = self.network
        M_power = 10.0  # Big-M for power flow
        M_current = 100.0  # Big-M for current
        
        # 1. 联络线潮流条件化约束（开关断开时潮流为0）
        def tie_p_upper(m, t, k):
            return m.P_tie[t, k] <= M_power * m.sw_tie[t, k]
        
        def tie_p_lower(m, t, k):
            return m.P_tie[t, k] >= -M_power * m.sw_tie[t, k]
        
        def tie_q_upper(m, t, k):
            return m.Q_tie[t, k] <= M_power * m.sw_tie[t, k]
        
        def tie_q_lower(m, t, k):
            return m.Q_tie[t, k] >= -M_power * m.sw_tie[t, k]
        
        def tie_l_upper(m, t, k):
            return m.l_tie[t, k] <= M_current * m.sw_tie[t, k]
        
        model.TieP_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_p_upper)
        model.TieP_Lower = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_p_lower)
        model.TieQ_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_q_upper)
        model.TieQ_Lower = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_q_lower)
        model.TieL_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_l_upper)
        
        # 2. 联络线SOC约束（当开关闭合时）
        def tie_soc_rule(m, t, k):
            tie = network.tie_switches[k]
            f_bus = tie["from"]
            return m.P_tie[t, k]**2 + m.Q_tie[t, k]**2 <= m.l_tie[t, k] * m.u[t, f_bus]
        
        model.TieSOC = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_soc_rule)
        
        # 3. 联络线电压降落约束（Big-M relaxation）
        M_voltage = 1.0  # Big-M for voltage
        
        def tie_voltage_drop_upper(m, t, k):
            tie = network.tie_switches[k]
            f_bus, t_bus = tie["from"], tie["to"]
            r, x = tie["r_pu"], tie["x_pu"]
            voltage_drop = m.u[t, t_bus] - m.u[t, f_bus] + 2*(r*m.P_tie[t,k] + x*m.Q_tie[t,k]) \
                          - (r**2 + x**2)*m.l_tie[t,k]
            return voltage_drop <= M_voltage * (1 - m.sw_tie[t, k])
        
        def tie_voltage_drop_lower(m, t, k):
            tie = network.tie_switches[k]
            f_bus, t_bus = tie["from"], tie["to"]
            r, x = tie["r_pu"], tie["x_pu"]
            voltage_drop = m.u[t, t_bus] - m.u[t, f_bus] + 2*(r*m.P_tie[t,k] + x*m.Q_tie[t,k]) \
                          - (r**2 + x**2)*m.l_tie[t,k]
            return voltage_drop >= -M_voltage * (1 - m.sw_tie[t, k])
        
        model.TieVoltageDrop_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_voltage_drop_upper)
        model.TieVoltageDrop_Lower = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_voltage_drop_lower)
        
        # 4. 开关状态变化检测（用于计算切换成本）
        def sw_change_upper(m, t, k):
            if t == 0:
                init_status = self.tie_config.get("initial_status", [0]*network.n_tie_switches)[k]
                return m.sw_change[t, k] >= m.sw_tie[t, k] - init_status
            else:
                return m.sw_change[t, k] >= m.sw_tie[t, k] - m.sw_tie[t-1, k]
        
        def sw_change_lower(m, t, k):
            if t == 0:
                init_status = self.tie_config.get("initial_status", [0]*network.n_tie_switches)[k]
                return m.sw_change[t, k] >= init_status - m.sw_tie[t, k]
            else:
                return m.sw_change[t, k] >= m.sw_tie[t-1, k] - m.sw_tie[t, k]
        
        model.SwChange_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=sw_change_upper)
        model.SwChange_Lower = pyo.Constraint(model.T, model.TieSwitch_Set, rule=sw_change_lower)
        
        # 5. 每日最大切换次数限制
        max_actions = self.tie_config.get("max_daily_actions", 6)
        
        def max_switching_rule(m):
            return sum(m.sw_change[t, k] for t in m.T for k in m.TieSwitch_Set) <= max_actions
        
        model.MaxSwitching = pyo.Constraint(rule=max_switching_rule)
        
        print(f"  添加联络线约束: Big-M={M_power}, 最大切换次数={max_actions}")
    
    def _add_radiality_constraints(self, model):
        """
        添加辐射状约束 - 确保网络始终保持树形结构
        
        使用单商品流(Single Commodity Flow)方法:
        - 从根节点注入(n-1)单位虚拟商品
        - 每个非根节点消耗1单位
        - 商品只能在闭合的支路上流动
        """
        network = self.network
        n_buses = network.n_buses
        n_branches = network.n_branches
        n_ties = network.n_tie_switches
        
        # 总支路数 = 固定支路 + 联络线
        n_total_branches = n_branches + n_ties
        
        # 商品流变量
        # 对于固定支路，商品流始终可以流动（开关始终闭合）
        # 对于联络线，商品流受开关状态限制
        
        # 固定支路商品流（无方向限制，可双向流动）
        model.f_fixed = pyo.Var(model.T, model.L, within=pyo.Reals,
                                bounds=(-(n_buses-1), n_buses-1), initialize=0)
        
        # 联络线商品流
        model.f_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Reals,
                              bounds=(-(n_buses-1), n_buses-1), initialize=0)
        
        # 辐射状约束1: 闭合支路数 = n_buses - 1 = 32
        def radiality_count_rule(m, t):
            # 固定支路始终闭合(32条) + 闭合的联络线数
            n_closed_ties = sum(m.sw_tie[t, k] for k in m.TieSwitch_Set)
            return n_branches + n_closed_ties == n_buses - 1
        
        model.RadialityCount = pyo.Constraint(model.T, rule=radiality_count_rule)
        
        # 辐射状约束2: 商品流平衡
        def commodity_flow_balance_rule(m, t, j):
            if j == network.root_bus:
                # 根节点：流出总量 = n-1
                flow_out = 0
                for child in network.children[j]:
                    branch_idx = network.branch_idx[(j, child)]
                    flow_out += m.f_fixed[t, branch_idx]
                return flow_out == n_buses - 1
            else:
                # 非根节点：流入 - 流出 = 1
                # 从父节点流入（固定支路）
                parent = network.parent[j]
                branch_idx = network.branch_idx[(parent, j)]
                flow_in = m.f_fixed[t, branch_idx]
                
                # 流向子节点（固定支路）
                flow_out = 0
                for child in network.children[j]:
                    child_branch_idx = network.branch_idx[(j, child)]
                    flow_out += m.f_fixed[t, child_branch_idx]
                
                # 联络线商品流贡献
                tie_flow = 0
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        tie_flow -= m.f_tie[t, k]  # 流出
                    elif j == tie["to"]:
                        tie_flow += m.f_tie[t, k]  # 流入
                
                return flow_in - flow_out + tie_flow == 1
        
        model.CommodityFlowBalance = pyo.Constraint(
            model.T, model.N, rule=commodity_flow_balance_rule
        )
        
        # 辐射状约束3: 联络线商品流受开关状态限制
        M_flow = n_buses - 1
        
        def tie_flow_upper(m, t, k):
            return m.f_tie[t, k] <= M_flow * m.sw_tie[t, k]
        
        def tie_flow_lower(m, t, k):
            return m.f_tie[t, k] >= -M_flow * m.sw_tie[t, k]
        
        model.TieFlow_Upper = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_flow_upper)
        model.TieFlow_Lower = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_flow_lower)
        
        print(f"  添加辐射状约束: 单商品流模型, 节点数={n_buses}")
    
    def _add_ess_constraints(self, model):
        """添加储能约束（含充放电互斥）"""
        ess_config = self.ess_config
        capacity = ess_config["capacity_mwh"]
        eta_ch = ess_config["efficiency_charge"]
        eta_dis = ess_config["efficiency_discharge"]
        soc_init = ess_config["soc_init"]
        soc_final_constraint = ess_config.get("soc_final_constraint", True)
        delta_t = self.delta_t
        
        def soc_dynamics_rule(m, t, k):
            if t == 0:
                E_prev = soc_init * capacity[k]
            else:
                E_prev = m.E_soc[t-1, k]
            return m.E_soc[t, k] == E_prev + eta_ch * m.P_ch[t, k] * delta_t - m.P_dis[t, k] / eta_dis * delta_t
        
        model.SOC_Dynamics = pyo.Constraint(model.T, model.ESS_Set, rule=soc_dynamics_rule)
        
        if soc_final_constraint:
            def final_soc_rule(m, k):
                E_init = soc_init * capacity[k]
                E_final = m.E_soc[self.n_periods - 1, k]
                return (0.9 * E_init, E_final, 1.1 * E_init)
            
            model.FinalSOC = pyo.Constraint(model.ESS_Set, rule=final_soc_rule)
        
        # 充放电互斥约束
        if self.ess_mutex_enabled:
            max_ch = model._ess_max_ch
            max_dis = model._ess_max_dis
            
            # 当 ess_mode = 1 (放电模式): P_ch <= 0, P_dis <= max_dis
            # 当 ess_mode = 0 (充电模式): P_ch <= max_ch, P_dis <= 0
            def ess_charge_mutex_rule(m, t, k):
                # P_ch <= max_ch * (1 - ess_mode)
                return m.P_ch[t, k] <= max_ch[k] * (1 - m.ess_mode[t, k])
            
            def ess_discharge_mutex_rule(m, t, k):
                # P_dis <= max_dis * ess_mode
                return m.P_dis[t, k] <= max_dis[k] * m.ess_mode[t, k]
            
            model.ESS_Charge_Mutex = pyo.Constraint(model.T, model.ESS_Set, rule=ess_charge_mutex_rule)
            model.ESS_Discharge_Mutex = pyo.Constraint(model.T, model.ESS_Set, rule=ess_discharge_mutex_rule)
            
            print(f"  添加储能互斥约束: {len(ess_config['buses'])}台ESS × {self.n_periods}时段 × 2 = {len(ess_config['buses']) * self.n_periods * 2}个约束")
    
    def _add_dg_constraints(self, model):
        """添加DG约束"""
        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        pv_capacity = self.pv_config["capacity"]
        wt_capacity = self.wt_config["capacity"]
        
        if self.allow_curtailment:
            def pv_curtail_rule(m, t, j):
                bus = pv_buses[j]
                p_pv_mw = pv_data[bus][t]
                return m.P_cut[t, j] <= p_pv_mw
            
            model.PV_Curtail_Limit = pyo.Constraint(model.T, model.PV_Set, rule=pv_curtail_rule)
        
        def pv_q_capacity_rule(m, t, j):
            bus = pv_buses[j]
            p_pv_mw = pv_data[bus][t]
            if self.allow_curtailment:
                p_actual = p_pv_mw - m.P_cut[t, j]
            else:
                p_actual = p_pv_mw
            s_max = pv_capacity[j] * 1.0
            return m.Q_pv[t, j]**2 + p_actual**2 <= s_max**2
        
        model.PV_Q_Capacity = pyo.Constraint(model.T, model.PV_Set, rule=pv_q_capacity_rule)
        
        def wt_q_capacity_rule(m, t, j):
            bus = wt_buses[j]
            p_wt_mw = wt_data[bus][t]
            s_max = wt_capacity[j] * 1.0
            return m.Q_wt[t, j]**2 + p_wt_mw**2 <= s_max**2
        
        model.WT_Q_Capacity = pyo.Constraint(model.T, model.WT_Set, rule=wt_q_capacity_rule)
    
    def _add_voltage_constraints(self, model):
        """添加电压限制约束"""
        u_min_sq = self.v_min ** 2
        u_max_sq = self.v_max ** 2
        
        def voltage_limit_rule(m, t, j):
            return (u_min_sq, m.u[t, j], u_max_sq)
        
        model.VoltageLimits = pyo.Constraint(model.T, model.N, rule=voltage_limit_rule)
    
    def _add_root_voltage_constraint(self, model):
        """添加根节点电压约束"""
        root = self.network.root_bus
        v0 = 1.0
        
        def root_voltage_rule(m, t):
            return m.u[t, root] == v0 ** 2
        
        model.RootVoltage = pyo.Constraint(model.T, rule=root_voltage_rule)
    
    def _add_objective(self, model):
        """添加目标函数：最小化日运行成本（含切换成本）"""
        network = self.network
        delta_t = self.delta_t
        s_base = self.s_base
        
        # 1. 购电成本
        grid_cost = sum(
            self.price_data[t] * model.P_grid[t] * delta_t
            for t in model.T
        )
        
        # 2. 网损成本（固定支路）
        loss_expr = 0
        for t in model.T:
            for l in range(network.n_branches):
                f = int(network.from_bus[l])
                to = int(network.to_bus[l])
                r = network.r_pu[l]
                loss_expr += model.l[t, (f, to)] * r * s_base
        
        # 联络线网损（如果启用）
        if self.tie_enabled and network.n_tie_switches > 0:
            for t in model.T:
                for k, tie in enumerate(network.tie_switches):
                    r = tie["r_pu"]
                    loss_expr += model.l_tie[t, k] * r * s_base
        
        loss_cost = self.loss_cost * loss_expr * delta_t
        
        # 3. 储能运行成本
        ess_cost_rate = self.ess_config["cost_per_mwh"]
        ess_cost = sum(
            ess_cost_rate * (model.P_ch[t, k] + model.P_dis[t, k]) * delta_t
            for t in model.T
            for k in model.ESS_Set
        )
        
        # 4. 弃光成本
        curtail_cost = 0
        if self.allow_curtailment:
            curtail_cost = sum(
                self.curtailment_cost * model.P_cut[t, j] * delta_t
                for t in model.T
                for j in model.PV_Set
            )
        
        # 5. 开关切换成本（如果启用重构）
        switching_cost = 0
        if self.tie_enabled and network.n_tie_switches > 0:
            switching_cost = sum(
                self.switching_cost * model.sw_change[t, k]
                for t in model.T
                for k in model.TieSwitch_Set
            )
        
        # 总目标
        model.obj = pyo.Objective(
            expr=grid_cost + loss_cost + ess_cost + curtail_cost + switching_cost,
            sense=pyo.minimize
        )
        
        model._grid_cost_expr = grid_cost
        model._loss_cost_expr = loss_cost
        model._ess_cost_expr = ess_cost
        model._curtail_cost_expr = curtail_cost
        model._switching_cost_expr = switching_cost
    
    def _update_model_statistics(self):
        """更新模型统计信息"""
        if self.model is None:
            return
        
        n_vars = sum(1 for v in self.model.component_objects(pyo.Var, active=True)
                     for _ in v)
        
        n_cons = sum(1 for c in self.model.component_objects(pyo.Constraint, active=True)
                     for _ in c)
        
        # 计算二进制变量数
        n_binary = 0
        for v in self.model.component_objects(pyo.Var, active=True):
            for idx in v:
                if v[idx].domain == pyo.Binary:
                    n_binary += 1
        
        self.statistics["n_variables"] = n_vars
        self.statistics["n_constraints"] = n_cons
        self.statistics["n_binary_vars"] = n_binary
        
        print(f"\n模型统计:")
        print(f"  总变量数: {n_vars}")
        print(f"  二进制变量数: {n_binary}")
        print(f"  约束数: {n_cons}")
    
    def solve(self, solver_name: str = "gurobi", **kwargs) -> Dict[str, Any]:
        """求解模型"""
        if self.model is None:
            raise RuntimeError("模型尚未构建，请先调用build_model()")
        
        solver_config = self.config["optimization"]["solver"]
        
        solver = pyo.SolverFactory(solver_name)
        
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = solver_config.get("time_limit", 600)
            solver.options["MIPGap"] = solver_config.get("mip_gap", 1e-3)
            solver.options["OutputFlag"] = 1 if solver_config.get("verbose", True) else 0
            solver.options["NonConvex"] = 2
            # MIP加速设置
            solver.options["Cuts"] = 2  # 启用切平面
            solver.options["Presolve"] = 2  # 预处理
        
        print(f"\n开始求解 {self.name}...")
        print(f"  二进制变量: {self.statistics.get('n_binary_vars', 0)}")
        start_time = time.time()
        
        results = solver.solve(self.model, tee=solver_config.get("verbose", True))
        
        self.solve_time = time.time() - start_time
        self.statistics["solve_time"] = self.solve_time
        
        self.status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        
        self.statistics["solver_status"] = f"{self.status} ({termination})"
        
        has_solution = False
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            has_solution = True
            print(f"求解成功！（最优解）")
        elif results.solver.termination_condition in [pyo.TerminationCondition.maxTimeLimit,
                                                       pyo.TerminationCondition.feasible]:
            has_solution = True
            print(f"求解达到时间限制，但找到可行解")
        
        if has_solution:
            try:
                self.statistics["objective_value"] = pyo.value(self.model.obj)
                print(f"目标值 (总成本): {self.statistics['objective_value']:.2f} 元")
                print(f"求解时间: {self.solve_time:.2f} 秒")
            except:
                print(f"警告: 无法获取目标函数值")
        else:
            print(f"求解状态: {self.status}, 终止条件: {termination}")
        
        return {"status": self.status, "termination": termination}
    
    def get_results(self) -> Dict[str, Any]:
        """获取优化结果"""
        if self.model is None:
            return {}
        
        model = self.model
        n_periods = self.n_periods
        network = self.network
        s_base = self.s_base
        delta_t = self.delta_t
        
        ess_buses = self.ess_config["buses"]
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        
        n_ess = len(ess_buses)
        n_pv = len(pv_buses)
        n_wt = len(wt_buses)
        
        # ====== 提取储能结果 ======
        ess_charge = np.zeros((n_periods, n_ess))
        ess_discharge = np.zeros((n_periods, n_ess))
        ess_soc = np.zeros((n_periods, n_ess))
        ess_mode = np.zeros((n_periods, n_ess), dtype=int)  # 充放电模式
        
        for t in range(n_periods):
            for k in range(n_ess):
                ess_charge[t, k] = pyo.value(model.P_ch[t, k])
                ess_discharge[t, k] = pyo.value(model.P_dis[t, k])
                ess_soc[t, k] = pyo.value(model.E_soc[t, k])
                if self.ess_mutex_enabled:
                    ess_mode[t, k] = int(round(pyo.value(model.ess_mode[t, k])))
        
        # ====== 提取DG结果 ======
        pv_curtail = np.zeros((n_periods, n_pv))
        pv_q = np.zeros((n_periods, n_pv))
        wt_q = np.zeros((n_periods, n_wt))
        
        for t in range(n_periods):
            for j in range(n_pv):
                if self.allow_curtailment:
                    pv_curtail[t, j] = pyo.value(model.P_cut[t, j])
                pv_q[t, j] = pyo.value(model.Q_pv[t, j])
            for j in range(n_wt):
                wt_q[t, j] = pyo.value(model.Q_wt[t, j])
        
        # ====== 提取购电功率 ======
        p_grid = np.zeros(n_periods)
        for t in range(n_periods):
            p_grid[t] = pyo.value(model.P_grid[t])
        
        # ====== 提取电压 ======
        voltage = np.zeros((n_periods, network.n_buses))
        for t in range(n_periods):
            for j in range(network.n_buses):
                u_sq = pyo.value(model.u[t, j])
                voltage[t, j] = np.sqrt(u_sq) if u_sq > 0 else 0
        
        # ====== 计算网损 ======
        loss_per_period = np.zeros(n_periods)
        for t in range(n_periods):
            loss = 0
            for l in range(network.n_branches):
                f = int(network.from_bus[l])
                to = int(network.to_bus[l])
                r = network.r_pu[l]
                loss += pyo.value(model.l[t, (f, to)]) * r
            loss_per_period[t] = loss * s_base * 1000
        
        # ====== 提取联络线结果（如果有）======
        tie_results = None
        if self.tie_enabled and network.n_tie_switches > 0:
            n_ties = network.n_tie_switches
            tie_status = np.zeros((n_periods, n_ties), dtype=int)
            tie_power = np.zeros((n_periods, n_ties))
            tie_changes = np.zeros((n_periods, n_ties), dtype=int)
            
            for t in range(n_periods):
                for k in range(n_ties):
                    tie_status[t, k] = int(round(pyo.value(model.sw_tie[t, k])))
                    tie_power[t, k] = pyo.value(model.P_tie[t, k])
                    tie_changes[t, k] = int(round(pyo.value(model.sw_change[t, k])))
            
            tie_results = {
                "status": tie_status,
                "power_mw": tie_power,
                "changes": tie_changes,
                "total_switches": int(np.sum(tie_changes)),
                "tie_info": [{"id": tie["id"], "from": tie["from"]+1, "to": tie["to"]+1} 
                            for tie in network.tie_switches],
            }
        
        # ====== 计算各项成本 ======
        grid_cost = sum(self.price_data[t] * p_grid[t] * delta_t for t in range(n_periods))
        loss_cost = self.loss_cost * sum(loss_per_period) / 1000 * delta_t
        ess_cost = self.ess_config["cost_per_mwh"] * np.sum(ess_charge + ess_discharge) * delta_t
        curtail_cost = self.curtailment_cost * np.sum(pv_curtail) * delta_t if self.allow_curtailment else 0
        switching_cost = 0
        if tie_results:
            switching_cost = self.switching_cost * tie_results["total_switches"]
        
        results = {
            "ess": {
                "buses": ess_buses,
                "charge_mw": ess_charge,
                "discharge_mw": ess_discharge,
                "soc_mwh": ess_soc,
                "mode": ess_mode if self.ess_mutex_enabled else None,  # 1=放电, 0=充电
            },
            "pv": {
                "buses": pv_buses,
                "curtailment_mw": pv_curtail,
                "reactive_mvar": pv_q,
            },
            "wt": {
                "buses": wt_buses,
                "reactive_mvar": wt_q,
            },
            "grid": {
                "power_mw": p_grid,
                "total_purchase_mwh": np.sum(p_grid[p_grid > 0]) * delta_t,
                "total_sell_mwh": -np.sum(p_grid[p_grid < 0]) * delta_t,
            },
            "voltage": {
                "values": voltage,
                "min": voltage.min(),
                "max": voltage.max(),
                "mean": voltage.mean(),
            },
            "loss": {
                "per_period_kw": loss_per_period,
                "total_kwh": np.sum(loss_per_period) * delta_t,
                "average_kw": loss_per_period.mean(),
            },
            "cost": {
                "total_yuan": self.statistics.get("objective_value", 0),
                "grid_yuan": grid_cost,
                "loss_yuan": loss_cost,
                "ess_yuan": ess_cost,
                "curtail_yuan": curtail_cost,
                "switching_yuan": switching_cost,
            },
            "price": {
                "profile": self.price_data.tolist(),
            },
            "reconfiguration": tie_results,
            "objective": self.statistics.get("objective_value"),
            "solve_time": self.statistics.get("solve_time", 0),
        }
        
        return results
    
    def print_summary(self):
        """打印优化结果摘要"""
        results = self.get_results()
        
        print("\n" + "=" * 60)
        print("经济调度(ED)优化结果摘要" + (" [含网络重构]" if self.tie_enabled else ""))
        print("=" * 60)
        
        cost = results["cost"]
        print(f"\n【成本分析】")
        print(f"  总成本: {cost['total_yuan']:.2f} 元")
        print(f"  购电成本: {cost['grid_yuan']:.2f} 元")
        print(f"  网损成本: {cost['loss_yuan']:.2f} 元")
        print(f"  储能运行成本: {cost['ess_yuan']:.2f} 元")
        print(f"  弃光成本: {cost['curtail_yuan']:.2f} 元")
        if self.tie_enabled:
            print(f"  开关切换成本: {cost['switching_yuan']:.2f} 元")
        
        grid = results["grid"]
        print(f"\n【购电分析】")
        print(f"  总购电量: {grid['total_purchase_mwh']:.2f} MWh")
        print(f"  总售电量: {grid['total_sell_mwh']:.2f} MWh")
        
        ess = results["ess"]
        print(f"\n【储能分析】")
        total_charge = np.sum(ess['charge_mw']) * self.delta_t
        total_discharge = np.sum(ess['discharge_mw']) * self.delta_t
        print(f"  总充电量: {total_charge:.2f} MWh")
        print(f"  总放电量: {total_discharge:.2f} MWh")
        
        # 储能模式切换统计
        if self.ess_mutex_enabled and ess['mode'] is not None:
            ess_mode = np.array(ess['mode'])
            n_ess = ess_mode.shape[1]
            for k in range(n_ess):
                mode_changes = np.sum(np.abs(np.diff(ess_mode[:, k])))
                charge_periods = np.sum(ess_mode[:, k] == 0)
                discharge_periods = np.sum(ess_mode[:, k] == 1)
                print(f"  ESS{k+1}: 充电{charge_periods}时段, 放电{discharge_periods}时段, 切换{mode_changes}次")
        
        volt = results["voltage"]
        print(f"\n【电压分析】")
        print(f"  最低电压: {volt['min']:.4f} pu")
        print(f"  最高电压: {volt['max']:.4f} pu")
        print(f"  平均电压: {volt['mean']:.4f} pu")
        
        loss = results["loss"]
        print(f"\n【网损分析】")
        print(f"  总网损: {loss['total_kwh']:.2f} kWh")
        print(f"  平均网损: {loss['average_kw']:.2f} kW")
        
        if results["reconfiguration"]:
            reconfig = results["reconfiguration"]
            print(f"\n【网络重构分析】")
            print(f"  总切换次数: {reconfig['total_switches']}")
            for info in reconfig["tie_info"]:
                print(f"  联络线{info['id']}: 节点{info['from']}-节点{info['to']}")
            
            # 打印每个时段的开关状态变化
            status = reconfig["status"]
            changes = np.sum(reconfig["changes"], axis=1)
            change_periods = np.where(changes > 0)[0]
            if len(change_periods) > 0:
                print(f"  切换发生时段: {change_periods.tolist()}")
        
        print(f"\n【求解信息】")
        print(f"  求解时间: {results['solve_time']:.2f} 秒")
        print(f"  二进制变量数: {self.statistics.get('n_binary_vars', 0)}")
        
        print("=" * 60)


def create_ed_model(config: dict) -> EDOptModel:
    """创建ED优化模型实例"""
    return EDOptModel(config)
