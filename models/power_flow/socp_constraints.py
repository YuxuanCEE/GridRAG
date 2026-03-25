# -*- coding: utf-8 -*-
"""
SOCP潮流约束模块
基于DistFlow方程的二阶锥松弛

参考文献：
[1] Farivar M, Low S H. Branch flow model: Relaxations and convexification[J]. 
    IEEE Transactions on Power Systems, 2013, 28(3): 2554-2564.
[2] 论文式(1)-(4)
"""

import pyomo.environ as pyo
from typing import Dict, List, Tuple, Optional
import numpy as np


def add_power_flow_variables(model, network, n_periods: int):
    """
    添加潮流相关变量
    
    Args:
        model: Pyomo模型
        network: 网络对象
        n_periods: 时段数
    
    Variables:
        P[t,i,j]: 支路(i,j)在时段t的有功功率流 (pu)
        Q[t,i,j]: 支路(i,j)在时段t的无功功率流 (pu)
        l[t,i,j]: 支路(i,j)在时段t的电流幅值平方 (pu)
        u[t,j]: 节点j在时段t的电压幅值平方 (pu)
    """
    # 集合定义
    model.T = pyo.RangeSet(0, n_periods - 1)  # 时段集合
    model.N = pyo.RangeSet(0, network.n_buses - 1)  # 节点集合
    model.L = pyo.RangeSet(0, network.n_branches - 1)  # 支路集合
    
    # 创建支路索引集合 (from_bus, to_bus)
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


def add_power_balance_constraints(model, network, scenario_data: Dict,
                                   pv_q_vars=None, wt_q_vars=None, sc_q_vars=None):
    """
    添加功率平衡约束 - 式(2)
    
    对于每个节点j:
    P_PV,j + P_WT,j - P_L,j = Σ(P_jv + l_jv*r_jv) - Σ(P_ij) (从j流出 - 流入j)
    Q_PV,j + Q_WT,j + Q_SC,j - Q_L,j = Σ(Q_jv + l_jv*x_jv) - Σ(Q_ij)
    
    Args:
        model: Pyomo模型
        network: 网络对象
        scenario_data: 场景数据，包含PV/WT/负荷信息
        pv_q_vars: PV无功变量（如果有）
        wt_q_vars: WT无功变量（如果有）
        sc_q_vars: SC无功变量（如果有）
    """
    n_periods = scenario_data["n_periods"]
    pv_data = scenario_data.get("pv", {})
    wt_data = scenario_data.get("wt", {})
    load_factor = scenario_data.get("load_factor", np.ones(n_periods))
    
    # 获取设备配置
    pv_buses = list(pv_data.keys())
    wt_buses = list(wt_data.keys())
    
    def active_power_balance_rule(m, t, j):
        """有功功率平衡"""
        # 负荷（乘以时变因子）
        p_load = network.p_load_pu[j] * load_factor[t]
        
        # PV有功注入（已知）
        p_pv = 0
        if j in pv_buses:
            # 转换为pu
            p_pv = pv_data[j][t] / network.s_base
        
        # WT有功注入（已知）
        p_wt = 0
        if j in wt_buses:
            p_wt = wt_data[j][t] / network.s_base
        
        # 净注入功率
        p_inject = p_pv + p_wt - p_load
        
        # 从父节点流入的功率
        parent = network.parent[j]
        if parent >= 0:
            branch_idx = network.branch_idx[(parent, j)]
            f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
            r = network.r_pu[branch_idx]
            
            # 流入功率 = P_parent_j - l_parent_j * r
            p_in = m.P[t, (f, to)] - m.l[t, (f, to)] * r
        else:
            p_in = 0
        
        # 流向子节点的功率之和
        p_out = 0
        for child in network.children[j]:
            branch_idx = network.branch_idx[(j, child)]
            f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
            p_out += m.P[t, (f, to)]
        
        # 平衡约束: 注入 = 流出 - 流入
        return p_inject == p_out - p_in
    
    def reactive_power_balance_rule(m, t, j):
        """无功功率平衡"""
        # 负荷
        q_load = network.q_load_pu[j] * load_factor[t]
        
        # PV无功（如果有无功变量则使用，否则假设功率因数为1）
        q_pv = 0
        if pv_q_vars is not None and j in pv_buses:
            q_pv = pv_q_vars[t, j]
        
        # WT无功
        q_wt = 0
        if wt_q_vars is not None and j in wt_buses:
            q_wt = wt_q_vars[t, j]
        
        # SC无功
        q_sc = 0
        if sc_q_vars is not None and hasattr(m, 'Q_sc'):
            sc_config = getattr(m, '_sc_config', None)
            if sc_config and j in sc_config['buses']:
                bus_idx = sc_config['buses'].index(j)
                q_sc = m.Q_sc[t, bus_idx]
        
        # 净无功注入
        q_inject = q_pv + q_wt + q_sc - q_load
        
        # 从父节点流入
        parent = network.parent[j]
        if parent >= 0:
            branch_idx = network.branch_idx[(parent, j)]
            f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
            x = network.x_pu[branch_idx]
            q_in = m.Q[t, (f, to)] - m.l[t, (f, to)] * x
        else:
            q_in = 0
        
        # 流向子节点
        q_out = 0
        for child in network.children[j]:
            branch_idx = network.branch_idx[(j, child)]
            f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
            q_out += m.Q[t, (f, to)]
        
        return q_inject == q_out - q_in
    
    # 添加约束（跳过根节点）
    model.ActivePowerBalance = pyo.Constraint(
        model.T, pyo.RangeSet(1, network.n_buses - 1),
        rule=active_power_balance_rule
    )
    
    model.ReactivePowerBalance = pyo.Constraint(
        model.T, pyo.RangeSet(1, network.n_buses - 1),
        rule=reactive_power_balance_rule
    )


def add_voltage_drop_constraints(model, network):
    """
    添加电压降落约束 - 式(3)
    
    u_j = u_i + 2(r*P + x*Q) + (r² + x²)*l
    
    简化为:
    u_j = u_i - 2(r*P_ij + x*Q_ij) + (r² + x²)*l_ij
    """
    def voltage_drop_rule(m, t, f, to):
        branch_idx = network.branch_idx[(f, to)]
        r = network.r_pu[branch_idx]
        x = network.x_pu[branch_idx]
        
        # 对于辐射状网络，to是下游节点
        # u_to = u_from - 2*(r*P + x*Q) + (r² + x²)*l
        return m.u[t, to] == m.u[t, f] - 2*(r*m.P[t,(f,to)] + x*m.Q[t,(f,to)]) + (r**2 + x**2)*m.l[t,(f,to)]
    
    model.VoltageDrop = pyo.Constraint(model.T, model.Branch, rule=voltage_drop_rule)


def add_soc_constraints(model, network):
    """
    添加二阶锥约束 - 式(4)
    
    ||[2*P_ij, 2*Q_ij, l_ij - u_j]||₂ ≤ l_ij + u_j
    
    等价于旋转二阶锥:
    (2*P)² + (2*Q)² + (l - u)² ≤ (l + u)²
    即: 4*P² + 4*Q² ≤ 4*l*u
    即: P² + Q² ≤ l*u
    """
    def soc_rule(m, t, f, to):
        # SOC约束: P² + Q² ≤ l * u_to
        # 使用Pyomo的Constraint表达式（Gurobi会识别为二阶锥）
        return m.P[t,(f,to)]**2 + m.Q[t,(f,to)]**2 <= m.l[t,(f,to)] * m.u[t, to]
    
    model.SOC = pyo.Constraint(model.T, model.Branch, rule=soc_rule)


def add_voltage_limits(model, network, v_min: float = 0.95, v_max: float = 1.05):
    """
    添加电压限制约束 - 式(5)部分
    
    U_min² ≤ u_j ≤ U_max²
    """
    u_min_sq = v_min ** 2
    u_max_sq = v_max ** 2
    
    def voltage_limit_rule(m, t, j):
        return (u_min_sq, m.u[t, j], u_max_sq)
    
    model.VoltageLimits = pyo.Constraint(model.T, model.N, rule=voltage_limit_rule)


def add_root_voltage_constraint(model, network, oltc_vars=None, v0: float = 1.0, 
                                 delta_v: float = 0.01):
    """
    添加根节点电压约束（考虑OLTC）- 式(5)、(9)
    
    如果有OLTC:
        u_root = (V0 + δ_oltc * ΔV)²
    否则:
        u_root = V0²
    
    Args:
        model: Pyomo模型
        network: 网络对象
        oltc_vars: OLTC档位变量（如果有）
        v0: 一次侧电压 pu
        delta_v: 每档调节量 pu
    """
    root = network.root_bus
    
    if oltc_vars is not None:
        # 有OLTC的情况
        # u_root = (V0 + δ_oltc * ΔV)²
        # 近似线性化: u_root ≈ V0² + 2*V0*ΔV*δ_oltc
        def root_voltage_rule(m, t):
            # 使用OLTC档位变量
            # δ_oltc 范围是 -5 到 5
            tap_position = m.oltc_tap[t]  # 整数变量
            v_tap = v0 + tap_position * delta_v
            return m.u[t, root] == v_tap ** 2
        
        model.RootVoltage = pyo.Constraint(model.T, rule=root_voltage_rule)
    else:
        # 无OLTC，固定电压
        def fixed_root_voltage_rule(m, t):
            return m.u[t, root] == v0 ** 2
        
        model.RootVoltage = pyo.Constraint(model.T, rule=fixed_root_voltage_rule)


def get_power_loss_expression(model, network):
    """
    获取网损表达式（目标函数）- 式(1)
    
    Loss = Σ_t Σ_(i,j)∈L l_ij * r_ij
    
    Returns:
        Pyomo表达式
    """
    loss_expr = 0
    for t in model.T:
        for l in range(network.n_branches):
            f = int(network.from_bus[l])
            to = int(network.to_bus[l])
            r = network.r_pu[l]
            loss_expr += model.l[t, (f, to)] * r
    
    return loss_expr


def build_basic_opf_model(network, scenario_data: Dict, v_min: float = 0.95, 
                          v_max: float = 1.05) -> pyo.ConcreteModel:
    """
    构建基本的最优潮流模型（无离散设备）
    
    Args:
        network: 网络对象
        scenario_data: 场景数据
        v_min: 电压下限
        v_max: 电压上限
    
    Returns:
        Pyomo模型
    """
    model = pyo.ConcreteModel(name="Basic_OPF")
    
    n_periods = scenario_data["n_periods"]
    
    # 添加变量
    add_power_flow_variables(model, network, n_periods)
    
    # 添加约束
    add_power_balance_constraints(model, network, scenario_data)
    add_voltage_drop_constraints(model, network)
    add_soc_constraints(model, network)
    add_voltage_limits(model, network, v_min, v_max)
    add_root_voltage_constraint(model, network)
    
    # 目标函数
    loss_expr = get_power_loss_expression(model, network)
    model.obj = pyo.Objective(expr=loss_expr, sense=pyo.minimize)
    
    return model
