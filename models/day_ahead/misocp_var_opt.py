# -*- coding: utf-8 -*-
"""
日前MISOCP无功优化模型
第一阶段：优化OLTC档位和SC投切状态

目标: 最小化网损
约束: SOCP潮流约束 + OLTC约束 + SC约束 + 电压约束

参考论文式(1)-(11)
"""

import time
import numpy as np
import pyomo.environ as pyo
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base_model import BaseOptimizationModel
from models.power_flow.socp_constraints import (
    add_power_flow_variables,
    add_voltage_drop_constraints,
    add_soc_constraints,
    add_voltage_limits,
    get_power_loss_expression,
)
from data.network.ieee33 import IEEE33Network


class DayAheadVarOptModel(BaseOptimizationModel):
    """日前无功优化MISOCP模型"""
    
    def __init__(self, config: dict):
        super().__init__(name="DayAhead_MISOCP_VarOpt", config=config)
        
        self.network = None
        self.scenario_data = None
        self.n_periods = config["optimization"]["day_ahead"]["n_periods"]
        
        # 设备配置
        self.oltc_config = config["devices"]["oltc"]
        self.sc_config = config["devices"]["sc"]
        self.pv_config = config["devices"]["pv"]
        self.wt_config = config["devices"]["wt"]
        
        # 网络配置
        self.v_min = config["network"]["v_min"]
        self.v_max = config["network"]["v_max"]
    
    def build_model(self, network: IEEE33Network, scenario_data: Dict, **kwargs):
        """
        构建MISOCP模型
        
        Args:
            network: IEEE33网络对象
            scenario_data: 场景数据
        """
        self.network = network
        self.scenario_data = scenario_data
        
        # 创建模型
        model = pyo.ConcreteModel(name="DayAhead_MISOCP")
        
        # ====== 1. 基本变量和集合 ======
        add_power_flow_variables(model, network, self.n_periods)
        
        # ====== 2. OLTC变量和约束 ======
        self._add_oltc_variables(model)
        self._add_oltc_constraints(model)
        
        # ====== 3. SC变量和约束 ======
        self._add_sc_variables(model)
        self._add_sc_constraints(model)
        
        # ====== 4. 潮流约束 ======
        self._add_power_balance_constraints(model)
        add_voltage_drop_constraints(model, network)
        add_soc_constraints(model, network)
        add_voltage_limits(model, network, self.v_min, self.v_max)
        
        # ====== 5. 根节点电压约束（含OLTC） ======
        self._add_root_voltage_constraint(model)
        
        # ====== 6. 目标函数 ======
        loss_expr = get_power_loss_expression(model, network)
        model.obj = pyo.Objective(expr=loss_expr, sense=pyo.minimize)
        
        self.model = model
        
        # 更新统计信息
        self._update_model_statistics()
        
        return model
    
    def _add_oltc_variables(self, model):
        """
        添加OLTC变量 - 式(9)、(10)
        
        变量:
            oltc_tap[t]: 时段t的档位 (整数, -5到5)
            oltc_lambda[t,d]: 档位指示变量 (二元)
            oltc_phi[t]: 相邻时段档位变化指示 (整数)
        """
        tap_min = self.oltc_config["tap_min"]
        tap_max = self.oltc_config["tap_max"]
        max_actions = self.oltc_config["max_daily_actions"]
        
        # 档位集合 {-5, -4, ..., 5}
        model.TapSet = pyo.RangeSet(tap_min, tap_max)
        
        # 档位整数变量
        model.oltc_tap = pyo.Var(model.T, within=pyo.Integers, 
                                  bounds=(tap_min, tap_max), initialize=0)
        
        # 档位指示二元变量 (用于线性化)
        # lambda[t,d] = 1 表示时段t档位为d
        model.oltc_lambda = pyo.Var(model.T, model.TapSet, within=pyo.Binary, initialize=0)
        
        # 相邻时段变化量（绝对值）
        model.oltc_phi = pyo.Var(model.T, within=pyo.NonNegativeIntegers, 
                                  bounds=(0, tap_max - tap_min), initialize=0)
    
    def _add_oltc_constraints(self, model):
        """
        添加OLTC约束 - 式(9)、(10)
        """
        tap_min = self.oltc_config["tap_min"]
        tap_max = self.oltc_config["tap_max"]
        max_actions = self.oltc_config["max_daily_actions"]
        
        # 约束1: 每个时段只能选择一个档位
        def single_tap_rule(m, t):
            return sum(m.oltc_lambda[t, d] for d in model.TapSet) == 1
        model.OltcSingleTap = pyo.Constraint(model.T, rule=single_tap_rule)
        
        # 约束2: 档位值与指示变量的关系
        def tap_value_rule(m, t):
            return m.oltc_tap[t] == sum(d * m.oltc_lambda[t, d] for d in model.TapSet)
        model.OltcTapValue = pyo.Constraint(model.T, rule=tap_value_rule)
        
        # 约束3: 档位变化量约束 - 式(10)
        # |tap[t+1] - tap[t]| <= phi[t]
        # 线性化: tap[t+1] - tap[t] <= phi[t]
        #         tap[t] - tap[t+1] <= phi[t]
        def tap_change_pos_rule(m, t):
            if t == self.n_periods - 1:
                return pyo.Constraint.Skip
            return m.oltc_tap[t+1] - m.oltc_tap[t] <= m.oltc_phi[t]
        
        def tap_change_neg_rule(m, t):
            if t == self.n_periods - 1:
                return pyo.Constraint.Skip
            return m.oltc_tap[t] - m.oltc_tap[t+1] <= m.oltc_phi[t]
        
        model.OltcChangePos = pyo.Constraint(model.T, rule=tap_change_pos_rule)
        model.OltcChangeNeg = pyo.Constraint(model.T, rule=tap_change_neg_rule)
        
        # 约束4: 每日最大动作次数 - 式(10)
        # Σ phi[t] <= max_actions
        # 注意：原文是相邻档位变化，这里简化为变化次数
        def max_actions_rule(m):
            # 使用变化指示变量
            return sum(m.oltc_phi[t] for t in range(self.n_periods - 1)) <= max_actions
        model.OltcMaxActions = pyo.Constraint(rule=max_actions_rule)
    
    def _add_sc_variables(self, model):
        """
        添加SC变量 - 式(6)、(11)
        
        每个SC有多个档位，每档提供固定无功
        """
        sc_buses = self.sc_config["buses"]
        n_stages = self.sc_config["n_stages"]
        q_per_stage = self.sc_config["q_per_stage"]  # MVar
        max_actions = self.sc_config["max_daily_actions"]
        
        n_sc = len(sc_buses)
        
        # SC集合
        model.SC_Set = pyo.RangeSet(0, n_sc - 1)
        model.SC_Stages = pyo.RangeSet(0, n_stages)  # 0到n_stages档
        
        # 投切状态变量 (整数: 0, 1, 2, ..., n_stages)
        model.sc_stage = pyo.Var(model.T, model.SC_Set, within=pyo.Integers,
                                  bounds=(0, n_stages), initialize=0)
        
        # 无功输出 (连续变量，但由整数变量决定)
        model.Q_sc = pyo.Var(model.T, model.SC_Set, within=pyo.NonNegativeReals,
                              bounds=(0, n_stages * q_per_stage), initialize=0)
        
        # 变化指示变量
        model.sc_change = pyo.Var(model.T, model.SC_Set, within=pyo.Binary, initialize=0)
        
        # 存储配置供后续使用
        model._sc_config = {
            'buses': sc_buses,
            'n_stages': n_stages,
            'q_per_stage': q_per_stage,
            'max_actions': max_actions,
        }
    
    def _add_sc_constraints(self, model):
        """
        添加SC约束 - 式(6)、(11)
        """
        sc_buses = self.sc_config["buses"]
        n_stages = self.sc_config["n_stages"]
        q_per_stage = self.sc_config["q_per_stage"] / self.network.s_base  # 转为pu
        max_actions = self.sc_config["max_daily_actions"]
        
        # 约束1: 无功输出与档位的关系
        # Q_sc[t,v] = sc_stage[t,v] * q_per_stage
        def sc_output_rule(m, t, v):
            return m.Q_sc[t, v] == m.sc_stage[t, v] * q_per_stage
        model.ScOutput = pyo.Constraint(model.T, model.SC_Set, rule=sc_output_rule)
        
        # 约束2: 变化指示约束 - 式(11)
        # 如果 stage[t+1] != stage[t], 则 change[t] = 1
        # 使用大M方法线性化
        M = n_stages + 1
        
        def sc_change_pos_rule(m, t, v):
            if t == self.n_periods - 1:
                return pyo.Constraint.Skip
            return m.sc_stage[t+1, v] - m.sc_stage[t, v] <= M * m.sc_change[t, v]
        
        def sc_change_neg_rule(m, t, v):
            if t == self.n_periods - 1:
                return pyo.Constraint.Skip
            return m.sc_stage[t, v] - m.sc_stage[t+1, v] <= M * m.sc_change[t, v]
        
        model.ScChangePos = pyo.Constraint(model.T, model.SC_Set, rule=sc_change_pos_rule)
        model.ScChangeNeg = pyo.Constraint(model.T, model.SC_Set, rule=sc_change_neg_rule)
        
        # 约束3: 每个SC每日最大动作次数 - 式(6)
        def sc_max_actions_rule(m, v):
            return sum(m.sc_change[t, v] for t in range(self.n_periods - 1)) <= max_actions
        model.ScMaxActions = pyo.Constraint(model.SC_Set, rule=sc_max_actions_rule)
    
    def _add_power_balance_constraints(self, model):
        """
        添加功率平衡约束（含SC无功）
        
        重写以包含SC无功输出
        """
        network = self.network
        n_periods = self.n_periods
        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        load_factor = self.scenario_data.get("load_factor", np.ones(n_periods))
        
        sc_buses = self.sc_config["buses"]
        pv_buses = list(pv_data.keys())
        wt_buses = list(wt_data.keys())
        
        def active_power_balance_rule(m, t, j):
            """有功功率平衡"""
            # 负荷
            p_load = network.p_load_pu[j] * load_factor[t]
            
            # PV有功
            p_pv = 0
            if j in pv_buses:
                p_pv = pv_data[j][t] / network.s_base
            
            # WT有功
            p_wt = 0
            if j in wt_buses:
                p_wt = wt_data[j][t] / network.s_base
            
            # 净注入
            p_inject = p_pv + p_wt - p_load
            
            # 从父节点流入
            parent = network.parent[j]
            if parent >= 0:
                branch_idx = network.branch_idx[(parent, j)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                r = network.r_pu[branch_idx]
                p_in = m.P[t, (f, to)] - m.l[t, (f, to)] * r
            else:
                p_in = 0
            
            # 流向子节点
            p_out = 0
            for child in network.children[j]:
                branch_idx = network.branch_idx[(j, child)]
                f, to = int(network.from_bus[branch_idx]), int(network.to_bus[branch_idx])
                p_out += m.P[t, (f, to)]
            
            return p_inject == p_out - p_in
        
        def reactive_power_balance_rule(m, t, j):
            """无功功率平衡"""
            # 负荷
            q_load = network.q_load_pu[j] * load_factor[t]
            
            # SC无功（如果该节点有SC）
            q_sc = 0
            if j in sc_buses:
                sc_idx = sc_buses.index(j)
                q_sc = m.Q_sc[t, sc_idx]
            
            # PV/WT无功（第一阶段假设功率因数为1，即无无功输出）
            # 无功调节留给第二阶段
            q_pv = 0
            q_wt = 0
            
            # 净注入
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
    
    def _add_root_voltage_constraint(self, model):
        """
        添加根节点电压约束（考虑OLTC）
        
        u_root = (V0 + δ_oltc * ΔV)²
        
        线性化近似: u_root ≈ V0² + 2*V0*ΔV*δ
        """
        root = self.network.root_bus
        v0 = self.oltc_config["v0"]
        delta_v = self.oltc_config["tap_step"]
        
        def root_voltage_rule(m, t):
            # 二次项展开: (v0 + tap*dv)² = v0² + 2*v0*dv*tap + (dv*tap)²
            # 忽略高阶小量: ≈ v0² + 2*v0*dv*tap
            # 或者使用精确的二次表达式（Gurobi可以处理）
            tap = m.oltc_tap[t]
            v_actual = v0 + tap * delta_v
            # 使用线性化
            return m.u[t, root] == v0**2 + 2*v0*delta_v*tap
        
        model.RootVoltage = pyo.Constraint(model.T, rule=root_voltage_rule)
    
    def _update_model_statistics(self):
        """更新模型统计信息"""
        if self.model is None:
            return
        
        # 计算变量数
        n_vars = sum(1 for v in self.model.component_objects(pyo.Var, active=True)
                     for _ in v)
        
        # 计算约束数
        n_cons = sum(1 for c in self.model.component_objects(pyo.Constraint, active=True)
                     for _ in c)
        
        # 计算二元变量数
        n_binary = sum(1 for v in self.model.component_objects(pyo.Var, active=True)
                       for idx in v if v[idx].domain == pyo.Binary)
        
        self.statistics["n_variables"] = n_vars
        self.statistics["n_constraints"] = n_cons
        self.statistics["n_binary_vars"] = n_binary
    
    def solve(self, solver_name: str = "gurobi", **kwargs) -> Dict[str, Any]:
        """
        求解模型
        
        Args:
            solver_name: 求解器名称
            **kwargs: 求解器参数
        """
        if self.model is None:
            raise RuntimeError("模型尚未构建，请先调用build_model()")
        
        solver_config = self.config["optimization"]["solver"]
        
        # 创建求解器
        solver = pyo.SolverFactory(solver_name)
        
        # 设置求解器参数
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = solver_config.get("time_limit", 300)
            solver.options["MIPGap"] = solver_config.get("mip_gap", 1e-4)
            solver.options["OutputFlag"] = 1 if solver_config.get("verbose", True) else 0
        
        # 求解
        print(f"\n开始求解 {self.name}...")
        start_time = time.time()
        
        results = solver.solve(self.model, tee=solver_config.get("verbose", True))
        
        self.solve_time = time.time() - start_time
        self.statistics["solve_time"] = self.solve_time
        
        # 检查求解状态
        self.status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        
        self.statistics["solver_status"] = f"{self.status} ({termination})"
        
        # 检查是否找到可行解（包括最优解和时间限制内的可行解）
        has_solution = False
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            has_solution = True
            print(f"求解成功！（最优解）")
        elif results.solver.termination_condition in [pyo.TerminationCondition.maxTimeLimit,
                                                       pyo.TerminationCondition.feasible]:
            # 时间限制或找到可行解
            has_solution = True
            print(f"求解达到时间限制，但找到可行解")
        
        if has_solution:
            try:
                self.statistics["objective_value"] = pyo.value(self.model.obj)
                print(f"目标值: {self.statistics['objective_value']:.6f}")
            except:
                print(f"警告: 无法获取目标函数值")
        else:
            print(f"求解状态: {self.status}, 终止条件: {termination}")
        
        return {"status": self.status, "termination": termination}
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取优化结果
        
        Returns:
            结果字典，包含OLTC档位、SC状态、电压、网损等
        """
        if self.model is None:
            return {}
        
        model = self.model
        n_periods = self.n_periods
        network = self.network
        
        # 提取OLTC档位
        oltc_tap = np.array([pyo.value(model.oltc_tap[t]) for t in range(n_periods)])
        
        # 提取SC状态和无功
        sc_buses = self.sc_config["buses"]
        n_sc = len(sc_buses)
        sc_stage = np.zeros((n_periods, n_sc))
        sc_q = np.zeros((n_periods, n_sc))
        
        for t in range(n_periods):
            for v in range(n_sc):
                sc_stage[t, v] = pyo.value(model.sc_stage[t, v])
                sc_q[t, v] = pyo.value(model.Q_sc[t, v]) * network.s_base  # 转回MVar
        
        # 提取电压
        voltage = np.zeros((n_periods, network.n_buses))
        for t in range(n_periods):
            for j in range(network.n_buses):
                u_sq = pyo.value(model.u[t, j])
                voltage[t, j] = np.sqrt(u_sq) if u_sq > 0 else 0
        
        # 计算每个时段的网损
        loss_per_period = np.zeros(n_periods)
        for t in range(n_periods):
            loss = 0
            for l in range(network.n_branches):
                f = int(network.from_bus[l])
                to = int(network.to_bus[l])
                r = network.r_pu[l]
                loss += pyo.value(model.l[t, (f, to)]) * r
            loss_per_period[t] = loss * network.s_base * 1000  # 转为kW
        
        # 统计OLTC和SC动作次数
        oltc_actions = sum(1 for t in range(n_periods-1) 
                          if abs(oltc_tap[t+1] - oltc_tap[t]) > 0.5)
        
        sc_actions = []
        for v in range(n_sc):
            actions = sum(1 for t in range(n_periods-1) 
                         if abs(sc_stage[t+1, v] - sc_stage[t, v]) > 0.5)
            sc_actions.append(actions)
        
        results = {
            "oltc": {
                "tap": oltc_tap,
                "n_actions": oltc_actions,
            },
            "sc": {
                "buses": sc_buses,
                "stage": sc_stage,
                "q_mvar": sc_q,
                "n_actions": sc_actions,
            },
            "voltage": {
                "values": voltage,
                "min": voltage.min(),
                "max": voltage.max(),
                "mean": voltage.mean(),
            },
            "loss": {
                "per_period_kw": loss_per_period,
                "total_kw": loss_per_period.sum(),
                "average_kw": loss_per_period.mean(),
            },
            "objective": self.statistics.get("objective_value"),
        }
        
        return results


def create_day_ahead_model(config: dict) -> DayAheadVarOptModel:
    """创建日前优化模型实例"""
    return DayAheadVarOptModel(config)


if __name__ == "__main__":
    # 测试模块
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import get_config
    from data.network.ieee33 import get_ieee33_network
    from data.data_loader import get_data_loader
    
    # 加载配置和数据
    config = get_config()
    network = get_ieee33_network()
    loader = get_data_loader(config)
    
    # 获取场景数据
    scenario = loader.get_scenario_data("2024-01-01", n_periods=96)
    
    # 创建并构建模型
    model = create_day_ahead_model(config)
    model.build_model(network, scenario)
    
    print(f"模型构建完成")
    print(f"  变量数: {model.statistics['n_variables']}")
    print(f"  约束数: {model.statistics['n_constraints']}")
    print(f"  二元变量数: {model.statistics['n_binary_vars']}")
