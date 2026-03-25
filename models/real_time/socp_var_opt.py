# -*- coding: utf-8 -*-
"""
实时SOCP无功优化模型 (第二阶段)
优化PV/WT逆变器无功和SVC无功功率

输入: 第一阶段的OLTC档位和SC状态
优化变量: PV无功、WT无功、SVC无功（连续变量）
目标: 最小化网损 + 电压越限惩罚

参考论文式(17)-(21)
"""

import time
import numpy as np
import pyomo.environ as pyo
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base_model import BaseOptimizationModel
from data.network.ieee33 import IEEE33Network


class RealTimeVarOptModel(BaseOptimizationModel):
    """实时无功优化SOCP模型（第二阶段）"""
    
    def __init__(self, config: dict):
        super().__init__(name="RealTime_SOCP_VarOpt", config=config)
        
        self.network = None
        self.scenario_data = None
        self.day_ahead_results = None  # 第一阶段结果
        self.n_periods = config["optimization"]["day_ahead"]["n_periods"]  # 使用相同时段数
        
        # 设备配置
        self.pv_config = config["devices"]["pv"]
        self.wt_config = config["devices"]["wt"]
        self.svc_config = config["devices"]["svc"]
        self.oltc_config = config["devices"]["oltc"]
        self.sc_config = config["devices"]["sc"]
        
        # 网络配置
        self.v_min = config["network"]["v_min"]
        self.v_max = config["network"]["v_max"]
        
        # 第二阶段特有参数
        rt_config = config["optimization"]["real_time"]
        self.voltage_penalty = rt_config.get("voltage_penalty", 1000)
        self.loss_price = rt_config.get("loss_price", 450)
    
    def build_model(self, network: IEEE33Network, scenario_data: Dict,
                    day_ahead_results: Dict, **kwargs):
        """
        构建SOCP模型
        
        Args:
            network: IEEE33网络对象
            scenario_data: 场景数据
            day_ahead_results: 第一阶段优化结果
        """
        self.network = network
        self.scenario_data = scenario_data
        self.day_ahead_results = day_ahead_results
        
        # 创建模型
        model = pyo.ConcreteModel(name="RealTime_SOCP")
        
        # ====== 1. 基本集合和变量 ======
        self._add_basic_sets_and_variables(model)
        
        # ====== 2. DG无功变量（PV、WT）======
        self._add_dg_reactive_variables(model)
        
        # ====== 3. SVC无功变量 ======
        self._add_svc_variables(model)
        
        # ====== 4. 电压越限松弛变量 ======
        self._add_voltage_slack_variables(model)
        
        # ====== 5. 潮流约束 ======
        self._add_power_flow_constraints(model)
        
        # ====== 6. DG无功约束 ======
        self._add_dg_reactive_constraints(model)
        
        # ====== 7. SVC约束 ======
        self._add_svc_constraints(model)
        
        # ====== 8. 电压约束（软约束）======
        self._add_voltage_constraints(model)
        
        # ====== 9. 根节点电压（固定，来自第一阶段OLTC）======
        self._add_root_voltage_constraint(model)
        
        # ====== 10. 目标函数 ======
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
        
        # 支路索引集合
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
    
    def _add_dg_reactive_variables(self, model):
        """添加DG无功变量"""
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        
        n_pv = len(pv_buses)
        n_wt = len(wt_buses)
        
        # PV集合和变量
        model.PV_Set = pyo.RangeSet(0, n_pv - 1)
        model.Q_pv = pyo.Var(model.T, model.PV_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)
        
        # WT集合和变量
        model.WT_Set = pyo.RangeSet(0, n_wt - 1)
        model.Q_wt = pyo.Var(model.T, model.WT_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)
        
        # 存储配置
        model._pv_buses = pv_buses
        model._wt_buses = wt_buses
    
    def _add_svc_variables(self, model):
        """添加SVC无功变量"""
        svc_buses = self.svc_config["buses"]
        q_min = self.svc_config["q_min"] / self.network.s_base  # 转为pu
        q_max = self.svc_config["q_max"] / self.network.s_base
        
        n_svc = len(svc_buses)
        
        model.SVC_Set = pyo.RangeSet(0, n_svc - 1)
        model.Q_svc = pyo.Var(model.T, model.SVC_Set, within=pyo.Reals,
                              bounds=(q_min, q_max), initialize=0)
        
        model._svc_buses = svc_buses
    
    def _add_voltage_slack_variables(self, model):
        """添加电压越限松弛变量（用于软约束）"""
        # 欠压松弛变量
        model.v_under = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals,
                                 initialize=0)
        # 过压松弛变量
        model.v_over = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals,
                                initialize=0)
    
    def _add_power_flow_constraints(self, model):
        """添加潮流约束"""
        network = self.network
        n_periods = self.n_periods
        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        load_factor = self.scenario_data.get("load_factor", np.ones(n_periods))
        
        # 从第一阶段获取SC无功
        sc_q = self.day_ahead_results["sc"]["q_mvar"]  # (n_periods, n_sc)
        sc_buses = self.sc_config["buses"]
        
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        svc_buses = self.svc_config["buses"]
        
        def active_power_balance_rule(m, t, j):
            """有功功率平衡"""
            # 负荷
            p_load = network.p_load_pu[j] * load_factor[t]
            
            # PV有功
            p_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                p_pv = pv_data[j][t] / network.s_base
            
            # WT有功
            p_wt = 0
            if j in wt_buses:
                wt_idx = wt_buses.index(j)
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
            
            # SC无功（来自第一阶段，固定值）
            q_sc = 0
            if j in sc_buses:
                sc_idx = sc_buses.index(j)
                q_sc = sc_q[t, sc_idx] / network.s_base  # 转为pu
            
            # PV无功（优化变量）
            q_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                q_pv = m.Q_pv[t, pv_idx]
            
            # WT无功（优化变量）
            q_wt = 0
            if j in wt_buses:
                wt_idx = wt_buses.index(j)
                q_wt = m.Q_wt[t, wt_idx]
            
            # SVC无功（优化变量）
            q_svc = 0
            if j in svc_buses:
                svc_idx = svc_buses.index(j)
                q_svc = m.Q_svc[t, svc_idx]
            
            # 净无功注入
            q_inject = q_pv + q_wt + q_sc + q_svc - q_load
            
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
        
        # 添加功率平衡约束（跳过根节点）
        model.ActivePowerBalance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1),
            rule=active_power_balance_rule
        )
        
        model.ReactivePowerBalance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1),
            rule=reactive_power_balance_rule
        )
        
        # 电压降落约束
        def voltage_drop_rule(m, t, f, to):
            branch_idx = network.branch_idx[(f, to)]
            r = network.r_pu[branch_idx]
            x = network.x_pu[branch_idx]
            return m.u[t, to] == m.u[t, f] - 2*(r*m.P[t,(f,to)] + x*m.Q[t,(f,to)]) + (r**2 + x**2)*m.l[t,(f,to)]
        
        model.VoltageDrop = pyo.Constraint(model.T, model.Branch, rule=voltage_drop_rule)
        
        # 二阶锥约束
        def soc_rule(m, t, f, to):
            return m.P[t,(f,to)]**2 + m.Q[t,(f,to)]**2 <= m.l[t,(f,to)] * m.u[t, to]
        
        model.SOC = pyo.Constraint(model.T, model.Branch, rule=soc_rule)
    
    def _add_dg_reactive_constraints(self, model):
        """
        添加DG无功约束 - 式(17)(18)
        
        |Q_DG| <= sqrt(S_DG^2 - P_DG^2)
        """
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        pv_capacity = self.pv_config["capacity"]  # MW
        wt_capacity = self.wt_config["capacity"]
        
        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        
        # PV无功约束
        def pv_q_upper_rule(m, t, i):
            bus = pv_buses[i]
            p_pv = pv_data[bus][t]  # MW
            s_pv = pv_capacity[i]   # MW (视在功率容量)
            
            # Q_max = sqrt(S^2 - P^2), 转为pu
            if s_pv > p_pv:
                q_max = np.sqrt(s_pv**2 - p_pv**2) / self.network.s_base
            else:
                q_max = 0
            
            return m.Q_pv[t, i] <= q_max
        
        def pv_q_lower_rule(m, t, i):
            bus = pv_buses[i]
            p_pv = pv_data[bus][t]
            s_pv = pv_capacity[i]
            
            if s_pv > p_pv:
                q_max = np.sqrt(s_pv**2 - p_pv**2) / self.network.s_base
            else:
                q_max = 0
            
            return m.Q_pv[t, i] >= -q_max
        
        model.PV_Q_Upper = pyo.Constraint(model.T, model.PV_Set, rule=pv_q_upper_rule)
        model.PV_Q_Lower = pyo.Constraint(model.T, model.PV_Set, rule=pv_q_lower_rule)
        
        # WT无功约束
        def wt_q_upper_rule(m, t, i):
            bus = wt_buses[i]
            p_wt = wt_data[bus][t]
            s_wt = wt_capacity[i]
            
            if s_wt > p_wt:
                q_max = np.sqrt(s_wt**2 - p_wt**2) / self.network.s_base
            else:
                q_max = 0
            
            return m.Q_wt[t, i] <= q_max
        
        def wt_q_lower_rule(m, t, i):
            bus = wt_buses[i]
            p_wt = wt_data[bus][t]
            s_wt = wt_capacity[i]
            
            if s_wt > p_wt:
                q_max = np.sqrt(s_wt**2 - p_wt**2) / self.network.s_base
            else:
                q_max = 0
            
            return m.Q_wt[t, i] >= -q_max
        
        model.WT_Q_Upper = pyo.Constraint(model.T, model.WT_Set, rule=wt_q_upper_rule)
        model.WT_Q_Lower = pyo.Constraint(model.T, model.WT_Set, rule=wt_q_lower_rule)
    
    def _add_svc_constraints(self, model):
        """SVC约束已在变量定义时通过bounds设置"""
        pass
    
    def _add_voltage_constraints(self, model):
        """
        添加电压约束（软约束）
        
        V_min^2 - v_under <= u <= V_max^2 + v_over
        """
        u_min_sq = self.v_min ** 2
        u_max_sq = self.v_max ** 2
        
        def voltage_lower_rule(m, t, j):
            return m.u[t, j] >= u_min_sq - m.v_under[t, j]
        
        def voltage_upper_rule(m, t, j):
            return m.u[t, j] <= u_max_sq + m.v_over[t, j]
        
        model.VoltageLower = pyo.Constraint(model.T, model.N, rule=voltage_lower_rule)
        model.VoltageUpper = pyo.Constraint(model.T, model.N, rule=voltage_upper_rule)
    
    def _add_root_voltage_constraint(self, model):
        """
        添加根节点电压约束（来自第一阶段OLTC结果）
        """
        root = self.network.root_bus
        v0 = self.oltc_config["v0"]
        delta_v = self.oltc_config["tap_step"]
        
        # 从第一阶段获取OLTC档位
        oltc_tap = self.day_ahead_results["oltc"]["tap"]
        
        def root_voltage_rule(m, t):
            tap = oltc_tap[t]
            v_tap = v0 + tap * delta_v
            return m.u[t, root] == v_tap ** 2
        
        model.RootVoltage = pyo.Constraint(model.T, rule=root_voltage_rule)
    
    def _add_objective(self, model):
        """
        添加目标函数 - 式(19)(20)(21)
        
        Min F = C_loss * P_loss + C_beta * V_r
        """
        network = self.network
        
        # 网损项
        loss_expr = 0
        for t in model.T:
            for l in range(network.n_branches):
                f = int(network.from_bus[l])
                to = int(network.to_bus[l])
                r = network.r_pu[l]
                loss_expr += model.l[t, (f, to)] * r
        
        # 电压越限惩罚项 - 式(20)
        # 使用松弛变量的平方和
        voltage_penalty_expr = 0
        for t in model.T:
            for j in model.N:
                voltage_penalty_expr += model.v_under[t, j] + model.v_over[t, j]
        
        # 总目标（转换为实际单位会在结果分析时做）
        # 这里简化处理，直接用pu值
        model.obj = pyo.Objective(
            expr=loss_expr + self.voltage_penalty * voltage_penalty_expr,
            sense=pyo.minimize
        )
    
    def _update_model_statistics(self):
        """更新模型统计信息"""
        if self.model is None:
            return
        
        n_vars = sum(1 for v in self.model.component_objects(pyo.Var, active=True)
                     for _ in v)
        
        n_cons = sum(1 for c in self.model.component_objects(pyo.Constraint, active=True)
                     for _ in c)
        
        # SOCP模型没有二元变量
        n_binary = 0
        
        self.statistics["n_variables"] = n_vars
        self.statistics["n_constraints"] = n_cons
        self.statistics["n_binary_vars"] = n_binary
    
    def solve(self, solver_name: str = "gurobi", **kwargs) -> Dict[str, Any]:
        """求解模型"""
        if self.model is None:
            raise RuntimeError("模型尚未构建，请先调用build_model()")
        
        solver_config = self.config["optimization"]["solver"]
        
        solver = pyo.SolverFactory(solver_name)
        
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = solver_config.get("time_limit", 300)
            solver.options["MIPGap"] = solver_config.get("mip_gap", 1e-4)
            solver.options["OutputFlag"] = 1 if solver_config.get("verbose", True) else 0
        
        print(f"\n开始求解 {self.name}...")
        start_time = time.time()
        
        results = solver.solve(self.model, tee=solver_config.get("verbose", True))
        
        self.solve_time = time.time() - start_time
        self.statistics["solve_time"] = self.solve_time
        
        self.status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        
        self.statistics["solver_status"] = f"{self.status} ({termination})"
        
        # 检查是否找到可行解
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
                print(f"目标值: {self.statistics['objective_value']:.6f}")
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
        
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        svc_buses = self.svc_config["buses"]
        
        # 提取PV无功
        n_pv = len(pv_buses)
        pv_q = np.zeros((n_periods, n_pv))
        for t in range(n_periods):
            for i in range(n_pv):
                pv_q[t, i] = pyo.value(model.Q_pv[t, i]) * network.s_base  # 转回MVar
        
        # 提取WT无功
        n_wt = len(wt_buses)
        wt_q = np.zeros((n_periods, n_wt))
        for t in range(n_periods):
            for i in range(n_wt):
                wt_q[t, i] = pyo.value(model.Q_wt[t, i]) * network.s_base
        
        # 提取SVC无功
        n_svc = len(svc_buses)
        svc_q = np.zeros((n_periods, n_svc))
        for t in range(n_periods):
            for i in range(n_svc):
                svc_q[t, i] = pyo.value(model.Q_svc[t, i]) * network.s_base
        
        # 提取电压
        voltage = np.zeros((n_periods, network.n_buses))
        for t in range(n_periods):
            for j in range(network.n_buses):
                u_sq = pyo.value(model.u[t, j])
                voltage[t, j] = np.sqrt(u_sq) if u_sq > 0 else 0
        
        # 计算网损
        loss_per_period = np.zeros(n_periods)
        for t in range(n_periods):
            loss = 0
            for l in range(network.n_branches):
                f = int(network.from_bus[l])
                to = int(network.to_bus[l])
                r = network.r_pu[l]
                loss += pyo.value(model.l[t, (f, to)]) * r
            loss_per_period[t] = loss * network.s_base * 1000  # 转为kW
        
        # 计算电压越限量
        v_violation = np.zeros((n_periods, network.n_buses))
        for t in range(n_periods):
            for j in range(network.n_buses):
                v_under = pyo.value(model.v_under[t, j])
                v_over = pyo.value(model.v_over[t, j])
                v_violation[t, j] = v_under + v_over
        
        results = {
            "pv_reactive": {
                "buses": pv_buses,
                "q_mvar": pv_q,
            },
            "wt_reactive": {
                "buses": wt_buses,
                "q_mvar": wt_q,
            },
            "svc_reactive": {
                "buses": svc_buses,
                "q_mvar": svc_q,
            },
            "voltage": {
                "values": voltage,
                "min": voltage.min(),
                "max": voltage.max(),
                "mean": voltage.mean(),
                "violation": v_violation,
                "total_violation": v_violation.sum(),
            },
            "loss": {
                "per_period_kw": loss_per_period,
                "total_kw": loss_per_period.sum(),
                "average_kw": loss_per_period.mean(),
            },
            "objective": self.statistics.get("objective_value"),
        }
        
        return results


def create_real_time_model(config: dict) -> RealTimeVarOptModel:
    """创建实时优化模型实例"""
    return RealTimeVarOptModel(config)
