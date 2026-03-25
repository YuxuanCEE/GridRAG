# -*- coding: utf-8 -*-
"""
Task C: 综合优化模型 (Joint Optimization) - 重写版
融合 VVC + ED + EV调度的多目标MISOCP优化

重写说明:
    严格对齐 socp_ed.py 的单位体系 —— 所有设备决策变量(P_grid, P_ch, P_dis,
    P_cut, Q_pv, Q_wt, Q_svc, P_tie, Q_tie, P_ev)均采用 **MW/MVar 实际值**;
    仅 DistFlow 内部变量 (P[branch], Q[branch], l[branch], u) 使用 pu。
    潮流节点功率平衡中, 设备功率通过 / s_base 转为 pu 注入; 目标函数中
    设备功率直接用 MW 参与成本计算, 消除了旧版 "pu 再 × s_base" 的往返转换。

    在 ED 已验证的框架上, 仅额外增加:
      - OLTC 离散档位 + 根节点电压联动
      - SC / SVC 无功补偿
      - 电压软约束
      - EV 可切负荷 + 阶梯惩罚 + 中断检测
    且上述新增部分采用与 ED 一致的单位约定。

参考文献:
[1] 陈海鹏《考虑电动汽车无功补偿与不确定性的配电网-电动汽车有功无功协同优化》
[2] 祁向龙《多时间尺度协同的配电网分层深度强化学习电压控制策略》
[3] emobpy: An open tool for creating battery electric vehicle time series

作者: GridRAG Team
日期: 2025-01
"""

import numpy as np
import pyomo.environ as pyo
from typing import Dict, Any, Optional, List
import time
import pandas as pd


class JointOptModel:
    """
    综合优化模型 (重写版, 单位体系与 socp_ed.py 对齐)

    单位约定:
      - 设备功率变量: MW / MVar (与 ED 一致)
      - DistFlow 内部: pu
      - SOC / 能量: MWh
    """

    def __init__(self, config: dict):
        self.config = config
        self.network_config = config["network"]
        self.device_config = config["devices"]
        self.opt_config = config["optimization"]["joint"]
        self.price_config = config["price"]

        # 设备子配置
        self.oltc_config = self.device_config["oltc"]
        self.sc_config = self.device_config["sc"]
        self.svc_config = self.device_config.get("svc", {"buses": []})
        self.ess_config = self.device_config["ess"]
        self.pv_config = self.device_config["pv"]
        self.wt_config = self.device_config["wt"]
        self.tie_config = self.device_config.get("tie_switches", {"enabled": False})
        self.ev_config = self.device_config.get("ev_stations", {"enabled": False})

        # 时间参数
        self.n_periods = self.opt_config.get("n_periods", 96)
        self.delta_t = self.opt_config.get("delta_t", 0.25)

        # 网络参数
        self.s_base = self.network_config["s_base"]
        self.v_min = self.network_config["v_min"]
        self.v_max = self.network_config["v_max"]

        # 控制开关
        self.use_oltc = self.opt_config.get("use_oltc", True)
        self.use_sc = self.opt_config.get("use_sc", True)
        self.use_svc = self.opt_config.get("use_svc", True)
        self.use_ess = self.opt_config.get("use_ess", True)
        self.use_tie = (self.opt_config.get("use_tie_switches", True)
                        and self.tie_config.get("enabled", False))
        self.use_ev = (self.opt_config.get("use_ev", True)
                       and self.ev_config.get("enabled", False))
        self.allow_curtailment = self.opt_config.get("allow_pv_curtailment", True)

        # 成本/惩罚系数
        self.voltage_soft = self.opt_config.get("voltage_soft_constraint", True)
        self.voltage_penalty_coef = self.opt_config.get("voltage_penalty_coef", 1000)
        self.curtailment_cost = self.opt_config.get("pv_curtailment_cost", 700)
        self.loss_cost = self.opt_config.get("loss_cost", 500)

        # 权重（峰/平/谷偏好, 仅用于偏好项）
        self.weights = self.opt_config.get("weights", {
            "loss": [0.25, 0.25, 0.25],
            "voltage": [0.25, 0.25, 0.25],
            "renewable": [0.25, 0.25, 0.25],
            "cost": [0.25, 0.25, 0.25],
            "ev_satisfaction": [0.2, 0.2, 0.2],
        })

        # 电价 / 场景 / EV 数据
        self.price_data = None
        self.period_type = None
        self.scenario_data = {}
        self.ev_data = {}

        # 网络和模型
        self.network = None
        self.model = None
        self.statistics = {}

    # ------------------------------------------------------------------ #
    #  数据加载
    # ------------------------------------------------------------------ #

    def load_network(self, network):
        self.network = network
        print(f"[Joint] 已加载网络: {network.n_buses}节点, "
              f"{network.n_branches}支路, "
              f"{network.n_tie_switches}联络线, "
              f"{network.n_ev_stations}充电站")

    def load_scenario_data(self, scenario_data: dict):
        self.scenario_data = scenario_data
        print(f"[Joint] 已加载场景数据: {len(scenario_data)}个数据项")

    def load_ev_data(self, ev_data: dict):
        self.ev_data = ev_data
        print(f"[Joint] 已加载EV数据: {len(ev_data)}个充电站")

    def load_price_data(self):
        peak_hours = self.price_config["peak_hours"]
        valley_hours = self.price_config["valley_hours"]
        price_data = np.zeros(self.n_periods)
        period_type = np.zeros(self.n_periods, dtype=int)
        for t in range(self.n_periods):
            hour = (t * 15 // 60) + 1
            if hour in peak_hours:
                price_data[t] = self.price_config["peak_price"]
                period_type[t] = 2
            elif hour in valley_hours:
                price_data[t] = self.price_config["valley_price"]
                period_type[t] = 0
            else:
                price_data[t] = self.price_config["flat_price"]
                period_type[t] = 1
        self.price_data = price_data
        self.period_type = period_type
        print(f"[Joint] 已加载电价数据: 峰{self.price_config['peak_price']}元/MWh")

    def get_period_weight(self, t: int, weight_name: str) -> float:
        weights = self.weights.get(weight_name, [0.25, 0.25, 0.25])
        pt = self.period_type[t]
        return {0: weights[2], 1: weights[1], 2: weights[0]}[pt]

    # ================================================================== #
    #  构建优化模型
    # ================================================================== #

    def build_model(self):
        print("\n" + "=" * 60)
        print("构建Task C综合优化模型")
        print("=" * 60)

        if self.network is None:
            raise ValueError("请先加载网络模型!")
        if self.price_data is None:
            self.load_price_data()

        model = pyo.ConcreteModel(name="JointOptimization")

        # 基本集合
        model.T = pyo.RangeSet(0, self.n_periods - 1)
        model.N = pyo.RangeSet(0, self.network.n_buses - 1)
        model.L = pyo.RangeSet(0, self.network.n_branches - 1)

        # 构建支路集合 (与 ED 一致)
        branch_set = []
        for l in range(self.network.n_branches):
            f = int(self.network.from_bus[l])
            to = int(self.network.to_bus[l])
            branch_set.append((f, to))
        model.Branch = pyo.Set(initialize=branch_set)

        # ====== 变量 ======
        print("\n添加变量...")
        self._add_distflow_variables(model)
        self._add_grid_variables(model)
        self._add_oltc_variables(model)
        self._add_sc_variables(model)
        self._add_svc_variables(model)
        self._add_ess_variables(model)
        self._add_dg_variables(model)
        if self.use_tie:
            self._add_tie_switch_variables(model)
        if self.use_ev:
            self._add_ev_variables(model)
        if self.voltage_soft:
            self._add_voltage_slack_variables(model)

        # ====== 约束 ======
        print("\n添加约束...")
        self._add_power_flow_constraints(model)
        self._add_voltage_constraints(model)
        self._add_root_voltage_constraint(model)
        self._add_oltc_constraints(model)
        self._add_sc_constraints(model)
        self._add_svc_constraints(model)
        self._add_ess_constraints(model)
        self._add_dg_constraints(model)
        if self.use_tie:
            self._add_tie_switch_constraints(model)
            self._add_radiality_constraints(model)
        if self.use_ev:
            self._add_ev_constraints(model)

        # ====== 目标函数 ======
        print("\n添加目标函数...")
        self._add_objective(model)

        self.model = model

        # 统计
        n_vars = len(list(model.component_objects(pyo.Var)))
        n_cons = len(list(model.component_objects(pyo.Constraint)))
        n_binary = sum(1 for v in model.component_data_objects(pyo.Var) if v.domain == pyo.Binary)
        self.statistics.update(n_variables=n_vars, n_constraints=n_cons, n_binary_vars=n_binary)
        print(f"\n模型构建完成:")
        print(f"  变量组数: {n_vars}")
        print(f"  约束组数: {n_cons}")
        print(f"  二进制变量数: {n_binary}")
        return model

    # ================================================================== #
    #  变量定义 —— DistFlow 内部变量 (pu)
    # ================================================================== #

    def _add_distflow_variables(self, model):
        """DistFlow 潮流变量 (与 ED _add_basic_sets_and_variables 一致)"""
        # 有功功率流 (pu)
        model.P = pyo.Var(model.T, model.Branch, within=pyo.Reals,
                          bounds=(-10, 10), initialize=0)
        # 无功功率流 (pu)
        model.Q = pyo.Var(model.T, model.Branch, within=pyo.Reals,
                          bounds=(-10, 10), initialize=0)
        # 电流幅值平方 (pu)
        model.l = pyo.Var(model.T, model.Branch, within=pyo.NonNegativeReals,
                          bounds=(0, 100), initialize=0)
        # 电压幅值平方 (pu)
        model.u = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals,
                          bounds=(0.8, 1.2), initialize=1.0)
        print(f"  DistFlow变量: P,Q,l {self.n_periods}×{self.network.n_branches}, "
              f"u {self.n_periods}×{self.network.n_buses}")

    # ================================================================== #
    #  变量定义 —— 设备变量 (MW/MVar, 与 ED 一致)
    # ================================================================== #

    def _add_grid_variables(self, model):
        """购电功率 (MW) —— 与 ED._add_grid_variables 一致"""
        model.P_grid = pyo.Var(model.T, within=pyo.Reals,
                               bounds=(-5, 10), initialize=1)
        print(f"  购电变量: P_grid {self.n_periods}时段 (MW)")

    def _add_oltc_variables(self, model):
        if not self.use_oltc:
            return
        oltc = self.oltc_config
        n_taps = oltc["tap_max"] - oltc["tap_min"] + 1
        model.OLTC_Taps = pyo.RangeSet(0, n_taps - 1)
        model.oltc_tap = pyo.Var(model.T, model.OLTC_Taps, within=pyo.Binary, initialize=0)
        model.oltc_change = pyo.Var(model.T, within=pyo.Binary, initialize=0)
        model._oltc_n_taps = n_taps
        model._oltc_tap_min = oltc["tap_min"]
        model._oltc_tap_step = oltc["tap_step"]
        model._oltc_v0 = oltc["v0"]
        model._oltc_max_actions = oltc["max_daily_actions"]
        print(f"  OLTC变量: {n_taps}档位 × {self.n_periods}时段 = {n_taps * self.n_periods}个二进制")

    def _add_sc_variables(self, model):
        if not self.use_sc:
            return
        sc = self.sc_config
        n_sc = len(sc["buses"])
        n_stages = sc["n_stages"]
        model.SC_Set = pyo.RangeSet(0, n_sc - 1)
        model.sc_stage = pyo.Var(model.T, model.SC_Set,
                                  within=pyo.NonNegativeIntegers,
                                  bounds=(0, n_stages), initialize=0)
        model.sc_change = pyo.Var(model.T, model.SC_Set, within=pyo.Binary, initialize=0)
        model._sc_buses = sc["buses"]
        model._sc_n_stages = n_stages
        model._sc_q_per_stage = sc["q_per_stage"]  # MVar
        model._sc_max_actions = sc["max_daily_actions"]
        print(f"  SC变量: {n_sc}组 × {self.n_periods}时段")

    def _add_svc_variables(self, model):
        if not self.use_svc:
            return
        svc = self.svc_config
        n_svc = len(svc.get("buses", []))
        if n_svc == 0:
            return
        model.SVC_Set = pyo.RangeSet(0, n_svc - 1)
        q_min = svc.get("q_min", -0.4)  # MVar
        q_max = svc.get("q_max", 0.4)
        model.Q_svc = pyo.Var(model.T, model.SVC_Set, within=pyo.Reals,
                              bounds=(q_min, q_max), initialize=0)
        model._svc_buses = svc["buses"]
        print(f"  SVC变量: {n_svc}台 × {self.n_periods}时段 (MVar)")

    def _add_ess_variables(self, model):
        """储能变量 (MW, MWh) —— 与 ED 一致"""
        if not self.use_ess:
            return
        ess = self.ess_config
        n_ess = len(ess["buses"])
        capacity = ess["capacity_mwh"]
        max_ch_rate = ess["max_charge_rate"]
        max_dis_rate = ess["max_discharge_rate"]
        soc_min = ess["soc_min"]
        soc_max = ess["soc_max"]

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

        # 充放电互斥
        ess_mutex = ess.get("charge_discharge_mutex", True)
        if ess_mutex:
            model.ess_mode = pyo.Var(model.T, model.ESS_Set, within=pyo.Binary, initialize=0)

        model._ess_buses = ess["buses"]
        model._ess_capacity = capacity
        model._ess_max_ch = [capacity[k] * max_ch_rate for k in range(n_ess)]
        model._ess_max_dis = [capacity[k] * max_dis_rate for k in range(n_ess)]
        model._ess_mutex = ess_mutex
        print(f"  ESS变量: {n_ess}台 × {self.n_periods}时段 (MW)" +
              (f" + {n_ess * self.n_periods}互斥二进制" if ess_mutex else ""))

    def _add_dg_variables(self, model):
        """DG变量 (MW / MVar) —— 与 ED 一致"""
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        pv_capacity = self.pv_config["capacity"]
        n_pv = len(pv_buses)
        n_wt = len(wt_buses)

        model.PV_Set = pyo.RangeSet(0, n_pv - 1)
        model.WT_Set = pyo.RangeSet(0, n_wt - 1)

        # 弃光功率 (MW)
        if self.allow_curtailment:
            def p_cut_bounds(m, t, j):
                return (0, pv_capacity[j])
            model.P_cut = pyo.Var(model.T, model.PV_Set, within=pyo.NonNegativeReals,
                                  bounds=p_cut_bounds, initialize=0)

        # PV 无功 (MVar)
        model.Q_pv = pyo.Var(model.T, model.PV_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)
        # WT 无功 (MVar)
        model.Q_wt = pyo.Var(model.T, model.WT_Set, within=pyo.Reals,
                             bounds=(-1, 1), initialize=0)

        model._pv_buses = pv_buses
        model._wt_buses = wt_buses
        model._pv_capacity = pv_capacity
        print(f"  DG变量: {n_pv}PV + {n_wt}WT × {self.n_periods}时段 (MW/MVar)")

    def _add_tie_switch_variables(self, model):
        """联络线变量 —— 与 ED 一致 (P_tie/Q_tie/l_tie 不额外除 s_base)"""
        network = self.network
        n_ties = network.n_tie_switches
        if n_ties == 0:
            return
        model.TieSwitch_Set = pyo.RangeSet(0, n_ties - 1)
        model.sw_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Binary, initialize=0)
        model.sw_change = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Binary, initialize=0)
        model.P_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Reals,
                              bounds=(-10, 10), initialize=0)
        model.Q_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Reals,
                              bounds=(-10, 10), initialize=0)
        model.l_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.NonNegativeReals,
                              bounds=(0, 100), initialize=0)
        model._tie_switches = network.tie_switches
        model._n_ties = n_ties
        model._tie_max_actions = self.tie_config.get("max_daily_actions", 6)
        print(f"  联络线变量: {n_ties}条 × {self.n_periods}时段 = {n_ties * self.n_periods * 2}二进制")

    def _add_ev_variables(self, model):
        """EV充电站变量 —— P_ev (MW), 可切负荷"""
        network = self.network
        n_ev = network.n_ev_stations
        if n_ev == 0:
            return
        model.EV_Set = pyo.RangeSet(0, n_ev - 1)

        # 实际充电功率 (MW)
        def p_ev_bounds(m, t, k):
            return (0, network.ev_stations[k]["max_power_mw"])
        model.P_ev = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                             bounds=p_ev_bounds, initialize=0)

        # 负荷削减比例
        max_ev_cut = self.ev_config.get("max_cut_ratio", 0.8)
        model.ev_cut_ratio = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                                      bounds=(0, max_ev_cut), initialize=0)
        # 阶梯削减
        model.ev_cut_tier1 = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                                      bounds=(0, 0.30), initialize=0)
        model.ev_cut_tier2 = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                                      bounds=(0, 0.30), initialize=0)
        model.ev_cut_tier3 = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                                      bounds=(0, 0.20), initialize=0)

        # 聚合能量 (MWh)
        def e_ev_bounds(m, t, k):
            return (0, network.ev_stations[k]["capacity_mwh"])
        model.E_ev = pyo.Var(model.T, model.EV_Set, within=pyo.NonNegativeReals,
                             bounds=e_ev_bounds, initialize=0)

        # 能量缺口松弛
        model.E_ev_shortage = pyo.Var(model.EV_Set, within=pyo.NonNegativeReals, initialize=0)

        # 中断检测
        model.z_int = pyo.Var(model.T, model.EV_Set, within=pyo.Binary, initialize=0)
        model.ev_charging = pyo.Var(model.T, model.EV_Set, within=pyo.Binary, initialize=0)

        model._ev_stations = network.ev_stations
        model._n_ev = n_ev
        model._max_ev_cut_ratio = max_ev_cut
        print(f"  EV变量: {n_ev}站 × {self.n_periods}时段 (MW), "
              f"可切负荷(最大{int(max_ev_cut*100)}%) + 阶梯惩罚 + 中断检测")

    def _add_voltage_slack_variables(self, model):
        model.v_over = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals, initialize=0)
        model.v_under = pyo.Var(model.T, model.N, within=pyo.NonNegativeReals, initialize=0)
        print(f"  电压松弛变量: {self.n_periods}×{self.network.n_buses}×2")

    # ================================================================== #
    #  约束定义
    # ================================================================== #

    def _add_power_flow_constraints(self, model):
        """
        DistFlow 潮流约束 —— 完全复刻 socp_ed._add_power_flow_constraints()

        关键: 设备功率 (MW) / s_base → pu 写入节点注入;
              DistFlow 方程全部 pu.
        """
        network = self.network
        s_base = self.s_base

        pv_data = self.scenario_data.get("pv", {})
        wt_data = self.scenario_data.get("wt", {})
        load_factor = self.scenario_data.get("load_factor", np.ones(self.n_periods))

        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        ess_buses = self.ess_config["buses"]
        sc_buses = self.sc_config["buses"] if self.use_sc else []
        svc_buses = self.svc_config.get("buses", []) if self.use_svc else []
        ev_buses = [s["bus"] for s in network.ev_stations] if self.use_ev else []

        tie_endpoints = set()
        if self.use_tie and network.n_tie_switches > 0:
            for tie in network.tie_switches:
                tie_endpoints.add(tie["from"])
                tie_endpoints.add(tie["to"])

        # ---- 有功功率平衡 (非根节点) ----
        def p_balance_rule(m, t, j):
            p_load = network.p_load_pu[j] * load_factor[t]

            # PV (MW → pu)
            p_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                p_pv_raw = pv_data.get(j, np.zeros(self.n_periods))[t] / s_base
                if self.allow_curtailment:
                    p_pv = p_pv_raw - m.P_cut[t, pv_idx] / s_base
                else:
                    p_pv = p_pv_raw

            # WT (MW → pu)
            p_wt = 0
            if j in wt_buses:
                p_wt = wt_data.get(j, np.zeros(self.n_periods))[t] / s_base

            # ESS (MW → pu)
            p_ess = 0
            if j in ess_buses:
                ess_idx = ess_buses.index(j)
                p_ess = (m.P_dis[t, ess_idx] - m.P_ch[t, ess_idx]) / s_base

            p_inject = p_pv + p_wt + p_ess - p_load

            # EV 负荷 (MW → pu)
            if j in ev_buses and self.use_ev:
                ev_idx = ev_buses.index(j)
                p_inject -= m.P_ev[t, ev_idx] / s_base

            # 父节点流入
            parent = network.parent[j]
            if parent >= 0:
                br = network.branch_idx[(parent, j)]
                f, to = int(network.from_bus[br]), int(network.to_bus[br])
                r = network.r_pu[br]
                p_in = m.P[t, (f, to)] - m.l[t, (f, to)] * r
            else:
                p_in = 0

            # 子节点流出
            p_out = 0
            for child in network.children[j]:
                br = network.branch_idx[(j, child)]
                f, to = int(network.from_bus[br]), int(network.to_bus[br])
                p_out += m.P[t, (f, to)]

            # 联络线贡献 (P_tie 与 ED 同维度)
            p_tie = 0
            if self.use_tie and network.n_tie_switches > 0 and j in tie_endpoints:
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        p_tie -= m.P_tie[t, k] / s_base
                    elif j == tie["to"]:
                        r_tie = tie["r_pu"]
                        p_tie += (m.P_tie[t, k] - m.l_tie[t, k] * r_tie) / s_base

            return p_inject == p_out - p_in + p_tie

        model.P_Balance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1), rule=p_balance_rule)

        # ---- 无功功率平衡 (非根节点) ----
        def q_balance_rule(m, t, j):
            q_load = network.q_load_pu[j] * load_factor[t]

            q_pv = 0
            if j in pv_buses:
                pv_idx = pv_buses.index(j)
                q_pv = m.Q_pv[t, pv_idx] / s_base

            q_wt = 0
            if j in wt_buses:
                wt_idx = wt_buses.index(j)
                q_wt = m.Q_wt[t, wt_idx] / s_base

            # SC (MVar → pu)
            q_sc = 0
            if j in sc_buses and self.use_sc:
                sc_idx = sc_buses.index(j)
                q_per_stage = model._sc_q_per_stage  # MVar
                q_sc = m.sc_stage[t, sc_idx] * q_per_stage / s_base

            # SVC (MVar → pu)
            q_svc = 0
            if j in svc_buses and self.use_svc:
                svc_idx = svc_buses.index(j)
                q_svc = m.Q_svc[t, svc_idx] / s_base

            q_inject = q_pv + q_wt + q_sc + q_svc - q_load

            parent = network.parent[j]
            if parent >= 0:
                br = network.branch_idx[(parent, j)]
                f, to = int(network.from_bus[br]), int(network.to_bus[br])
                x = network.x_pu[br]
                q_in = m.Q[t, (f, to)] - m.l[t, (f, to)] * x
            else:
                q_in = 0

            q_out = 0
            for child in network.children[j]:
                br = network.branch_idx[(j, child)]
                f, to = int(network.from_bus[br]), int(network.to_bus[br])
                q_out += m.Q[t, (f, to)]

            q_tie = 0
            if self.use_tie and network.n_tie_switches > 0 and j in tie_endpoints:
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        q_tie -= m.Q_tie[t, k] / s_base
                    elif j == tie["to"]:
                        x_tie = tie["x_pu"]
                        q_tie += (m.Q_tie[t, k] - m.l_tie[t, k] * x_tie) / s_base

            return q_inject == q_out - q_in + q_tie

        model.Q_Balance = pyo.Constraint(
            model.T, pyo.RangeSet(1, network.n_buses - 1), rule=q_balance_rule)

        # ---- 根节点购电定义 (与 ED 第452-461行一致) ----
        def root_power_rule(m, t):
            root = network.root_bus
            p_out = 0
            for child in network.children[root]:
                br = network.branch_idx[(root, child)]
                f, to = int(network.from_bus[br]), int(network.to_bus[br])
                p_out += m.P[t, (f, to)]
            return m.P_grid[t] == p_out * s_base   # pu → MW
        model.RootPower = pyo.Constraint(model.T, rule=root_power_rule)

        # ---- 电压降落 ----
        def voltage_drop_rule(m, t, f, to):
            br = network.branch_idx[(f, to)]
            r = network.r_pu[br]
            x = network.x_pu[br]
            return m.u[t, to] == m.u[t, f] - 2*(r*m.P[t,(f,to)] + x*m.Q[t,(f,to)]) \
                   + (r**2 + x**2)*m.l[t,(f,to)]
        model.Voltage_Drop = pyo.Constraint(model.T, model.Branch, rule=voltage_drop_rule)

        # ---- SOCP 松弛 ----
        def socp_rule(m, t, f, to):
            return m.P[t,(f,to)]**2 + m.Q[t,(f,to)]**2 <= m.l[t,(f,to)] * m.u[t, f]
        model.SOCP = pyo.Constraint(model.T, model.Branch, rule=socp_rule)

        print(f"  潮流约束: P平衡{self.n_periods}×{network.n_buses-1}, "
              f"Q平衡{self.n_periods}×{network.n_buses-1}")
        print(f"  根节点(bus {network.root_bus})下游: children={network.children[network.root_bus]}")

    def _add_voltage_constraints(self, model):
        if self.voltage_soft:
            def v_upper(m, t, j):
                return m.u[t, j] <= self.v_max**2 + m.v_over[t, j]
            def v_lower(m, t, j):
                return m.u[t, j] >= self.v_min**2 - m.v_under[t, j]
            model.V_Upper_Soft = pyo.Constraint(model.T, model.N, rule=v_upper)
            model.V_Lower_Soft = pyo.Constraint(model.T, model.N, rule=v_lower)
            print(f"  电压软约束: {self.n_periods}×{self.network.n_buses}×2")
        else:
            u_min_sq = self.v_min ** 2
            u_max_sq = self.v_max ** 2
            def voltage_limit_rule(m, t, j):
                return (u_min_sq, m.u[t, j], u_max_sq)
            model.VoltageLimits = pyo.Constraint(model.T, model.N, rule=voltage_limit_rule)
            print(f"  电压硬约束: {self.n_periods}×{self.network.n_buses}")

    def _add_root_voltage_constraint(self, model):
        """根节点电压: 如果使用OLTC则由OLTC确定, 否则固定为1.0"""
        if self.use_oltc:
            return  # OLTC 约束中处理
        root = self.network.root_bus
        def root_v_rule(m, t):
            return m.u[t, root] == 1.0
        model.RootVoltage = pyo.Constraint(model.T, rule=root_v_rule)

    def _add_oltc_constraints(self, model):
        if not self.use_oltc:
            return
        n_taps = model._oltc_n_taps
        tap_min = model._oltc_tap_min
        tap_step = model._oltc_tap_step
        v0 = model._oltc_v0
        max_actions = model._oltc_max_actions

        def single_tap(m, t):
            return sum(m.oltc_tap[t, k] for k in m.OLTC_Taps) == 1
        model.OLTC_SingleTap = pyo.Constraint(model.T, rule=single_tap)

        def oltc_voltage(m, t):
            v_sq = sum(m.oltc_tap[t, k] * (v0 + (tap_min + k) * tap_step)**2
                       for k in m.OLTC_Taps)
            return m.u[t, 0] == v_sq
        model.OLTC_Voltage = pyo.Constraint(model.T, rule=oltc_voltage)

        def oltc_chg_u(m, t, k):
            if t == 0: return pyo.Constraint.Skip
            return m.oltc_change[t] >= m.oltc_tap[t, k] - m.oltc_tap[t-1, k]
        def oltc_chg_l(m, t, k):
            if t == 0: return pyo.Constraint.Skip
            return m.oltc_change[t] >= m.oltc_tap[t-1, k] - m.oltc_tap[t, k]
        model.OLTC_Change_U = pyo.Constraint(model.T, model.OLTC_Taps, rule=oltc_chg_u)
        model.OLTC_Change_L = pyo.Constraint(model.T, model.OLTC_Taps, rule=oltc_chg_l)

        def oltc_max(m):
            return sum(m.oltc_change[t] for t in m.T) <= max_actions
        model.OLTC_MaxActions = pyo.Constraint(rule=oltc_max)
        print(f"  OLTC约束: 单档选择×{self.n_periods}, 动作限制≤{max_actions}")

    def _add_sc_constraints(self, model):
        if not self.use_sc:
            return
        n_stages = model._sc_n_stages
        max_actions = model._sc_max_actions
        def sc_chg_u(m, t, k):
            if t == 0: return pyo.Constraint.Skip
            return m.sc_change[t, k] * n_stages >= m.sc_stage[t, k] - m.sc_stage[t-1, k]
        def sc_chg_l(m, t, k):
            if t == 0: return pyo.Constraint.Skip
            return m.sc_change[t, k] * n_stages >= m.sc_stage[t-1, k] - m.sc_stage[t, k]
        model.SC_Change_U = pyo.Constraint(model.T, model.SC_Set, rule=sc_chg_u)
        model.SC_Change_L = pyo.Constraint(model.T, model.SC_Set, rule=sc_chg_l)
        def sc_max(m, k):
            return sum(m.sc_change[t, k] for t in m.T) <= max_actions
        model.SC_MaxActions = pyo.Constraint(model.SC_Set, rule=sc_max)
        print(f"  SC约束: 变化检测×{len(self.sc_config['buses'])}, 动作限制≤{max_actions}/组")

    def _add_svc_constraints(self, model):
        if not self.use_svc or not hasattr(model, 'SVC_Set'):
            return
        print(f"  SVC约束: 功率限制在bounds中定义")

    def _add_ess_constraints(self, model):
        """储能约束 —— 与 ED 完全一致 (P_ch/P_dis 单位 MW)"""
        if not self.use_ess:
            return
        ess = self.ess_config
        capacity = ess["capacity_mwh"]
        eta_ch = ess["efficiency_charge"]
        eta_dis = ess["efficiency_discharge"]
        soc_init = ess["soc_init"]
        delta_t = self.delta_t

        def soc_dynamics(m, t, k):
            E_prev = soc_init * capacity[k] if t == 0 else m.E_soc[t-1, k]
            return m.E_soc[t, k] == E_prev + eta_ch * m.P_ch[t, k] * delta_t \
                   - m.P_dis[t, k] / eta_dis * delta_t
        model.SOC_Dynamics = pyo.Constraint(model.T, model.ESS_Set, rule=soc_dynamics)

        if ess.get("soc_final_constraint", True):
            def final_soc(m, k):
                E_init = soc_init * capacity[k]
                return (0.9 * E_init, m.E_soc[self.n_periods - 1, k], 1.1 * E_init)
            model.FinalSOC = pyo.Constraint(model.ESS_Set, rule=final_soc)

        if model._ess_mutex:
            max_ch = model._ess_max_ch
            max_dis = model._ess_max_dis
            def ess_ch_mutex(m, t, k):
                return m.P_ch[t, k] <= max_ch[k] * (1 - m.ess_mode[t, k])
            def ess_dis_mutex(m, t, k):
                return m.P_dis[t, k] <= max_dis[k] * m.ess_mode[t, k]
            model.ESS_Charge_Mutex = pyo.Constraint(model.T, model.ESS_Set, rule=ess_ch_mutex)
            model.ESS_Discharge_Mutex = pyo.Constraint(model.T, model.ESS_Set, rule=ess_dis_mutex)
        print(f"  ESS约束: SOC动态, 互斥约束")

    def _add_dg_constraints(self, model):
        """DG约束 —— 与 ED 一致 (P_cut/Q_pv/Q_wt 单位 MW/MVar)"""
        pv_data = self.scenario_data.get("pv", {})
        pv_buses = self.pv_config["buses"]
        pv_capacity = self.pv_config["capacity"]
        wt_data = self.scenario_data.get("wt", {})
        wt_buses = self.wt_config["buses"]
        wt_capacity = self.wt_config.get("capacity", [0.5]*len(wt_buses))

        if self.allow_curtailment:
            def pv_curtail(m, t, j):
                bus = pv_buses[j]
                return m.P_cut[t, j] <= pv_data.get(bus, np.zeros(self.n_periods))[t]
            model.PV_Curtail_Limit = pyo.Constraint(model.T, model.PV_Set, rule=pv_curtail)

        # PV 逆变器容量 (MW/MVar)
        def pv_inv(m, t, j):
            bus = pv_buses[j]
            p_pv_mw = pv_data.get(bus, np.zeros(self.n_periods))[t]
            p_actual = (p_pv_mw - m.P_cut[t, j]) if self.allow_curtailment else p_pv_mw
            s_max = pv_capacity[j] * 1.1
            return p_actual**2 + m.Q_pv[t, j]**2 <= s_max**2
        model.PV_Inverter = pyo.Constraint(model.T, model.PV_Set, rule=pv_inv)

        # WT 逆变器容量
        def wt_inv(m, t, j):
            bus = wt_buses[j]
            p_wt_mw = wt_data.get(bus, np.zeros(self.n_periods))[t]
            s_max = wt_capacity[j] * 1.0
            return m.Q_wt[t, j]**2 + p_wt_mw**2 <= s_max**2
        model.WT_Inverter = pyo.Constraint(model.T, model.WT_Set, rule=wt_inv)
        print(f"  DG约束: 弃光限制, 逆变器容量")

    def _add_tie_switch_constraints(self, model):
        """联络线约束 —— 与 ED 一致"""
        network = self.network
        n_ties = model._n_ties
        max_actions = model._tie_max_actions
        M_power = 10.0
        M_current = 100.0

        # Big-M
        def tie_p_u(m, t, k): return m.P_tie[t, k] <= M_power * m.sw_tie[t, k]
        def tie_p_l(m, t, k): return m.P_tie[t, k] >= -M_power * m.sw_tie[t, k]
        def tie_q_u(m, t, k): return m.Q_tie[t, k] <= M_power * m.sw_tie[t, k]
        def tie_q_l(m, t, k): return m.Q_tie[t, k] >= -M_power * m.sw_tie[t, k]
        def tie_l_u(m, t, k): return m.l_tie[t, k] <= M_current * m.sw_tie[t, k]
        model.TieP_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_p_u)
        model.TieP_L = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_p_l)
        model.TieQ_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_q_u)
        model.TieQ_L = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_q_l)
        model.TieL_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_l_u)

        # 联络线 SOCP
        def tie_soc(m, t, k):
            f_bus = network.tie_switches[k]["from"]
            return m.P_tie[t, k]**2 + m.Q_tie[t, k]**2 <= m.l_tie[t, k] * m.u[t, f_bus]
        model.TieSOC = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_soc)

        # 联络线电压降落 (Big-M relaxation)
        M_voltage = 1.0
        def tie_vd_u(m, t, k):
            tie = network.tie_switches[k]
            f_bus, t_bus = tie["from"], tie["to"]
            r, x = tie["r_pu"], tie["x_pu"]
            vd = m.u[t, t_bus] - m.u[t, f_bus] + 2*(r*m.P_tie[t,k] + x*m.Q_tie[t,k]) \
                 - (r**2 + x**2)*m.l_tie[t,k]
            return vd <= M_voltage * (1 - m.sw_tie[t, k])
        def tie_vd_l(m, t, k):
            tie = network.tie_switches[k]
            f_bus, t_bus = tie["from"], tie["to"]
            r, x = tie["r_pu"], tie["x_pu"]
            vd = m.u[t, t_bus] - m.u[t, f_bus] + 2*(r*m.P_tie[t,k] + x*m.Q_tie[t,k]) \
                 - (r**2 + x**2)*m.l_tie[t,k]
            return vd >= -M_voltage * (1 - m.sw_tie[t, k])
        model.TieVD_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_vd_u)
        model.TieVD_L = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tie_vd_l)

        # 开关变化
        def sw_chg_u(m, t, k):
            if t == 0:
                init = self.tie_config.get("initial_status", [0]*n_ties)[k]
                return m.sw_change[t, k] >= m.sw_tie[t, k] - init
            return m.sw_change[t, k] >= m.sw_tie[t, k] - m.sw_tie[t-1, k]
        def sw_chg_l(m, t, k):
            if t == 0:
                init = self.tie_config.get("initial_status", [0]*n_ties)[k]
                return m.sw_change[t, k] >= init - m.sw_tie[t, k]
            return m.sw_change[t, k] >= m.sw_tie[t-1, k] - m.sw_tie[t, k]
        model.SwChg_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=sw_chg_u)
        model.SwChg_L = pyo.Constraint(model.T, model.TieSwitch_Set, rule=sw_chg_l)

        def max_sw(m):
            return sum(m.sw_change[t, k] for t in m.T for k in m.TieSwitch_Set) <= max_actions
        model.Tie_MaxActions = pyo.Constraint(rule=max_sw)
        print(f"  联络线约束: Big-M, 电压降落, SOC, 动作限制≤{max_actions}")

    def _add_radiality_constraints(self, model):
        """辐射状约束 —— 与 ED 一致 (单商品流)"""
        network = self.network
        n_buses = network.n_buses
        n_branches = network.n_branches

        model.f_fixed = pyo.Var(model.T, model.L, within=pyo.Reals,
                                bounds=(-(n_buses-1), n_buses-1), initialize=0)
        model.f_tie = pyo.Var(model.T, model.TieSwitch_Set, within=pyo.Reals,
                              bounds=(-(n_buses-1), n_buses-1), initialize=0)

        # 闭合支路数 = n - 1
        def radiality_count(m, t):
            return n_branches + sum(m.sw_tie[t, k] for k in m.TieSwitch_Set) == n_buses - 1
        model.RadialityCount = pyo.Constraint(model.T, rule=radiality_count)

        # 商品流平衡
        def cf_balance(m, t, j):
            if j == network.root_bus:
                flow_out = 0
                for child in network.children[j]:
                    br = network.branch_idx[(j, child)]
                    flow_out += m.f_fixed[t, br]
                return flow_out == n_buses - 1
            else:
                parent = network.parent[j]
                br = network.branch_idx[(parent, j)]
                flow_in = m.f_fixed[t, br]
                flow_out = 0
                for child in network.children[j]:
                    cbr = network.branch_idx[(j, child)]
                    flow_out += m.f_fixed[t, cbr]
                tie_flow = 0
                for k, tie in enumerate(network.tie_switches):
                    if j == tie["from"]:
                        tie_flow -= m.f_tie[t, k]
                    elif j == tie["to"]:
                        tie_flow += m.f_tie[t, k]
                return flow_in - flow_out + tie_flow == 1
        model.CF_Balance = pyo.Constraint(model.T, model.N, rule=cf_balance)

        # 联络线商品流受开关限制
        M_flow = n_buses - 1
        def tf_u(m, t, k): return m.f_tie[t, k] <= M_flow * m.sw_tie[t, k]
        def tf_l(m, t, k): return m.f_tie[t, k] >= -M_flow * m.sw_tie[t, k]
        model.TF_U = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tf_u)
        model.TF_L = pyo.Constraint(model.T, model.TieSwitch_Set, rule=tf_l)
        print(f"  辐射状约束: 单商品流, 节点数={n_buses}")

    def _add_ev_constraints(self, model):
        """EV约束 —— P_ev (MW)"""
        network = self.network
        n_ev = model._n_ev
        if n_ev == 0:
            return
        eta_ch = self.ev_config.get("charge_efficiency", 0.95)
        max_cut = model._max_ev_cut_ratio

        # 1. 负荷跟踪: P_ev = P_required * (1 - cut_ratio)  [MW]
        if len(self.ev_data) > 0:
            def ev_load_track(m, t, k):
                if k not in self.ev_data:
                    return pyo.Constraint.Skip
                required_mw = self.ev_data[k]["load_kw"][t] / 1000.0  # kW → MW
                return m.P_ev[t, k] == required_mw * (1 - m.ev_cut_ratio[t, k])
            model.EV_Load_Track = pyo.Constraint(model.T, model.EV_Set, rule=ev_load_track)
            print(f"  EV负荷跟踪: P_ev = P_required × (1 - cut_ratio), cut_ratio ≤ {int(max_cut*100)}%")

        # 2. 阶梯削减: tier1 + tier2 + tier3 = cut_ratio
        def ev_tier_sum(m, t, k):
            return m.ev_cut_tier1[t, k] + m.ev_cut_tier2[t, k] + m.ev_cut_tier3[t, k] \
                   == m.ev_cut_ratio[t, k]
        model.EV_Tier_Sum = pyo.Constraint(model.T, model.EV_Set, rule=ev_tier_sum)
        print(f"  EV阶梯削减: tier1(0-30%) + tier2(30-60%) + tier3(60-80%) = cut_ratio")

        # 3. 能量动态: P_ev (MW), E_ev (MWh)
        def ev_energy(m, t, k):
            station = network.ev_stations[k]
            E_prev = station["arrival_energy_mwh"] if t == 0 else m.E_ev[t-1, k]
            return m.E_ev[t, k] == E_prev + eta_ch * m.P_ev[t, k] * self.delta_t
        model.EV_Energy = pyo.Constraint(model.T, model.EV_Set, rule=ev_energy)

        # 4. 能量缺口
        def ev_shortage(m, k):
            target = network.ev_stations[k]["target_energy_mwh"]
            return m.E_ev_shortage[k] >= target - m.E_ev[self.n_periods - 1, k]
        model.EV_Shortage = pyo.Constraint(model.EV_Set, rule=ev_shortage)

        # 5. 充电状态检测
        M_pw = max(s["max_power_mw"] for s in network.ev_stations)
        eps = 0.001
        def ev_chg_u(m, t, k):
            return m.P_ev[t, k] <= M_pw * m.ev_charging[t, k]
        def ev_chg_l(m, t, k):
            return m.P_ev[t, k] >= eps * m.ev_charging[t, k]
        model.EV_Chg_U = pyo.Constraint(model.T, model.EV_Set, rule=ev_chg_u)
        model.EV_Chg_L = pyo.Constraint(model.T, model.EV_Set, rule=ev_chg_l)

        # 6. 中断检测
        def ev_int(m, t, k):
            if t == 0: return m.z_int[t, k] == 0
            return m.z_int[t, k] >= m.ev_charging[t-1, k] - m.ev_charging[t, k]
        model.EV_Interrupt = pyo.Constraint(model.T, model.EV_Set, rule=ev_int)
        print(f"  EV约束: 能量动态, 缺口检测, 中断检测")

    # ================================================================== #
    #  目标函数 —— 对齐 ED, 物理成本无权重
    # ================================================================== #

    def _add_objective(self, model):
        """
        目标函数: 与 ED 对齐, 物理成本不加权重, 偏好项可加权重.

        1. 购电成本 = Σ price * P_grid(MW) * Δt       [无权重]
        2. 网损成本 = loss_cost * Σ r*l*s_base * Δt    [无权重]
        3. ESS运行  = ess_rate * Σ (P_ch+P_dis) * Δt   [无权重]
        4. 弃光成本 = curt_cost * Σ P_cut(MW) * Δt     [无权重]
        5. 联络线切换 = sw_cost * N_switches            [无权重]
        6. 电压越限惩罚                                  [加权重, 偏好项]
        7. EV用户体验                                    [加权重, 偏好项]
        """
        network = self.network
        delta_t = self.delta_t
        s_base = self.s_base

        # 1. 购电成本 (P_grid 单位 MW)
        grid_cost = sum(self.price_data[t] * model.P_grid[t] * delta_t for t in model.T)

        # 2. 网损成本 (固定支路 + 联络线)
        loss_expr = 0
        for t in model.T:
            for l_idx in range(network.n_branches):
                f = int(network.from_bus[l_idx])
                to = int(network.to_bus[l_idx])
                r = network.r_pu[l_idx]
                loss_expr += model.l[t, (f, to)] * r * s_base
        if self.use_tie and network.n_tie_switches > 0:
            for t in model.T:
                for k, tie in enumerate(network.tie_switches):
                    loss_expr += model.l_tie[t, k] * tie["r_pu"] * s_base
        loss_cost = self.loss_cost * loss_expr * delta_t

        # 3. ESS运行成本 (P_ch/P_dis 单位 MW)
        ess_cost = 0
        if self.use_ess:
            ess_rate = self.ess_config.get("cost_per_mwh", 40)
            ess_cost = sum(
                ess_rate * (model.P_ch[t, k] + model.P_dis[t, k]) * delta_t
                for t in model.T for k in model.ESS_Set
            )

        # 4. 弃光成本 (P_cut 单位 MW)
        curtail_cost = 0
        if self.allow_curtailment:
            curtail_cost = sum(
                self.curtailment_cost * model.P_cut[t, j] * delta_t
                for t in model.T for j in model.PV_Set
            )

        # 5. 联络线切换成本
        sw_cost = 0
        if self.use_tie and hasattr(model, 'sw_change'):
            sw_unit = self.tie_config.get("switching_cost", 50)
            sw_cost = sum(
                sw_unit * model.sw_change[t, k]
                for t in model.T for k in model.TieSwitch_Set
            )

        # 6. 电压越限惩罚 (偏好项, 加权重)
        volt_penalty = 0
        if self.voltage_soft:
            for t in model.T:
                w = self.get_period_weight(t, "voltage")
                for j in model.N:
                    volt_penalty += w * self.voltage_penalty_coef * \
                                    (model.v_over[t, j] + model.v_under[t, j])

        # 7. EV用户体验惩罚 (偏好项)
        ev_penalty = 0
        if self.use_ev and hasattr(model, 'P_ev'):
            pen_shortage = self.ev_config.get("penalty_soc_shortage", 500)
            pen_interrupt = self.ev_config.get("penalty_interruption", 50)
            pen_t1 = self.opt_config.get("ev_curtailment_penalty_tier1", 200)
            pen_t2 = self.opt_config.get("ev_curtailment_penalty_tier2", 800)
            pen_t3 = self.opt_config.get("ev_curtailment_penalty_tier3", 1500)

            # 中断惩罚
            for t in model.T:
                w = self.get_period_weight(t, "ev_satisfaction")
                for k in model.EV_Set:
                    ev_penalty += w * model.z_int[t, k] * pen_interrupt

            # 能量缺口惩罚
            for k in model.EV_Set:
                ev_penalty += model.E_ev_shortage[k] * pen_shortage

            # 阶梯削减惩罚 (MW)
            if len(self.ev_data) > 0 and hasattr(model, 'ev_cut_tier1'):
                for t in model.T:
                    for k in model.EV_Set:
                        if k in self.ev_data:
                            req_mw = self.ev_data[k]["load_kw"][t] / 1000.0
                            ev_penalty += req_mw * model.ev_cut_tier1[t, k] * pen_t1 * delta_t
                            ev_penalty += req_mw * model.ev_cut_tier2[t, k] * pen_t2 * delta_t
                            ev_penalty += req_mw * model.ev_cut_tier3[t, k] * pen_t3 * delta_t

        total = grid_cost + loss_cost + ess_cost + curtail_cost + sw_cost + volt_penalty + ev_penalty
        model.objective = pyo.Objective(expr=total, sense=pyo.minimize)
        print(f"  目标函数: 购电+网损+ESS+弃光+开关+电压惩罚+EV体验")

    # ================================================================== #
    #  求解
    # ================================================================== #

    def solve(self, solver_name: str = "gurobi",
              time_limit: int = 3600,
              mip_gap: float = 1e-3,
              verbose: bool = True) -> dict:
        if self.model is None:
            raise ValueError("请先构建模型!")

        print("\n" + "=" * 60)
        print(f"开始求解 (求解器: {solver_name})")
        print("=" * 60)

        solver = pyo.SolverFactory(solver_name)
        if solver_name == "gurobi":
            solver.options["TimeLimit"] = time_limit
            solver.options["MIPGap"] = mip_gap
            solver.options["OutputFlag"] = 1 if verbose else 0
            solver.options["NonConvex"] = 2   # 与 ED 一致
            solver.options["Cuts"] = 2
            solver.options["Presolve"] = 2

        start_time = time.time()
        result = solver.solve(self.model, tee=verbose)
        solve_time = time.time() - start_time

        self.statistics["solve_time"] = solve_time
        self.statistics["solver_status"] = str(result.solver.status)
        self.statistics["termination_condition"] = str(result.solver.termination_condition)

        # 获取目标值
        try:
            obj_value = pyo.value(self.model.objective)
            self.statistics["objective_value"] = obj_value
        except:
            obj_value = None
            self.statistics["objective_value"] = 0

        # MIP gap
        actual_gap = None
        try:
            lb = result.problem.lower_bound
            ub = result.problem.upper_bound
            if ub != 0:
                actual_gap = abs(ub - lb) / abs(ub)
        except:
            pass
        self.statistics["mip_gap"] = actual_gap

        termination = result.solver.termination_condition
        if termination == pyo.TerminationCondition.optimal:
            print(f"\n求解成功! (最优解)")
            print(f"  目标函数值: {obj_value:.2f} 元")
            print(f"  求解时间: {solve_time:.2f}秒")
            if actual_gap is not None:
                print(f"  MIP Gap: {actual_gap * 100:.4f}%")
            self.statistics["solve_status"] = "optimal"
        elif termination == pyo.TerminationCondition.feasible:
            print(f"\n求解成功! (可行解)")
            print(f"  目标函数值: {obj_value:.2f} 元")
            print(f"  求解时间: {solve_time:.2f}秒")
            if actual_gap is not None:
                print(f"  MIP Gap: {actual_gap * 100:.4f}%")
            self.statistics["solve_status"] = "feasible"
        elif termination == pyo.TerminationCondition.maxTimeLimit:
            if obj_value is not None and (actual_gap is None or actual_gap < 0.005):
                print(f"\n求解基本成功! (达到时间限制，解质量可接受)")
                print(f"  目标函数值: {obj_value:.2f} 元")
                print(f"  求解时间: {solve_time:.2f}秒")
                if actual_gap is not None:
                    print(f"  MIP Gap: {actual_gap * 100:.4f}%")
                self.statistics["solve_status"] = "acceptable"
            else:
                print(f"\n求解未完成 (达到时间限制)")
                if obj_value:
                    print(f"  当前最优解: {obj_value:.2f} 元")
                if actual_gap is not None:
                    print(f"  MIP Gap: {actual_gap * 100:.4f}%")
                self.statistics["solve_status"] = "timeout"
        else:
            print(f"\n求解失败: {termination}")
            self.statistics["solve_status"] = "failed"

        return self.statistics

    # ================================================================== #
    #  结果提取 —— 保持原接口格式不变
    # ================================================================== #

    def get_results(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        model = self.model
        network = self.network
        n_periods = self.n_periods
        delta_t = self.delta_t
        s_base = self.s_base

        results = {}

        # ===== 电压 =====
        voltage = np.zeros((n_periods, network.n_buses))
        for t in range(n_periods):
            for j in range(network.n_buses):
                u_sq = pyo.value(model.u[t, j])
                voltage[t, j] = np.sqrt(max(u_sq, 0))
        results["voltage"] = {
            "values": voltage,
            "min": float(voltage.min()),
            "max": float(voltage.max()),
            "mean": float(voltage.mean()),
        }

        # ===== 网损 (pu → kW) =====
        loss_per_period = np.zeros(n_periods)
        for t in range(n_periods):
            loss = 0
            for l_idx in range(network.n_branches):
                f = int(network.from_bus[l_idx])
                to = int(network.to_bus[l_idx])
                r = network.r_pu[l_idx]
                loss += pyo.value(model.l[t, (f, to)]) * r * s_base * 1000  # kW
            loss_per_period[t] = loss
        results["loss"] = {
            "per_period_kw": loss_per_period,
            "total_kwh": float(np.sum(loss_per_period) * delta_t),
            "average_kw": float(loss_per_period.mean()),
        }

        # ===== 购电 (MW) =====
        p_grid = np.zeros(n_periods)
        for t in range(n_periods):
            p_grid[t] = pyo.value(model.P_grid[t])  # 已是 MW
        results["grid"] = {
            "power_mw": p_grid,
            "total_purchase_mwh": float(np.sum(p_grid[p_grid > 0]) * delta_t),
        }

        # ===== OLTC =====
        if self.use_oltc and hasattr(model, 'oltc_tap'):
            oltc_tap = np.zeros(n_periods, dtype=int)
            n_taps = model._oltc_n_taps
            tap_min = model._oltc_tap_min
            for t in range(n_periods):
                for k in range(n_taps):
                    if pyo.value(model.oltc_tap[t, k]) > 0.5:
                        oltc_tap[t] = tap_min + k
                        break
            results["oltc"] = {
                "tap_position": oltc_tap.tolist(),
                "changes": int(np.sum(np.abs(np.diff(oltc_tap)))),
            }

        # ===== ESS (MW) =====
        if self.use_ess:
            n_ess = len(self.ess_config["buses"])
            ess_charge = np.zeros((n_periods, n_ess))
            ess_discharge = np.zeros((n_periods, n_ess))
            ess_soc = np.zeros((n_periods, n_ess))
            for t in range(n_periods):
                for k in range(n_ess):
                    ess_charge[t, k] = pyo.value(model.P_ch[t, k])    # MW
                    ess_discharge[t, k] = pyo.value(model.P_dis[t, k])  # MW
                    ess_soc[t, k] = pyo.value(model.E_soc[t, k])       # MWh
            results["ess"] = {
                "charge_mw": ess_charge,
                "discharge_mw": ess_discharge,
                "soc_mwh": ess_soc,
                "total_charge_mwh": float(np.sum(ess_charge) * delta_t),
                "total_discharge_mwh": float(np.sum(ess_discharge) * delta_t),
            }

        # SC
        if self.use_sc and hasattr(model, 'sc_stage'):
            n_sc = len(self.sc_config["buses"])
            sc_stages = np.zeros((n_periods, n_sc))
            for t in range(n_periods):
                for k in range(n_sc):
                    sc_stages[t, k] = pyo.value(model.sc_stage[t, k])
            results["sc"] = {
                "buses": self.sc_config["buses"],
                "stages": sc_stages,
                "q_per_stage_mvar": self.sc_config["q_per_stage"],
            }

        # ===== EV (MW) =====
        if self.use_ev and hasattr(model, 'P_ev'):
            n_ev = network.n_ev_stations
            ev_power = np.zeros((n_periods, n_ev))
            ev_energy = np.zeros((n_periods, n_ev))
            ev_interrupt = np.zeros((n_periods, n_ev), dtype=int)
            ev_shortage = np.zeros(n_ev)
            ev_cut_ratio = np.zeros((n_periods, n_ev))
            ev_cut_tier1 = np.zeros((n_periods, n_ev))
            ev_cut_tier2 = np.zeros((n_periods, n_ev))
            ev_cut_tier3 = np.zeros((n_periods, n_ev))

            for t in range(n_periods):
                for k in range(n_ev):
                    ev_power[t, k] = pyo.value(model.P_ev[t, k])  # MW
                    ev_energy[t, k] = pyo.value(model.E_ev[t, k])
                    ev_interrupt[t, k] = int(round(pyo.value(model.z_int[t, k])))
                    if hasattr(model, 'ev_cut_ratio'):
                        ev_cut_ratio[t, k] = pyo.value(model.ev_cut_ratio[t, k])
                    if hasattr(model, 'ev_cut_tier1'):
                        ev_cut_tier1[t, k] = pyo.value(model.ev_cut_tier1[t, k])
                    if hasattr(model, 'ev_cut_tier2'):
                        ev_cut_tier2[t, k] = pyo.value(model.ev_cut_tier2[t, k])
                    if hasattr(model, 'ev_cut_tier3'):
                        ev_cut_tier3[t, k] = pyo.value(model.ev_cut_tier3[t, k])
            for k in range(n_ev):
                ev_shortage[k] = pyo.value(model.E_ev_shortage[k])

            results["ev"] = {
                "power_mw": ev_power,
                "energy_mwh": ev_energy,
                "interruptions": ev_interrupt,
                "total_interruptions": int(np.sum(ev_interrupt)),
                "shortage_mwh": ev_shortage,
                "total_shortage_mwh": float(np.sum(ev_shortage)),
                "cut_ratio": ev_cut_ratio,
                "avg_cut_ratio": float(ev_cut_ratio.mean()),
                "max_cut_ratio": float(ev_cut_ratio.max()),
                "cut_tier1": ev_cut_tier1,
                "cut_tier2": ev_cut_tier2,
                "cut_tier3": ev_cut_tier3,
            }

        # ===== 联络线 =====
        if self.use_tie and hasattr(model, 'sw_tie'):
            n_ties = network.n_tie_switches
            tie_status = np.zeros((n_periods, n_ties), dtype=int)
            for t in range(n_periods):
                for k in range(n_ties):
                    tie_status[t, k] = int(round(pyo.value(model.sw_tie[t, k])))
            results["tie_switches"] = {
                "status": tie_status,
                "total_switches": int(np.sum(np.abs(np.diff(tie_status, axis=0)))),
            }

        # ===== 成本 =====
        results["cost"] = {
            "total": float(self.statistics.get("objective_value", 0)),
        }
        results["statistics"] = self.statistics

        return results

    # ================================================================== #
    #  打印摘要
    # ================================================================== #

    def print_summary(self):
        results = self.get_results()

        print("\n" + "=" * 60)
        print("Task C 综合优化结果摘要")
        print("=" * 60)

        print(f"\n【总成本】{results['cost']['total']:.2f} 元")

        volt = results["voltage"]
        print(f"\n【电压分析】")
        print(f"  最低电压: {volt['min']:.4f} pu")
        print(f"  最高电压: {volt['max']:.4f} pu")
        print(f"  平均电压: {volt['mean']:.4f} pu")

        loss = results["loss"]
        print(f"\n【网损分析】")
        print(f"  总网损: {loss['total_kwh']:.2f} kWh")
        print(f"  平均网损: {loss['average_kw']:.2f} kW")

        grid = results["grid"]
        print(f"\n【购电分析】")
        print(f"  总购电量: {grid['total_purchase_mwh']:.2f} MWh")

        if "oltc" in results:
            print(f"\n【OLTC分析】")
            print(f"  档位变化次数: {results['oltc']['changes']}")

        if "ess" in results:
            ess = results["ess"]
            print(f"\n【储能分析】")
            print(f"  总充电量: {ess['total_charge_mwh']:.2f} MWh")
            print(f"  总放电量: {ess['total_discharge_mwh']:.2f} MWh")

        if "ev" in results:
            ev = results["ev"]
            print(f"\n【EV充电分析】")
            print(f"  总充电中断次数: {ev['total_interruptions']}")
            print(f"  总能量缺口: {ev['total_shortage_mwh']:.4f} MWh")
            if "avg_cut_ratio" in ev:
                max_cut = self.ev_config.get("max_cut_ratio", 0.8)
                print(f"  平均削减比例: {ev['avg_cut_ratio']*100:.2f}%")
                print(f"  最大削减比例: {ev['max_cut_ratio']*100:.2f}% (上限{int(max_cut*100)}%)")

        if "tie_switches" in results:
            print(f"\n【联络线分析】")
            print(f"  总切换次数: {results['tie_switches']['total_switches']}")

        print(f"\n【求解信息】")
        print(f"  求解时间: {self.statistics.get('solve_time', 0):.2f} 秒")
        print(f"  二进制变量数: {self.statistics.get('n_binary_vars', 0)}")
        print("=" * 60)


def create_joint_model(config: dict) -> JointOptModel:
    """创建综合优化模型实例"""
    return JointOptModel(config)
