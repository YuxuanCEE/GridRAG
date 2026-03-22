# -*- coding: utf-8 -*-
"""main_online.py

GridRAG 在线 zero-shot 主 pipeline：
1) 读取在线场景文件路径（data/online_inf，格式与 profiles 完全一致）
2) 检索最相似历史场景（top-1）
3) 加载历史优化结果（opt_results/{task}/..._results.json）
4) 提取 warm-start 参数（控制量）
5) （可选）Diffusion refine：预留接口
6) 将 warm-start 写入 Pyomo 模型变量，并启用 warmstart 求解
7) 保存在线结果到 opt_results/online/

说明：
- 目前只实现 zero-shot 线路；冻结NN/RL + 投影层微调不在本文件实现。
- diffusion 仅预留接口，默认开启，可通过参数关闭或基于相似度阈值自动触发。
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyomo
import pyomo.environ as pyo

# 项目根目录
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_networks import (
    get_network_config,
    get_network_instance,
    get_result_filename,
)
from data.data_loader import get_data_loader

from Model.retrieval.retriever import ScenarioRetriever
from utils.result_loader import ResultLoader
from utils.warm_start_extractor import WarmStartExtractor

from models.day_ahead.misocp_var_opt import create_day_ahead_model
from models.real_time.socp_var_opt import create_real_time_model
from models.ed.socp_ed import create_ed_model
from models.joint.socp_joint import create_joint_model


# =========================
# Timing utilities
# =========================

class StageTimer:
    """Lightweight stage timer for online pipeline benchmarking.

    Uses time.perf_counter() (high-resolution, monotonic).
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self.times: Dict[str, float] = {}

    def tic(self) -> float:
        return time.perf_counter()

    def toc(self, name: str, start: float) -> None:
        self.times[name] = float(time.perf_counter() - start)

    def total(self) -> float:
        return float(time.perf_counter() - self._t0)

# =========================
# Diffusion refine
# =========================

from Model.diffusion.diffusion_model import diffusion_refine_pipeline


def diffusion_refine(
    warm_start: Dict[str, Any],
    retrieved_json: Dict[str, Any],
    query_features: np.ndarray,
    best_distance: float,
    task: str,
    network: str,
    s_base: float,
    diffusion_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Diffusion refine 接口。

    调用 SDEdit-based diffusion pipeline：
        retrieved solution -> q_sample加噪 -> reverse denoise -> refined warm_start

    Returns
    -------
    refined_warm_start : dict
    diff_info : dict
        diffusion 相关信息（noise_level, metrics, timing 等）
    """
    opt_results_dir = str(PROJECT_ROOT / "opt_results")

    refined_ws, diff_info = diffusion_refine_pipeline(
        retrieved_json=retrieved_json,
        warm_start=warm_start,
        query_features=query_features,
        best_distance=best_distance,
        task=task,
        network=network,
        s_base=s_base,
        opt_results_dir=opt_results_dir,
        diffusion_cfg=diffusion_cfg,
    )
    return refined_ws, diff_info


# =========================
# 工具函数：在线数据加载
# =========================

def _make_online_loader(base_config: dict, scenario_file: Path) -> Any:
    """构造一个 DataLoader，使其 profiles_dir 指向在线数据所在目录。"""
    cfg = deepcopy(base_config)
    cfg.setdefault("paths", {})
    cfg["paths"]["profiles"] = str(scenario_file.parent)
    return get_data_loader(cfg)


def load_online_scenario(config: dict, network_name: str, scenario_file: Path,
                         n_periods: int, scenario_id: Optional[int] = None) -> Dict[str, Any]:
    """加载在线 DER 场景（与 profiles 相同格式）。"""
    loader = _make_online_loader(config, scenario_file)
    scenario_data = loader.get_scenario_data(
        filename=scenario_file.name,
        scenario_id=scenario_id,
        n_periods=n_periods,
    )
    return scenario_data


def load_online_ev(config: dict, ev_file: Path) -> Dict[str, Any]:
    """Joint 任务用：加载在线 EV 场景（与 profiles 相同格式）。"""
    # main_joint 里有 load_ev_data，但那是基于 config 路径的；这里用相同策略临时指向 online_inf
    cfg = deepcopy(config)
    cfg.setdefault("paths", {})
    cfg["paths"]["profiles"] = str(ev_file.parent)
    # 直接复用 joint 模型内部加载逻辑：socp_joint 中并没有 loader，所以这里手写读取
    import pandas as pd
    df = pd.read_csv(ev_file)
    if df.columns[0].startswith('\ufeff'):
        df.columns = [c.replace('\ufeff', '') for c in df.columns]
    # 期望列：timestamp + station_xxx 或类似格式；项目中实现较灵活
    # 这里直接把 df 返回，让 create_joint_model 的 build_model 内部用你们已有的 load_* 逻辑？
    # 为避免侵入式修改，我们在 main_online 里使用 models.joint.socp_joint 的辅助函数读取
    return {"dataframe": df, "source_file": str(ev_file)}


# =========================
# Warm-start apply
# =========================

def _safe_set(var, value):
    """安全设置 Pyomo Var 的 value（忽略None）。"""
    if value is None:
        return
    try:
        var.set_value(value)
    except Exception:
        # fallback
        try:
            var.value = value
        except Exception:
            pass


def convert_for_json(obj):
    """递归转换 numpy / pathlib 等对象为 JSON 可序列化的原生类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def apply_warm_start_vvc_day_ahead(da_model, warm: Dict[str, Any]):
    """把 VVC 日前 warm-start 写回 MISOCP 模型。

    关键：该模型包含整数/二元的“辅助变量”用于表达 OLTC 档位 one-hot、动作次数等。
    如果只写 oltc_tap / sc_stage 而不同时写对应的二元/辅助变量，Gurobi 的 MIP start
    很容易因违反约束而被丢弃（日志里会出现 "User MIP start violates constraint ..."）。
    """
    m = da_model.model

    # -------- OLTC --------
    tap = warm.get("oltc", {}).get("tap")
    if tap is not None and hasattr(m, "oltc_tap"):
        tap_list = tap if isinstance(tap, list) else None

        # 先写 oltc_tap
        for t in m.T:
            if tap_list is not None:
                tt = int(round(tap_list[int(t)]))
            else:
                tt = int(round(tap))

            # clip to var bounds if available
            try:
                lb = m.oltc_tap[t].lb
                ub = m.oltc_tap[t].ub
                if lb is not None:
                    tt = max(int(lb), tt)
                if ub is not None:
                    tt = min(int(ub), tt)
            except Exception:
                pass

            _safe_set(m.oltc_tap[t], tt)

        # 写 one-hot lambda（满足 sum lambda == 1 和 tap = sum d*lambda）
        if tap_list is not None and hasattr(m, "oltc_lambda") and hasattr(m, "TapSet"):
            tap_vals = list(m.TapSet)
            tap_min, tap_max = int(min(tap_vals)), int(max(tap_vals))
            for t in m.T:
                tt = int(round(tap_list[int(t)]))
                tt = max(tap_min, min(tap_max, tt))
                for d in m.TapSet:
                    _safe_set(m.oltc_lambda[t, d], 1 if int(d) == tt else 0)

        # 写 oltc_phi（满足相邻时段动作约束）。最小可行取 abs(diff)。
        if tap_list is not None and hasattr(m, "oltc_phi"):
            for t in m.T:
                if int(t) >= (len(tap_list) - 1):
                    continue
                diff = abs(int(round(tap_list[int(t) + 1])) - int(round(tap_list[int(t)])))
                _safe_set(m.oltc_phi[t], int(diff))

    # -------- Shunt Capacitor (SC) --------
    stage = warm.get("sc", {}).get("stage")
    if stage is not None and hasattr(m, "sc_stage"):
        # stage shape: (n_periods, n_sc)
        for t in m.T:
            for i in m.SC_Set:
                try:
                    vv = int(round(stage[int(t)][int(i)]))
                except Exception:
                    continue

                # clip to bounds
                try:
                    lb = m.sc_stage[t, i].lb
                    ub = m.sc_stage[t, i].ub
                    if lb is not None:
                        vv = max(int(lb), vv)
                    if ub is not None:
                        vv = min(int(ub), vv)
                except Exception:
                    pass

                _safe_set(m.sc_stage[t, i], vv)

        # sc_change 是动作指示二元；若 stage 发生变化，需要把 sc_change=1，否则 MIP start 会违反约束
        if hasattr(m, "sc_change") and isinstance(stage, list) and len(stage) > 1:
            nT = len(stage)
            for t in m.T:
                if int(t) >= nT - 1:
                    continue
                for i in m.SC_Set:
                    try:
                        now_v = int(round(stage[int(t)][int(i)]))
                        nxt_v = int(round(stage[int(t) + 1][int(i)]))
                    except Exception:
                        continue
                    _safe_set(m.sc_change[t, i], 1 if now_v != nxt_v else 0)


def apply_warm_start_vvc_real_time(rt_model, warm: Dict[str, Any]):
    """把 VVC 实时 warm-start 写回 SOCP 模型（Q_pv/Q_wt/Q_svc 均为 pu）。"""
    m = rt_model.model
    qpv = warm.get("rt", {}).get("pv", {}).get("q_pu")
    if qpv is not None and hasattr(m, "Q_pv"):
        for t in m.T:
            for i in m.PV_Set:
                _safe_set(m.Q_pv[t, i], float(qpv[t][i]))

    qwt = warm.get("rt", {}).get("wt", {}).get("q_pu")
    if qwt is not None and hasattr(m, "Q_wt"):
        for t in m.T:
            for i in m.WT_Set:
                _safe_set(m.Q_wt[t, i], float(qwt[t][i]))

    qsvc = warm.get("rt", {}).get("svc", {}).get("q_pu")
    if qsvc is not None and hasattr(m, "Q_svc"):
        for t in m.T:
            for i in m.SVC_Set:
                _safe_set(m.Q_svc[t, i], float(qsvc[t][i]))


def apply_warm_start_ed(ed_model, warm: Dict[str, Any]):
    """把 ED warm-start 写回 SOCP+MIP 模型。

    说明：
    - ED 模型变量命名以 models/ed/socp_ed.py 为准：
      P_ch, P_dis, E_soc, (可选) ess_mode, P_cut, Q_pv, Q_wt, P_grid,
      (可选联络线) sw_tie, P_tie, Q_tie, l_tie, sw_change, TieSwitch_Set
    - 对有界变量（例如 E_soc）做 clip，避免微小数值误差触发 Pyomo W1002 警告。
    """
    m = ed_model.model

    def _clip_to_bounds(var, val: float) -> float:
        lb = getattr(var, "lb", None)
        ub = getattr(var, "ub", None)
        # Pyomo VarData 的 lb/ub 可能是表达式；这里尽量转 float
        try:
            lbv = float(lb) if lb is not None else None
        except Exception:
            lbv = None
        try:
            ubv = float(ub) if ub is not None else None
        except Exception:
            ubv = None

        if lbv is not None and val < lbv:
            return lbv
        if ubv is not None and val > ubv:
            return ubv
        return val

    # ====== ESS ======
    ess = warm.get("ess")
    if ess and hasattr(m, "P_ch") and hasattr(m, "ESS_Set"):
        ch = ess.get("charge_mw")
        dis = ess.get("discharge_mw")
        soc = ess.get("soc_mwh")
        mode = ess.get("mode")

        for t in m.T:
            for k in m.ESS_Set:
                if ch is not None:
                    _safe_set(m.P_ch[t, k], float(ch[t][k]))
                if dis is not None:
                    _safe_set(m.P_dis[t, k], float(dis[t][k]))
                if soc is not None and hasattr(m, "E_soc"):
                    v = float(soc[t][k])
                    v = _clip_to_bounds(m.E_soc[t, k], v)
                    _safe_set(m.E_soc[t, k], v)
                if mode is not None and hasattr(m, "ess_mode"):
                    _safe_set(m.ess_mode[t, k], int(round(mode[t][k])))

    # ====== PV curtailment & Q ======
    pv = warm.get("pv")
    if pv and hasattr(m, "PV_Set"):
        curt = pv.get("curtailment_mw")
        q = pv.get("reactive_mvar")

        # ED 中的 curtailment 变量名通常是 P_cut（allow_curtailment=True 时存在）
        if curt is not None and hasattr(m, "P_cut"):
            for t in m.T:
                for i in m.PV_Set:
                    _safe_set(m.P_cut[t, i], float(curt[t][i]))

        if q is not None and hasattr(m, "Q_pv"):
            for t in m.T:
                for i in m.PV_Set:
                    _safe_set(m.Q_pv[t, i], float(q[t][i]))

    # ====== WT Q ======
    wt = warm.get("wt")
    if wt and hasattr(m, "WT_Set") and hasattr(m, "Q_wt"):
        q = wt.get("reactive_mvar")
        if q is not None:
            for t in m.T:
                for i in m.WT_Set:
                    _safe_set(m.Q_wt[t, i], float(q[t][i]))

    # ====== Grid purchase power ======
    grid = warm.get("grid")
    if grid and hasattr(m, "P_grid"):
        p = grid.get("power_mw")
        if p is not None:
            for t in m.T:
                _safe_set(m.P_grid[t], float(p[t]))

    # ====== Tie switches (if enabled) ======
    tie = warm.get("tie_switches") or warm.get("reconfiguration")
    if tie and hasattr(m, "sw_tie"):
        status = tie.get("status")
        changes = tie.get("changes")
        p_tie = tie.get("power_mw") or tie.get("power")  # 兼容字段名
        # ED 模型：TieSwitch_Set
        tie_set = None
        if hasattr(m, "TieSwitch_Set"):
            tie_set = m.TieSwitch_Set
        elif hasattr(m, "TIE_Set"):
            # 兼容旧命名（如果未来版本存在）
            tie_set = m.TIE_Set

        if tie_set is not None and status is not None:
            for t in m.T:
                for k in tie_set:
                    _safe_set(m.sw_tie[t, k], int(round(status[t][int(k)] if isinstance(k, (int,)) else status[t][k])))

        if tie_set is not None and changes is not None and hasattr(m, "sw_change"):
            for t in m.T:
                for k in tie_set:
                    _safe_set(m.sw_change[t, k], int(round(changes[t][int(k)] if isinstance(k, (int,)) else changes[t][k])))

        if tie_set is not None and p_tie is not None and hasattr(m, "P_tie"):
            for t in m.T:
                for k in tie_set:
                    _safe_set(m.P_tie[t, k], float(p_tie[t][int(k)] if isinstance(k, (int,)) else p_tie[t][k]))


def apply_warm_start_joint(joint_model, warm: Dict[str, Any], s_base: float):
    """把 Joint warm-start 写回 joint MISOCP 模型（模型内部 pu）。"""
    m = joint_model.model

    def _clip_to_bounds(var, val: float) -> float:
        """边界裁剪（参考ED模型）"""
        lb = getattr(var, "lb", None)
        ub = getattr(var, "ub", None)
        try:
            lbv = float(lb) if lb is not None else None
        except Exception:
            lbv = None
        try:
            ubv = float(ub) if ub is not None else None
        except Exception:
            ubv = None
        
        if lbv is not None and val < lbv:
            return lbv
        if ubv is not None and val > ubv:
            return ubv
        return val

    # ====== OLTC ======
    oltc = warm.get("oltc", {}).get("tap_position")
    if oltc is not None and hasattr(m, "oltc_tap"):
        # oltc_tap[t,k] one-hot, TapSet 为 0..n_taps-1? 需查模型，简单按已有结构做 best-effort
        # 若模型里也有 oltc_pos 整数变量，则优先写它
        if hasattr(m, "oltc_pos"):
            for t in m.T:
                t_idx = int(t)
                _safe_set(m.oltc_pos[t], int(round(oltc[t_idx])) if isinstance(oltc, list) else int(round(oltc)))
        else:
            for t in m.T:
                t_idx = int(t)
                kpos = int(round(oltc[t_idx])) if isinstance(oltc, list) else int(round(oltc))
                if hasattr(m, "OLTC_TAP_SET"):
                    tap_set = list(m.OLTC_TAP_SET)
                    kpos = max(min(kpos, max(tap_set)), min(tap_set))
                    for k in m.OLTC_TAP_SET:
                        _safe_set(m.oltc_tap[t, k], 1 if int(k) == kpos else 0)

    # ====== ESS ======
    ess = warm.get("ess", {})
    ch = ess.get("charge_pu")
    dis = ess.get("discharge_pu")
    if hasattr(m, "P_ch") and ch is not None:
        for t in m.T:
            t_idx = int(t)
            for k in m.ESS_Set:
                k_idx = int(k)
                val = float(ch[t_idx][k_idx])
                val = _clip_to_bounds(m.P_ch[t, k], val)
                _safe_set(m.P_ch[t, k], val)
    if hasattr(m, "P_dis") and dis is not None:
        for t in m.T:
            t_idx = int(t)
            for k in m.ESS_Set:
                k_idx = int(k)
                val = float(dis[t_idx][k_idx])
                val = _clip_to_bounds(m.P_dis[t, k], val)
                _safe_set(m.P_dis[t, k], val)

    # ====== EV ======
    ev = warm.get("ev", {})
    pev = ev.get("power_pu")
    cut = ev.get("cut_ratio")
    e_ev = ev.get("energy_mwh")
    tier1 = ev.get("cut_tier1")
    tier2 = ev.get("cut_tier2")
    tier3 = ev.get("cut_tier3")
    
    if hasattr(m, "P_ev") and pev is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(pev[t_idx][s_idx])
                val = _clip_to_bounds(m.P_ev[t, s], val)
                _safe_set(m.P_ev[t, s], val)
    
    if hasattr(m, "ev_cut_ratio") and cut is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(cut[t_idx][s_idx])
                val = _clip_to_bounds(m.ev_cut_ratio[t, s], val)
                _safe_set(m.ev_cut_ratio[t, s], val)
    
    # ✅ 新增：设置能量状态（满足能量动态约束）
    if hasattr(m, "E_ev") and e_ev is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(e_ev[t_idx][s_idx])
                val = _clip_to_bounds(m.E_ev[t, s], val)
                _safe_set(m.E_ev[t, s], val)
    
    # ✅ 新增：设置阶梯削减变量（满足 tier1 + tier2 + tier3 = cut_ratio 约束）
    if hasattr(m, "ev_cut_tier1") and tier1 is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(tier1[t_idx][s_idx])
                val = _clip_to_bounds(m.ev_cut_tier1[t, s], val)
                _safe_set(m.ev_cut_tier1[t, s], val)
    
    if hasattr(m, "ev_cut_tier2") and tier2 is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(tier2[t_idx][s_idx])
                val = _clip_to_bounds(m.ev_cut_tier2[t, s], val)
                _safe_set(m.ev_cut_tier2[t, s], val)
    
    if hasattr(m, "ev_cut_tier3") and tier3 is not None:
        for t in m.T:
            t_idx = int(t)
            for s in m.EV_Set:
                s_idx = int(s)
                val = float(tier3[t_idx][s_idx])
                val = _clip_to_bounds(m.ev_cut_tier3[t, s], val)
                _safe_set(m.ev_cut_tier3[t, s], val)
    
    # ✅ 新增：根据P_ev设置充电状态二进制变量
    if hasattr(m, "ev_charging") and pev is not None:
        epsilon = 0.001 / s_base
        for t in m.T:
            for s in m.EV_Set:
                # 如果P_ev > epsilon，则charging=1，否则=0
                is_charging = 1 if float(pev[t_idx][s_idx]) > epsilon else 0
                _safe_set(m.ev_charging[t, s], is_charging)
    
    # ✅ 新增：根据charging状态设置中断检测
    if hasattr(m, "z_int") and hasattr(m, "ev_charging"):
        for t in m.T:
            for s in m.EV_Set:
                if t == 0:
                    _safe_set(m.z_int[t, s], 0)
                else:
                    # z_int[t] = max(0, charging[t-1] - charging[t])
                    try:
                        prev_charging = m.ev_charging[t-1, s].value or 0
                        curr_charging = m.ev_charging[t, s].value or 0
                        interrupt = max(0, int(prev_charging - curr_charging))
                        _safe_set(m.z_int[t, s], interrupt)
                    except Exception:
                        pass

    # ====== Tie switches ======
    tie = warm.get("tie_switches", {}).get("status")
    if tie is not None and hasattr(m, "sw_tie"):
        tie_set = getattr(m, "TieSwitch_Set", None) or getattr(m, "TIE_Set", None)
        if tie_set is not None:
            for t in m.T:
                t_idx = int(t)
                for l in tie_set:
                    l_idx = int(l)
                    _safe_set(m.sw_tie[t, l], int(round(tie[t_idx][l_idx])))

    # ====== Grid ======
    grid = warm.get("grid", {}).get("power_pu")
    if grid is not None and hasattr(m, "P_grid"):
        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
            for t in m.T:
                t_idx = int(t)
                _safe_set(m.P_grid[t], float(grid[t_idx][0]))
        else:
            for t in m.T:
                t_idx = int(t)
                _safe_set(m.P_grid[t], float(grid[t_idx]))


# =========================
# Solve helpers
# =========================

def solve_pyomo(model: pyo.ConcreteModel, solver_name: str, solver_cfg: dict, warmstart: bool) -> Any:
    solver = pyo.SolverFactory(solver_name)
    if solver_name == "gurobi":
        solver.options["TimeLimit"] = solver_cfg.get("time_limit", 600)
        solver.options["MIPGap"] = solver_cfg.get("mip_gap", 1e-3)
        solver.options["OutputFlag"] = 1 if solver_cfg.get("verbose", True) else 0
        solver.options["NonConvex"] = 2
        solver.options["Cuts"] = 2
        solver.options["Presolve"] = 2
    # warmstart=True 会对支持的求解器启用 MIP start
    return solver.solve(model, tee=solver_cfg.get("verbose", True), warmstart=warmstart)


def extract_solver_info(results: Any) -> dict:
    """从 Pyomo SolverResults 中提取可序列化的求解信息（时间/状态等）。"""
    info = {}
    try:
        sol = getattr(results, "solver", None)
        if sol is None:
            return info
        # status / termination_condition 通常是 enum，转成 str
        info["status"] = str(getattr(sol, "status", ""))
        info["termination_condition"] = str(getattr(sol, "termination_condition", ""))
        # 不同接口字段名略有差异，尽量都取到
        if hasattr(sol, "time"):
            info["time"] = float(sol.time) if sol.time is not None else None
        if hasattr(sol, "wallclock_time"):
            info["wallclock_time"] = float(sol.wallclock_time) if sol.wallclock_time is not None else None
        if hasattr(sol, "message"):
            msg = getattr(sol, "message", None)
            info["message"] = str(msg) if msg is not None else None
    except Exception:
        # 不能让统计信息影响主流程
        return info
    return info


# =========================
# main
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="GridRAG Online Pipeline (zero-shot)")
    p.add_argument("--task", type=str, default="vvc", choices=["vvc", "ed", "joint"], help="任务类型")
    p.add_argument("--network", type=str, default="ieee69", choices=["ieee13", "ieee33", "ieee69", "ieee123"], help="网络")
    p.add_argument("--scenario_file", type=str, default=None, help="在线DER场景csv路径（默认 data/online_inf/scenario_online_{bus}.csv）")
    p.add_argument("--ev_file", type=str, default=None, help="在线EV场景csv路径（joint任务可选）")
    p.add_argument("--use_diffusion", action="store_true", help="强制开启diffusion refine")
    p.add_argument("--no_diffusion", action="store_true", help="强制关闭diffusion refine")
    p.add_argument("--diffusion_mode", type=str, default="always", choices=["auto", "always", "off"],
                   help="diffusion策略：auto=按相似度阈值触发；always=总是触发；off=关闭")
    p.add_argument("--distance_threshold", type=float, default=0.8, help="auto模式下，distance大于阈值则触发diffusion")
    p.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "opt_results" / "online"), help="在线结果保存目录")
    return p.parse_args()


def _default_online_paths(network: str) -> Tuple[Path, Optional[Path]]:
    bus_map = {"ieee13": "13", "ieee33": "33", "ieee69": "69", "ieee123": "123"}
    suffix = bus_map[network.lower()]
    scenario = PROJECT_ROOT / "data" / "online_inf" / f"scenario_online_{suffix}.csv"
    ev = PROJECT_ROOT / "data" / "online_inf" / f"ev_profiles_online_{suffix}.csv"
    return scenario, ev


def main():
    args = parse_args()

    timer = StageTimer()

    task = args.task.lower()
    network_name = args.network.lower()

    # 1) config & network
    config = get_network_config(network_name)
    network = get_network_instance(network_name, config)
    s_base = float(config["network"]["s_base"])

    # 2) 在线文件路径
    default_scenario, default_ev = _default_online_paths(network_name)
    scenario_file = Path(args.scenario_file) if args.scenario_file else default_scenario
    ev_file = Path(args.ev_file) if args.ev_file else (default_ev if default_ev.exists() else None)

    if not scenario_file.exists():
        raise FileNotFoundError(f"在线场景文件不存在: {scenario_file}")
    if task == "joint" and ev_file is None:
        raise FileNotFoundError("joint 任务需要 ev_file（或默认 ev_profiles_online_xx.csv 存在）")

    # 3) 初始化检索器并检索
    t_retr = timer.tic()
    retriever = ScenarioRetriever(network_name)
    if hasattr(retriever, "retrieve_with_info"):
        best_id, best_dist, query_feat = retriever.retrieve_with_info(scenario_file=scenario_file, ev_file=ev_file)
    else:
        best_id = retriever.retrieve_top1(scenario_file=scenario_file, ev_file=ev_file)
        best_dist, query_feat = float("inf"), np.zeros(1)
    timer.toc("retrieval", t_retr)

    # 4) 加载历史结果
    t_hist = timer.tic()
    loader = ResultLoader(project_root=PROJECT_ROOT)
    hist = loader.load(network=network_name, task=task, scenario_id=str(best_id))
    timer.toc("load_retrieved_result", t_hist)

    # 5) warm-start 提取（标准化）
    t_ws = timer.tic()
    extractor = WarmStartExtractor(task=task, s_base=s_base)
    warm = extractor.extract(hist)
    timer.toc("extract_warm_start", t_ws)

    # 6) diffusion策略判定
    mode = args.diffusion_mode
    if args.no_diffusion:
        mode = "off"
    if args.use_diffusion:
        mode = "always"

    use_diff = False
    if mode == "always":
        use_diff = True
    elif mode == "off":
        use_diff = False
    else:
        use_diff = (best_dist > float(args.distance_threshold))

    t_diff = timer.tic()
    diff_info = {}
    if use_diff:
        warm_refined, diff_info = diffusion_refine(
            warm_start=warm,
            retrieved_json=hist,
            query_features=query_feat,
            best_distance=best_dist,
            task=task,
            network=network_name,
            s_base=s_base,
            diffusion_cfg=None,
        )
    else:
        warm_refined = warm
    timer.toc("diffusion_refine", t_diff)
    # 修正：如果有训练时间，从diffusion_refine中扣除
    if diff_info.get("training_time_sec", 0) > 0:
        timer.times["diffusion_refine"] -= diff_info["training_time_sec"]
        timer.times["diffusion_training_offline"] = diff_info["training_time_sec"]

    # 7) 构建模型 + apply warm-start + 求解
    solver_name = config["optimization"]["solver"]["name"]
    solver_cfg = config["optimization"]["solver"]

    if task == "vvc":
        # VVC 两阶段：day-ahead + real-time
        n_periods = config["optimization"]["day_ahead"]["n_periods"]
        t_load = timer.tic()
        scenario_data = load_online_scenario(config, network_name, scenario_file, n_periods=n_periods)
        timer.toc("load_online_data", t_load)

        da = create_day_ahead_model(config)
        t_build_da = timer.tic()
        da.build_model(network, scenario_data)
        timer.toc("build_day_ahead_model", t_build_da)
        t_apply_da = timer.tic()
        apply_warm_start_vvc_day_ahead(da, warm_refined)
        timer.toc("apply_warm_start_day_ahead", t_apply_da)
        t_solve_da = timer.tic()
        da_solver_res = solve_pyomo(da.model, solver_name, solver_cfg, warmstart=True)
        timer.toc("solve_day_ahead", t_solve_da)
        da_solver_info = extract_solver_info(da_solver_res)
        t_post_da = timer.tic()
        da_results = da.get_results()
        timer.toc("postprocess_day_ahead", t_post_da)

        rt = create_real_time_model(config)
        t_build_rt = timer.tic()
        rt.build_model(network, scenario_data, day_ahead_results=da_results)
        timer.toc("build_real_time_model", t_build_rt)
        t_apply_rt = timer.tic()
        apply_warm_start_vvc_real_time(rt, warm_refined)
        timer.toc("apply_warm_start_real_time", t_apply_rt)
        t_solve_rt = timer.tic()
        rt_solver_res = solve_pyomo(rt.model, solver_name, solver_cfg, warmstart=True)
        timer.toc("solve_real_time", t_solve_rt)
        rt_solver_info = extract_solver_info(rt_solver_res)
        t_post_rt = timer.tic()
        rt_results = rt.get_results()
        timer.toc("postprocess_real_time", t_post_rt)

        final = {
            "task": "vvc",
            "network": network_name,
            "retrieved_scenario_id": best_id,
            "retrieval_distance": best_dist,
            "use_diffusion": use_diff,
            "diffusion_info": diff_info,
            "timestamp": datetime.now().isoformat(),
            "day_ahead": da_results,
            "real_time": rt_results,
            "solver": {
                "day_ahead": da_solver_info,
                "real_time": rt_solver_info,
            },
            "timing": timer.times,
            "total_time_sec": timer.total(),
        }

    elif task == "ed":
        n_periods = config["optimization"]["ed"]["n_periods"]
        t_load = timer.tic()
        scenario_data = load_online_scenario(config, network_name, scenario_file, n_periods=n_periods)
        timer.toc("load_online_data", t_load)

        ed = create_ed_model(config)
        t_build = timer.tic()
        ed.build_model(network, scenario_data)
        timer.toc("build_ed_model", t_build)
        t_apply = timer.tic()
        apply_warm_start_ed(ed, warm_refined)
        timer.toc("apply_warm_start_ed", t_apply)
        t_solve = timer.tic()
        ed_solver_res = solve_pyomo(ed.model, solver_name, solver_cfg, warmstart=True)
        timer.toc("solve_ed", t_solve)
        ed_solver_info = extract_solver_info(ed_solver_res)
        t_post = timer.tic()
        results = ed.get_results()
        timer.toc("postprocess_ed", t_post)

        final = {
            "task": "ed",
            "network": network_name,
            "retrieved_scenario_id": best_id,
            "retrieval_distance": best_dist,
            "use_diffusion": use_diff,
            "diffusion_info": diff_info,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "solver": ed_solver_info,
            "timing": timer.times,
            "total_time_sec": timer.total(),
        }

    else:  # joint
        n_periods = config["optimization"]["joint"]["n_periods"]
        t_load = timer.tic()
        scenario_data = load_online_scenario(config, network_name, scenario_file, n_periods=n_periods)
        timer.toc("load_online_data", t_load)

        # joint 模型沿用 main_joint 的数据加载流程：load_* 后 build_model()
        from main_joint import load_ev_data  # 复用你们已有实现
        # 临时改 config 路径，让 load_ev_data 能在 online_inf 找到文件
        cfg2 = deepcopy(config)
        cfg2.setdefault("paths", {})
        cfg2["paths"]["profiles"] = Path(ev_file.parent)
        t_ev = timer.tic()
        ev_data = load_ev_data(cfg2, ev_file=ev_file.name)
        timer.toc("load_ev_data", t_ev)

        joint = create_joint_model(config)
        t_load_joint = timer.tic()
        joint.load_network(network)
        joint.load_scenario_data(scenario_data)
        joint.load_ev_data(ev_data)
        timer.toc("load_joint_inputs", t_load_joint)

        # JointOptModel.build_model() 在你们当前实现里不接收任何参数（依赖 load_* 先缓存输入）。
        # 为了兼容可能的旧接口，这里做一次签名检测：
        import inspect
        t_build = timer.tic()
        try:
            n_params = len(inspect.signature(joint.build_model).parameters)
            if n_params <= 1:
                joint.build_model()
            else:
                # 兼容旧版本：build_model(network, scenario_data, ev_data)
                joint.build_model(network, scenario_data, ev_data)
        except Exception:
            # 兜底：按新接口调用
            joint.build_model()
        timer.toc("build_joint_model", t_build)
        t_apply = timer.tic()
        apply_warm_start_joint(joint, warm_refined, s_base=s_base)
        timer.toc("apply_warm_start_joint", t_apply)
        t_solve = timer.tic()
        joint_solver_res = solve_pyomo(joint.model, solver_name, solver_cfg, warmstart=True)
        timer.toc("solve_joint", t_solve)
        joint_solver_info = extract_solver_info(joint_solver_res)
        t_post = timer.tic()
        results = joint.get_results()
        timer.toc("postprocess_joint", t_post)

        final = {
            "task": "joint",
            "network": network_name,
            "retrieved_scenario_id": best_id,
            "retrieval_distance": best_dist,
            "use_diffusion": use_diff,
            "diffusion_info": diff_info,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "solver": joint_solver_info,
            "timing": timer.times,
            "total_time_sec": timer.total(),
        }

    # 8) 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = get_result_filename(network_name, f"online_{task}", "000", suffix=f"{best_id}")
    # 由于 get_result_filename 固定模板，我们这里自定义在线命名更清晰
    out_name = f"{network_name}_online_{task}_query_{scenario_file.stem}_retr_{best_id}.json"
    out_path = out_dir / out_name
    t_save = timer.tic()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert_for_json(final), f, indent=2, ensure_ascii=False)
    timer.toc("save_json", t_save)

    print("\n" + "=" * 60)
    print("✅ Online pipeline finished")
    print(f"task={task}, network={network_name}")
    # timing summary
    print("timing (sec):")
    for k, v in sorted(timer.times.items(), key=lambda kv: kv[0]):
        print(f"  {k}: {v:.4f}")
    print(f"  total: {timer.total():.4f}")
    print(f"retrieved={best_id}, distance={best_dist:.4f}, diffusion={use_diff}")
    if diff_info:
        print(f"  diffusion_noise_level={diff_info.get('noise_level', 'N/A')}")
        print(f"  diffusion_quality_passed={diff_info.get('quality_passed', 'N/A')}")
        if diff_info.get('metrics'):
            print(f"  diffusion_metrics={diff_info['metrics']}")
    print(f"saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
