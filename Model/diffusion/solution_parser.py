# -*- coding: utf-8 -*-
"""Model.diffusion.solution_parser

将优化结果 JSON 中的连续/离散量与 numpy 矩阵 (96, D) 相互转换。

设计原则：
- 训练和推理都以"原始 JSON 单位"(MW, MVar, MWh, ratio 等) 作为基准空间
- MinMaxScaler 在此空间上 fit；diffusion 模型在 [-1,1] 空间上训练/推理
- 推理时直接从 raw JSON 提取，SDEdit 后由本模块将 refined 矩阵重组为
  与 WarmStartExtractor.extract() 相同格式的 warm-start 字典
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Column specifications per task
# ===========================================================================
# 每个元素: (field_path, prefix, is_scalar_per_timestep)
#   field_path: 在 JSON 中的访问路径 (tuple of keys)
#   prefix: 列名前缀，用于反向重组
#   is_scalar: True → shape (96,), False → shape (96, n_dev)

# ---- VVC (Task A) --------------------------------------------------------
# 连续: pv_reactive.q_mvar, wt_reactive.q_mvar, svc_reactive.q_mvar
# 离散: oltc.tap, sc.stage
VVC_CONTINUOUS = [
    (("pv_reactive", "q_mvar"),  "pv_q",  False),
    (("wt_reactive", "q_mvar"),  "wt_q",  False),
    (("svc_reactive", "q_mvar"), "svc_q", False),
]

VVC_DISCRETE = [
    (("oltc", "tap"),   "oltc_tap"),
    (("sc", "stage"),   "sc_stage"),
]

# ---- ED (Task B) ---------------------------------------------------------
# 连续: ess.{charge,discharge,soc}, pv.{curtailment,reactive}, wt.reactive, grid.power
# 离散: ess.mode, reconfiguration.{status,changes}
ED_CONTINUOUS = [
    (("ess", "charge_mw"),       "ess_ch",   False),
    (("ess", "discharge_mw"),    "ess_dis",  False),
    (("ess", "soc_mwh"),         "ess_soc",  False),
    (("pv", "curtailment_mw"),   "pv_curt",  False),
    (("pv", "reactive_mvar"),    "pv_q",     False),
    (("wt", "reactive_mvar"),    "wt_q",     False),
    (("grid", "power_mw"),       "grid_p",   True),
]

ED_DISCRETE = [
    (("ess", "mode"),                "ess_mode"),
    (("reconfiguration", "status"),  "tie_status"),
    (("reconfiguration", "changes"), "tie_changes"),
]

# ---- Joint (Task C) -------------------------------------------------------
# 连续: ess.{charge,discharge,soc}, ev.{power,cut_ratio}, grid.power
# 离散: oltc.tap_position, tie_switches.status, ev.interruptions
# 不参与 diffusion: ev.energy_mwh, ev.cut_tier1/2/3 (用检索原值)
JOINT_CONTINUOUS = [
    (("ess", "charge_mw"),     "ess_ch",      False),
    (("ess", "discharge_mw"),  "ess_dis",     False),
    (("ess", "soc_mwh"),       "ess_soc",     False),
    (("ev", "power_mw"),       "ev_p",        False),
    (("ev", "cut_ratio"),      "ev_cut",      False),
    (("grid", "power_mw"),     "grid_p",      True),
]

JOINT_DISCRETE = [
    (("oltc", "tap_position"),    "oltc_tap"),
    (("tie_switches", "status"),  "tie_status"),
    (("ev", "interruptions"),     "ev_int"),
]

# Joint 中不参与 diffusion 但需要原样保留进 warm-start 的字段
JOINT_PASSTHROUGH = [
    (("ev", "energy_mwh"),   "ev_energy"),
    (("ev", "cut_tier1"),    "ev_tier1"),
    (("ev", "cut_tier2"),    "ev_tier2"),
    (("ev", "cut_tier3"),    "ev_tier3"),
]

TASK_SPECS = {
    "vvc":   (VVC_CONTINUOUS,   VVC_DISCRETE,   []),
    "ed":    (ED_CONTINUOUS,    ED_DISCRETE,     []),
    "joint": (JOINT_CONTINUOUS, JOINT_DISCRETE,  JOINT_PASSTHROUGH),
}


# ===========================================================================
# Helper: 从嵌套 dict 按路径取值
# ===========================================================================

def _get_nested(d: dict, path: tuple) -> Any:
    """从 dict 中按 key 路径取值，找不到返回 None。"""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _resolve_root(json_data: dict, task: str) -> dict:
    """ED 和 Joint 的数据字段嵌套在 'results' 下，VVC 在顶层。"""
    if task == "vvc":
        return json_data
    # ED / Joint: 尝试取 "results"
    if "results" in json_data and isinstance(json_data["results"], dict):
        return json_data["results"]
    return json_data


# ===========================================================================
# SolutionParser
# ===========================================================================

class SolutionParser:
    """优化解 JSON ↔ numpy (96, D) 矩阵的双向转换器。"""

    def __init__(self, task: str, n_periods: int = 96):
        task = task.lower()
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.n_periods = n_periods
        self.cont_spec, self.disc_spec, self.pass_spec = TASK_SPECS[task]

    # ------------------------------------------------------------------
    # JSON → numpy
    # ------------------------------------------------------------------

    def parse_json(
        self, json_data: dict
    ) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
        """从一个 opt_results JSON 中提取连续矩阵和离散字典。

        Returns
        -------
        continuous : np.ndarray, shape (n_periods, D)
            所有连续控制量拼接而成的矩阵（原始单位）
        discrete : dict
            离散量原值（按 prefix 索引）
        col_names : list[str]
            每列的名字，用于反向重组（如 "ess_ch_0", "ess_ch_1", ...）
        """
        root = _resolve_root(json_data, self.task)
        T = self.n_periods

        cols = []       # list of 1-D np arrays, each length T
        col_names = []
        
        for path, prefix, is_scalar in self.cont_spec:
            val = _get_nested(root, path)
            if val is None:
                # 该字段不存在（例如某些网络没有 wt），跳过
                continue
            arr = np.array(val, dtype=np.float64)
            if is_scalar:
                # shape (T,) → 单列
                arr = arr.reshape(T, 1) if arr.ndim == 1 else arr[:T, :1]
                cols.append(arr[:T, 0])
                col_names.append(f"{prefix}_0")
            else:
                # shape (T, n_dev)
                if arr.ndim == 1:
                    arr = arr.reshape(T, 1)
                for j in range(arr.shape[1]):
                    cols.append(arr[:T, j])
                    col_names.append(f"{prefix}_{j}")

        continuous = np.column_stack(cols) if cols else np.empty((T, 0))

        # 离散量
        discrete = {}
        for path, prefix in self.disc_spec:
            val = _get_nested(root, path)
            if val is not None:
                discrete[prefix] = val

        # passthrough（joint 的 energy/tier）
        for path, prefix in self.pass_spec:
            val = _get_nested(root, path)
            if val is not None:
                discrete[prefix] = val

        return continuous, discrete, col_names

    def get_feature_size(self, json_data: dict) -> int:
        """快速获取连续维度 D（不构建完整矩阵）。"""
        cont, _, col_names = self.parse_json(json_data)
        return cont.shape[1]

    # ------------------------------------------------------------------
    # numpy → warm-start dict (与 WarmStartExtractor 输出格式一致)
    # ------------------------------------------------------------------

    def to_warmstart(
        self,
        continuous: np.ndarray,
        discrete: Dict[str, Any],
        col_names: List[str],
        s_base: float,
    ) -> Dict[str, Any]:
        """将 refined 连续矩阵 + 离散量重组为 warm-start 字典。

        输出格式与 WarmStartExtractor.extract() 一致，可直接传入
        main_online 中的 apply_warm_start_* 函数。
        """
        if self.task == "vvc":
            return self._to_ws_vvc(continuous, discrete, col_names, s_base)
        elif self.task == "ed":
            return self._to_ws_ed(continuous, discrete, col_names)
        elif self.task == "joint":
            return self._to_ws_joint(continuous, discrete, col_names, s_base)
        raise ValueError(f"Unknown task: {self.task}")

    # ---- VVC ----
    def _to_ws_vvc(self, cont, disc, names, s_base):
        T = self.n_periods
        idx = _build_col_index(names)

        def _extract_2d(prefix):
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [[float(cont[t, c]) / s_base for c in cols_i] for t in range(T)]

        ws = {
            "task": "vvc",
            "oltc": {"tap": disc.get("oltc_tap")},
            "sc": {"stage": disc.get("sc_stage")},
            "rt": {
                "pv":  {"q_pu": _extract_2d("pv_q")},
                "wt":  {"q_pu": _extract_2d("wt_q")},
                "svc": {"q_pu": _extract_2d("svc_q")},
            },
        }
        return ws

    # ---- ED ----
    def _to_ws_ed(self, cont, disc, names):
        T = self.n_periods
        idx = _build_col_index(names)

        def _extract_2d(prefix):
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [[float(cont[t, c]) for c in cols_i] for t in range(T)]

        def _extract_1d(prefix):
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [float(cont[t, cols_i[0]]) for t in range(T)]

        ws = {"task": "ed"}

        ws["ess"] = {
            "charge_mw":    _extract_2d("ess_ch"),
            "discharge_mw": _extract_2d("ess_dis"),
            "soc_mwh":      _extract_2d("ess_soc"),
            "mode":         disc.get("ess_mode"),
        }
        ws["pv"] = {
            "curtailment_mw": _extract_2d("pv_curt"),
            "reactive_mvar":  _extract_2d("pv_q"),
        }
        ws["wt"] = {
            "reactive_mvar": _extract_2d("wt_q"),
        }
        ws["grid"] = {
            "power_mw": _extract_1d("grid_p"),
        }
        ws["reconfiguration"] = {
            "status":  disc.get("tie_status"),
            "changes": disc.get("tie_changes"),
        }
        # 为兼容 WarmStartExtractor 的 key
        ws["tie_switches"] = ws["reconfiguration"]

        return ws

    # ---- Joint ----
    def _to_ws_joint(self, cont, disc, names, s_base):
        T = self.n_periods
        idx = _build_col_index(names)

        def _extract_2d_pu(prefix):
            """从原始 MW → pu"""
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [[float(cont[t, c]) / s_base for c in cols_i] for t in range(T)]

        def _extract_2d_raw(prefix):
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [[float(cont[t, c]) for c in cols_i] for t in range(T)]

        def _extract_1d_pu(prefix):
            cols_i = idx.get(prefix, [])
            if not cols_i:
                return None
            return [float(cont[t, cols_i[0]]) / s_base for t in range(T)]

        ws = {"task": "joint"}

        ws["oltc"] = {"tap_position": disc.get("oltc_tap")}

        ws["ess"] = {
            "charge_pu":    _extract_2d_pu("ess_ch"),
            "discharge_pu": _extract_2d_pu("ess_dis"),
            "soc_mwh":      _extract_2d_raw("ess_soc"),
        }

        ws["ev"] = {
            "power_pu":   _extract_2d_pu("ev_p"),
            "cut_ratio":  _extract_2d_raw("ev_cut"),
            # passthrough: 直接使用检索解中的原值
            "energy_mwh": disc.get("ev_energy"),
            "cut_tier1":  disc.get("ev_tier1"),
            "cut_tier2":  disc.get("ev_tier2"),
            "cut_tier3":  disc.get("ev_tier3"),
        }

        ws["tie_switches"] = {"status": disc.get("tie_status")}

        grid_pu = _extract_1d_pu("grid_p")
        ws["grid"] = {"power_pu": grid_pu}

        return ws


# ===========================================================================
# 辅助：根据 col_names 建立 prefix → [col_indices] 映射
# ===========================================================================

def _build_col_index(col_names: List[str]) -> Dict[str, List[int]]:
    """col_names 形如 ['ess_ch_0', 'ess_ch_1', 'grid_p_0', ...]
    返回 {'ess_ch': [0,1], 'grid_p': [2], ...}
    """
    idx: Dict[str, List[int]] = {}
    for i, name in enumerate(col_names):
        # 去掉最后的 _数字 部分
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
        else:
            prefix = name
        idx.setdefault(prefix, []).append(i)
    return idx


# ===========================================================================
# 批量提取：从整个 opt_results 目录读取所有 JSON
# ===========================================================================

def load_all_solutions(
    opt_dir: str,
    task: str,
    network: str,
    n_periods: int = 96,
) -> Tuple[np.ndarray, List[str]]:
    """扫描 opt_results/{task}/ 目录，提取所有匹配 {network} 的 JSON。

    Returns
    -------
    all_continuous : np.ndarray, shape (N, n_periods, D)
    col_names : list[str]
    """
    import json
    from pathlib import Path

    task_dir = Path(opt_dir) / task
    if not task_dir.exists():
        raise FileNotFoundError(f"opt_results directory not found: {task_dir}")

    parser = SolutionParser(task, n_periods)
    arrays = []
    col_names_ref = None

    for f in sorted(task_dir.glob(f"{network}_*_results.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        cont, _, col_names = parser.parse_json(data)
        if cont.shape[1] == 0:
            continue
        if col_names_ref is None:
            col_names_ref = col_names
        arrays.append(cont)

    if not arrays:
        raise ValueError(
            f"No solutions found for task={task}, network={network} in {task_dir}"
        )

    all_cont = np.stack(arrays, axis=0)  # (N, T, D)
    return all_cont, col_names_ref
