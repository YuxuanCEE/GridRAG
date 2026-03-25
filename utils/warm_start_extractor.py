# -*- coding: utf-8 -*-
"""utils.warm_start_extractor

从离线保存的结果JSON中提取 warm-start 所需的“控制量/决策量”。

注意：
- VVC 的实时无功结果在JSON里是 MVar；但 real_time 模型内部变量为 pu，
  因此这里会根据 s_base 自动转换为 pu（需要调用方传入 s_base）。
- Joint 的多数功率输出在JSON里是 MW；模型内部为 pu，这里同样会转换为 pu（需要 s_base）。
- ED 模型内部即使用 MW/MWh 表示，通常不需要缩放。

本模块只负责“提取并标准化字段”，不负责把值写回 Pyomo 变量（apply 逻辑放在 main_online）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WarmStartExtractor:
    task: str
    s_base: Optional[float] = None  # MVA，用于MW/MVar <-> pu换算（vvc/joint需要）

    def extract(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        task = self.task.lower()
        if task == "vvc":
            return self._extract_vvc(json_data)
        if task == "ed":
            return self._extract_ed(json_data)
        if task == "joint":
            return self._extract_joint(json_data)
        raise ValueError(f"Unknown task: {self.task}")

    def _require_s_base(self) -> float:
        if self.s_base is None:
            raise ValueError("s_base is required for this task (vvc/joint). Please pass network s_base.")
        return float(self.s_base)

    def _extract_vvc(self, d: Dict[str, Any]) -> Dict[str, Any]:
        s_base = self._require_s_base()
        out: Dict[str, Any] = {
            "task": "vvc",
            "oltc": {},
            "sc": {},
            "rt": {"pv": {}, "wt": {}, "svc": {}},
        }

        # Day-ahead
        if "oltc" in d and isinstance(d["oltc"], dict):
            out["oltc"]["tap"] = d["oltc"].get("tap")
        if "sc" in d and isinstance(d["sc"], dict):
            out["sc"]["stage"] = d["sc"].get("stage")

        # Real-time: JSON为MVar -> pu
        if "pv_reactive" in d:
            q = d["pv_reactive"].get("q_mvar")
            out["rt"]["pv"]["q_pu"] = None if q is None else [[float(x) / s_base for x in row] for row in q]
        if "wt_reactive" in d:
            q = d["wt_reactive"].get("q_mvar")
            out["rt"]["wt"]["q_pu"] = None if q is None else [[float(x) / s_base for x in row] for row in q]
        if "svc_reactive" in d:
            q = d["svc_reactive"].get("q_mvar")
            out["rt"]["svc"]["q_pu"] = None if q is None else [[float(x) / s_base for x in row] for row in q]

        return out

    def _extract_ed(self, d: Dict[str, Any]) -> Dict[str, Any]:
        # ED结果由模型 get_results() 输出，单位一般为MW/MWh/MVar，无需缩放
        out: Dict[str, Any] = {"task": "ed"}

        for k in ["ess", "pv", "wt", "grid", "tie_switches", "reconfiguration"]:
            if k in d:
                out[k] = d[k]
            elif "results" in d and isinstance(d["results"], dict) and k in d["results"]:
                out[k] = d["results"][k]

        # 兼容 main_ed 保存格式（通常外层是 {scenario_id, network, task, results, statistics}）
        if "results" in d and isinstance(d["results"], dict):
            out.setdefault("results", d["results"])
        return out

    def _extract_joint(self, d: Dict[str, Any]) -> Dict[str, Any]:
        s_base = self._require_s_base()
        out: Dict[str, Any] = {"task": "joint"}

        # Joint JSON 中字段可能在顶层，也可能嵌套在 "results" 下
        r = d.get("results", d) if "results" in d and isinstance(d["results"], dict) else d

        # 关键控制量：oltc tap、ess charge/discharge、ev power/cut_ratio、tie switches、grid power
        if "oltc" in r:
            out["oltc"] = {"tap_position": r["oltc"].get("tap_position")}

        # 功率：MW -> pu
        def mw_to_pu(arr):
            if arr is None:
                return None
            return [[float(x) / s_base for x in row] for row in arr]

        if "ess" in r:
            out["ess"] = {
                "charge_pu": mw_to_pu(r["ess"].get("charge_mw")),
                "discharge_pu": mw_to_pu(r["ess"].get("discharge_mw")),
                "soc_mwh": r["ess"].get("soc_mwh"),  # E_soc通常本身是能量量纲（模型里可能是MWh），由apply时决定是否写
            }

        if "ev" in r:
            out["ev"] = {
                "power_pu": mw_to_pu(r["ev"].get("power_mw")),
                "cut_ratio": r["ev"].get("cut_ratio"),
                # ✅ 新增：提取能量状态（用于满足能量动态约束）
                "energy_mwh": r["ev"].get("energy_mwh"),
                # ✅ 新增：提取阶梯削减变量（用于满足tier sum约束）
                "cut_tier1": r["ev"].get("cut_tier1"),
                "cut_tier2": r["ev"].get("cut_tier2"),
                "cut_tier3": r["ev"].get("cut_tier3"),
            }

        if "tie_switches" in r:
            out["tie_switches"] = {"status": r["tie_switches"].get("status")}

        if "grid" in r:
            # grid.power_mw 可能是一维(n_periods,)；也可能二维
            p = r["grid"].get("power_mw")
            if p is None:
                out["grid"] = {"power_pu": None}
            elif isinstance(p, list) and len(p) > 0 and isinstance(p[0], list):
                out["grid"] = {"power_pu": mw_to_pu(p)}
            else:
                out["grid"] = {"power_pu": [float(x)/s_base for x in p]}

        return out
