# -*- coding: utf-8 -*-
"""
RL 辅助函数

- VVC约束违反度检查
- 动作后处理（离散取整、连续裁剪）
- 指标计算与报告输出
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config_networks import get_network_config


class VVCConstraintChecker:
    """VVC 约束违反度检查器"""

    def __init__(self, network_name: str, scenario_id: str, project_root: Path = None):
        self.network_name = network_name.lower()
        self.scenario_id = scenario_id
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root

        self.config = get_network_config(network_name, project_root)
        self.n_periods = self.config["optimization"]["day_ahead"]["n_periods"]

        self.oltc_cfg = self.config["devices"]["oltc"]
        self.sc_cfg = self.config["devices"]["sc"]
        self.pv_cfg = self.config["devices"]["pv"]
        self.wt_cfg = self.config["devices"]["wt"]
        self.svc_cfg = self.config["devices"]["svc"]

        self.n_pv = len(self.pv_cfg["buses"])
        self.n_wt = len(self.wt_cfg["buses"])
        self.n_sc = len(self.sc_cfg["buses"])
        self.n_svc = len(self.svc_cfg["buses"])
        self.v_min = self.config["network"]["v_min"]
        self.v_max = self.config["network"]["v_max"]

    # ---------- 加载 ----------
    def load_rl_results(self, results_path: Path = None) -> Dict:
        if results_path is None:
            results_path = self.project_root / "opt_results" / "rl" / \
                           f"{self.network_name}_vvc_scenario_{self.scenario_id}_rl_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"RL结果文件不存在: {results_path}")
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_ground_truth(self) -> Dict:
        gt_path = self.project_root / "opt_results" / "vvc" / \
                  f"{self.network_name}_vvc_scenario_{self.scenario_id}_results.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT结果文件不存在: {gt_path}")
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ---------- 各项检查 ----------
    def check_oltc_bounds(self, oltc_tap: np.ndarray) -> Dict:
        tap_min, tap_max = self.oltc_cfg["tap_min"], self.oltc_cfg["tap_max"]
        below = np.sum(oltc_tap < tap_min)
        above = np.sum(oltc_tap > tap_max)
        total = len(oltc_tap)
        viol = int(below + above)
        return {"violation_count": viol, "total_checks": total,
                "violation_pct": 100.0 * viol / total}

    def check_oltc_actions(self, oltc_tap: np.ndarray) -> Dict:
        max_actions = self.oltc_cfg["max_daily_actions"]
        changes = np.sum(np.abs(np.diff(oltc_tap)) > 0.5)
        exceeded = max(0, int(changes) - max_actions)
        return {"n_changes": int(changes), "max_allowed": max_actions,
                "exceeded": exceeded}

    def check_sc_bounds(self, sc_stage: np.ndarray) -> Dict:
        n_stages = self.sc_cfg["n_stages"]
        below = np.sum(sc_stage < 0)
        above = np.sum(sc_stage > n_stages)
        total = sc_stage.size
        viol = int(below + above)
        return {"violation_count": viol, "total_checks": total,
                "violation_pct": 100.0 * viol / total if total > 0 else 0}

    def check_sc_actions(self, sc_stage: np.ndarray) -> Dict:
        max_actions = self.sc_cfg["max_daily_actions"]
        results = []
        total_exceeded = 0
        for k in range(self.n_sc):
            changes = np.sum(np.abs(np.diff(sc_stage[:, k])) > 0.5)
            exceeded = max(0, int(changes) - max_actions)
            total_exceeded += exceeded
            results.append({"sc_id": k, "n_changes": int(changes),
                            "max_allowed": max_actions, "exceeded": exceeded})
        return {"details": results, "total_exceeded": total_exceeded}

    def check_pv_q_bounds(self, pv_q_mvar: np.ndarray, pv_capacity_factors: np.ndarray) -> Dict:
        """检查 |Q_pv| ≤ sqrt(S² - P²)"""
        viol_count = 0
        total = 0
        for i in range(self.n_pv):
            s_pv = self.pv_cfg["capacity"][i]  # MW = MVA
            for t in range(self.n_periods):
                p_pv = pv_capacity_factors[t, i] * s_pv
                q_max = np.sqrt(max(s_pv ** 2 - p_pv ** 2, 0))
                if abs(pv_q_mvar[t, i]) > q_max + 1e-4:
                    viol_count += 1
                total += 1
        return {"violation_count": viol_count, "total_checks": total,
                "violation_pct": 100.0 * viol_count / total if total > 0 else 0}

    def check_wt_q_bounds(self, wt_q_mvar: np.ndarray, wt_capacity_factors: np.ndarray) -> Dict:
        viol_count = 0
        total = 0
        for i in range(self.n_wt):
            s_wt = self.wt_cfg["capacity"][i]
            for t in range(self.n_periods):
                p_wt = wt_capacity_factors[t, i] * s_wt
                q_max = np.sqrt(max(s_wt ** 2 - p_wt ** 2, 0))
                if abs(wt_q_mvar[t, i]) > q_max + 1e-4:
                    viol_count += 1
                total += 1
        return {"violation_count": viol_count, "total_checks": total,
                "violation_pct": 100.0 * viol_count / total if total > 0 else 0}

    def check_svc_bounds(self, svc_q_mvar: np.ndarray) -> Dict:
        q_min = self.svc_cfg["q_min"]
        q_max = self.svc_cfg["q_max"]
        below = np.sum(svc_q_mvar < q_min - 1e-4)
        above = np.sum(svc_q_mvar > q_max + 1e-4)
        total = svc_q_mvar.size
        viol = int(below + above)
        return {"violation_count": viol, "total_checks": total,
                "violation_pct": 100.0 * viol / total if total > 0 else 0}

    def check_voltage_bounds(self, voltage: np.ndarray) -> Dict:
        if voltage is None or voltage.size == 0:
            return {"status": "skipped"}
        below = np.sum(voltage < self.v_min - 1e-4)
        above = np.sum(voltage > self.v_max + 1e-4)
        total = voltage.size
        viol = int(below + above)
        return {"violation_count": viol, "total_checks": total,
                "violation_pct": 100.0 * viol / total,
                "v_min": float(np.min(voltage)),
                "v_max": float(np.max(voltage))}

    # ---------- 汇总 ----------
    def check_all(self, results: Dict, pv_cf: np.ndarray = None,
                  wt_cf: np.ndarray = None) -> Dict:
        """
        Args:
            results: RL结果字典（应包含 oltc.tap, sc.stage, pv_reactive.q_mvar 等）
            pv_cf: (96, n_pv) PV容量因子
            wt_cf: (96, n_wt) WT容量因子
        """
        oltc_tap = np.array(results["oltc"]["tap"])
        sc_stage = np.array(results["sc"]["stage"])
        if sc_stage.ndim == 1:
            sc_stage = sc_stage.reshape(-1, 1)
        pv_q = np.array(results["pv_reactive"]["q_mvar"])
        if pv_q.ndim == 1 and self.n_pv > 0:
            pv_q = pv_q.reshape(-1, 1)
        wt_q = np.array(results["wt_reactive"]["q_mvar"])
        if wt_q.ndim == 1 and self.n_wt > 0:
            wt_q = wt_q.reshape(-1, 1)
        svc_q = np.array(results["svc_reactive"]["q_mvar"])
        if svc_q.ndim == 1:
            svc_q = svc_q.reshape(-1, 1)

        checks = {}
        checks["oltc_bounds"] = self.check_oltc_bounds(oltc_tap)
        checks["oltc_actions"] = self.check_oltc_actions(oltc_tap)
        checks["sc_bounds"] = self.check_sc_bounds(sc_stage)
        checks["sc_actions"] = self.check_sc_actions(sc_stage)

        if pv_cf is not None and self.n_pv > 0:
            checks["pv_q_bounds"] = self.check_pv_q_bounds(pv_q, pv_cf)
        else:
            checks["pv_q_bounds"] = {"status": "skipped"}

        if wt_cf is not None and self.n_wt > 0:
            checks["wt_q_bounds"] = self.check_wt_q_bounds(wt_q, wt_cf)
        else:
            checks["wt_q_bounds"] = {"status": "skipped"}

        checks["svc_bounds"] = self.check_svc_bounds(svc_q)

        voltage = np.array(results.get("voltage", {}).get("values", []))
        checks["voltage_bounds"] = self.check_voltage_bounds(voltage if voltage.size > 0 else None)

        # 聚合
        total_viol = 0
        total_checks = 0
        for key in ["oltc_bounds", "sc_bounds", "pv_q_bounds", "wt_q_bounds", "svc_bounds", "voltage_bounds"]:
            c = checks[key]
            if "violation_count" in c:
                total_viol += c["violation_count"]
                total_checks += c["total_checks"]

        # OLTC/SC 动作超限也计入
        total_viol += checks["oltc_actions"]["exceeded"]
        total_checks += 1
        total_viol += checks["sc_actions"]["total_exceeded"]
        total_checks += self.n_sc

        pct = 100.0 * total_viol / total_checks if total_checks > 0 else 0.0

        return {
            "network": self.network_name,
            "scenario_id": self.scenario_id,
            "constraints": checks,
            "summary": {
                "total_violation_count": total_viol,
                "total_check_count": total_checks,
                "total_violation_percentage": pct,
            },
        }

    # ---------- 与GT对比 ----------
    def compare_with_ground_truth(self, rl_results: Dict) -> Dict:
        gt = self.load_ground_truth()
        gt_loss = float(gt["loss"]["total_kw"])
        rl_loss = float(rl_results["loss"]["total_kw"])
        rel_err = abs(rl_loss - gt_loss) / max(abs(gt_loss), 1e-6) * 100
        mae = abs(rl_loss - gt_loss)
        return {
            "gt_loss_kw": gt_loss,
            "rl_loss_kw": rl_loss,
            "relative_error_pct": rel_err,
            "mae_kw": mae,
        }


def print_vvc_violation_report(report: Dict):
    """打印 VVC 约束违反度报告"""
    print("\n" + "=" * 70)
    print("VVC 约束违反度检查报告")
    print("=" * 70)
    print(f"  网络: {report['network']}")
    print(f"  场景: {report['scenario_id']}")

    c = report["constraints"]

    print("\n[1] OLTC档位边界")
    print(f"    违反率: {c['oltc_bounds']['violation_pct']:.2f}%")
    print(f"[2] OLTC日动作次数")
    oa = c["oltc_actions"]
    print(f"    变化次数: {oa['n_changes']}, 上限: {oa['max_allowed']}, 超限: {oa['exceeded']}")

    print(f"[3] SC档位边界")
    print(f"    违反率: {c['sc_bounds']['violation_pct']:.2f}%")
    print(f"[4] SC日动作次数")
    for d in c["sc_actions"]["details"]:
        print(f"    SC{d['sc_id']}: 变化{d['n_changes']}次, 上限{d['max_allowed']}, 超限{d['exceeded']}")

    for tag, label in [("pv_q_bounds", "PV无功"), ("wt_q_bounds", "WT无功"), ("svc_bounds", "SVC无功")]:
        info = c[tag]
        if "status" in info:
            print(f"[{label}] 跳过")
        else:
            print(f"[{label}] 违反率: {info['violation_pct']:.2f}%")

    vb = c["voltage_bounds"]
    if "status" in vb:
        print(f"[电压] 跳过")
    else:
        print(f"[电压] 违反率: {vb['violation_pct']:.2f}%, "
              f"V∈[{vb['v_min']:.4f}, {vb['v_max']:.4f}]")

    s = report["summary"]
    print("\n" + "-" * 70)
    print(f"【总违反】{s['total_violation_count']} / {s['total_check_count']} = "
          f"{s['total_violation_percentage']:.2f}%")
    print("=" * 70)


def postprocess_actions(raw_actions: Dict, config: Dict) -> Dict:
    """后处理：离散取整、连续裁剪"""
    oltc_tap = np.round(raw_actions["oltc_tap"]).astype(np.int64)
    oltc_tap = np.clip(oltc_tap, config["devices"]["oltc"]["tap_min"],
                       config["devices"]["oltc"]["tap_max"])

    sc_stage = np.round(raw_actions["sc_stage"]).astype(np.int64)
    sc_stage = np.clip(sc_stage, 0, config["devices"]["sc"]["n_stages"])

    sc_q_mvar = sc_stage.astype(np.float64) * config["devices"]["sc"]["q_per_stage"]

    # PV Q 裁剪
    pv_q = raw_actions["pv_q_mvar"].copy()
    pv_cap = config["devices"]["pv"]["capacity"]
    for i in range(len(pv_cap)):
        pv_q[:, i] = np.clip(pv_q[:, i], -pv_cap[i], pv_cap[i])

    # WT Q 裁剪
    wt_q = raw_actions["wt_q_mvar"].copy()
    wt_cap = config["devices"]["wt"]["capacity"]
    for i in range(len(wt_cap)):
        wt_q[:, i] = np.clip(wt_q[:, i], -wt_cap[i], wt_cap[i])

    # SVC Q 裁剪
    svc_q = raw_actions["svc_q_mvar"].copy()
    svc_q = np.clip(svc_q, config["devices"]["svc"]["q_min"],
                    config["devices"]["svc"]["q_max"])

    return {
        "oltc_tap": oltc_tap,
        "sc_stage": sc_stage,
        "sc_q_mvar": sc_q_mvar,
        "pv_q_mvar": pv_q,
        "wt_q_mvar": wt_q,
        "svc_q_mvar": svc_q,
    }