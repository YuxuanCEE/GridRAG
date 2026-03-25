# -*- coding: utf-8 -*-
"""
Task C (Joint) 评估环境 — 简化 DistFlow 潮流计算
接口对标 env_vvc.VVCEnvironment.evaluate_actions()
"""

import numpy as np
from collections import deque
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_networks import get_network_config, get_network_instance


class JointEnvironment:
    """Joint 评估环境"""

    def __init__(self, network_name: str, project_root):
        self.config = get_network_config(network_name, project_root)
        self.network = get_network_instance(network_name, self.config)
        self.s_base = self.config["network"]["s_base"]
        self.v_min = self.config["network"]["v_min"]
        self.v_max = self.config["network"]["v_max"]
        self.n_periods = self.config["optimization"]["joint"].get("n_periods", 96)
        self.delta_t = self.config["optimization"]["joint"].get("delta_t", 0.25)

        dev = self.config["devices"]
        self.pv_buses = dev["pv"]["buses"]
        self.wt_buses = dev["wt"]["buses"]
        self.sc_buses = dev["sc"]["buses"]
        self.svc_buses = dev.get("svc", {}).get("buses", [])
        self.ess_buses = dev["ess"]["buses"]
        self.n_pv = len(self.pv_buses)
        self.n_wt = len(self.wt_buses)
        self.n_sc = len(self.sc_buses)
        self.n_svc = len(self.svc_buses)
        self.n_ess = len(self.ess_buses)
        self.n_ev = self.network.n_ev_stations
        self.ev_buses = [s["bus"] for s in self.network.ev_stations] if self.n_ev > 0 else []

        self.pv_capacity = dev["pv"]["capacity"]
        self.wt_capacity = dev["wt"].get("capacity", [0.5]*self.n_wt)
        self.sc_q_per_stage_pu = dev["sc"]["q_per_stage"] / self.s_base

        oltc = dev["oltc"]
        self.tap_step = oltc["tap_step"]
        self.v0 = oltc.get("v0", 1.0)

        self.topo = self._bfs()

    def _bfs(self):
        order, vis = [], set()
        q = deque([self.network.root_bus]); vis.add(self.network.root_bus)
        while q:
            b = q.popleft(); order.append(b)
            for d in self.network.get_downstream_branches(b):
                c = int(self.network.to_bus[d])
                if c not in vis:
                    vis.add(c); q.append(c)
        return order

    # ================================================================
    def evaluate_actions(self, scenario_data, ev_data, actions) -> dict:
        """
        actions: dict with
            oltc_tap       (T,) int actual tap position
            sc_stage       (T, n_sc) int
            ess_charge_mw  (T, n_ess)
            ess_discharge_mw (T, n_ess)
            ev_cut_ratio   (T, n_ev) actual ratio ∈ [0, max_cut]
            pv_q_pu        (T, n_pv) optional
            wt_q_pu        (T, n_wt) optional
            svc_q_pu       (T, n_svc) optional
            pv_curtail_mw  (T, n_pv) optional
        """
        T = self.n_periods
        nw = self.network
        nb = nw.n_buses; nbr = nw.n_branches

        # 场景曲线
        pv_mw = np.zeros((T, self.n_pv))
        if scenario_data and "pv" in scenario_data:
            for i, bus in enumerate(self.pv_buses):
                cap = self.pv_capacity[i] if i < len(self.pv_capacity) else 1.0
                d = scenario_data["pv"]
                if isinstance(d, dict) and bus in d:
                    pv_mw[:, i] = np.array(d[bus])[:T]
                elif isinstance(d, np.ndarray) and d.ndim == 2 and i < d.shape[1]:
                    pv_mw[:, i] = d[:T, i]

        wt_mw = np.zeros((T, self.n_wt))
        if scenario_data and "wt" in scenario_data:
            for i, bus in enumerate(self.wt_buses):
                d = scenario_data["wt"]
                if isinstance(d, dict) and bus in d:
                    wt_mw[:, i] = np.array(d[bus])[:T]
                elif isinstance(d, np.ndarray) and d.ndim == 2 and i < d.shape[1]:
                    wt_mw[:, i] = d[:T, i]

        lf = np.ones(T)
        if scenario_data and "load_factor" in scenario_data:
            lf = np.array(scenario_data["load_factor"])[:T]

        ev_req_mw = np.zeros((T, self.n_ev))
        if ev_data:
            for k in range(self.n_ev):
                if k in ev_data and "load_kw" in ev_data[k]:
                    ev_req_mw[:, k] = np.array(ev_data[k]["load_kw"])[:T] / 1000.0

        # 动作
        oltc_tap = actions["oltc_tap"]
        sc_st = actions["sc_stage"]
        ess_ch = actions["ess_charge_mw"]
        ess_dis = actions["ess_discharge_mw"]
        ev_cut = actions["ev_cut_ratio"]
        pv_q = actions.get("pv_q_pu", np.zeros((T, self.n_pv)))
        wt_q = actions.get("wt_q_pu", np.zeros((T, self.n_wt)))
        svc_q = actions.get("svc_q_pu", np.zeros((T, self.n_svc)))
        pv_curt = actions.get("pv_curtail_mw", np.zeros((T, self.n_pv)))

        # 输出
        voltage = np.ones((T, nb))
        loss_kw = np.zeros(T)
        grid_mw = np.zeros(T)

        for t in range(T):
            v_root = self.v0 + oltc_tap[t] * self.tap_step
            u_root = v_root ** 2

            P_inj = np.zeros(nb)
            Q_inj = np.zeros(nb)
            for j in range(nb):
                P_inj[j] = -nw.p_load_pu[j] * lf[t]
                Q_inj[j] = -nw.q_load_pu[j] * lf[t]

            for i, bus in enumerate(self.pv_buses):
                P_inj[bus] += (pv_mw[t,i] - pv_curt[t,i]) / self.s_base
                Q_inj[bus] += pv_q[t,i]
            for i, bus in enumerate(self.wt_buses):
                P_inj[bus] += wt_mw[t,i] / self.s_base
                Q_inj[bus] += wt_q[t,i]
            for k, bus in enumerate(self.ess_buses):
                net = (ess_dis[t,k] - ess_ch[t,k]) / self.s_base
                P_inj[bus] += net
            for i, bus in enumerate(self.sc_buses):
                Q_inj[bus] += sc_st[t,i] * self.sc_q_per_stage_pu
            for i, bus in enumerate(self.svc_buses):
                Q_inj[bus] += svc_q[t,i]
            for k, bus in enumerate(self.ev_buses):
                mw = ev_req_mw[t,k] * (1 - ev_cut[t,k])
                P_inj[bus] -= mw / self.s_base

            # DistFlow
            Pb = np.zeros(nbr); Qb = np.zeros(nbr); u = np.zeros(nb)
            u[nw.root_bus] = u_root
            for j in reversed(self.topo):
                up = nw.get_upstream_branch(j)
                if up is None:
                    continue
                ds = nw.get_downstream_branches(j)
                Pb[up] = sum(Pb[d] for d in ds) - P_inj[j]
                Qb[up] = sum(Qb[d] for d in ds) - Q_inj[j]
            for j in self.topo:
                up = nw.get_upstream_branch(j)
                if up is None:
                    continue
                fb = int(nw.from_bus[up])
                u[j] = u[fb] - 2*(nw.r_pu[up]*Pb[up] + nw.x_pu[up]*Qb[up])

            voltage[t] = np.sqrt(np.maximum(u, 0.01))

            # loss
            total_loss = 0
            for l in range(nbr):
                uf = max(u[int(nw.from_bus[l])], 0.8)
                total_loss += nw.r_pu[l] * (Pb[l]**2 + Qb[l]**2) / uf
            loss_kw[t] = total_loss * self.s_base * 1000

            # grid
            root_ds = nw.get_downstream_branches(nw.root_bus)
            pg = sum(Pb[d] for d in root_ds) - P_inj[nw.root_bus]
            grid_mw[t] = max(pg * self.s_base, 0)

        return {
            "voltage": voltage,
            "v_min": float(voltage.min()),
            "v_max": float(voltage.max()),
            "v_mean": float(voltage.mean()),
            "loss_per_period_kw": loss_kw,
            "loss_total_kw": float(loss_kw.sum()),
            "loss_average_kw": float(loss_kw.mean()),
            "grid_power_mw": grid_mw,
        }