# -*- coding: utf-8 -*-
"""
VVC 环境模块

提供简化DistFlow潮流计算，用于评估RL方案的电压和网损。
BC训练阶段不需要环境交互，本模块仅用于推理后评估。
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config_networks import get_network_config, get_network_instance


class VVCEnvironment:
    """VVC评估环境"""

    def __init__(self, network_name: str, project_root: Path = None):
        self.network_name = network_name.lower()
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root

        self.config = get_network_config(network_name, project_root)
        self.network = get_network_instance(network_name, self.config)

        self.n_buses = self.config["network"]["n_buses"]
        self.n_periods = self.config["optimization"]["day_ahead"]["n_periods"]
        self.s_base = self.config["network"]["s_base"]
        self.v_min = self.config["network"]["v_min"]
        self.v_max = self.config["network"]["v_max"]

        self.pv_config = self.config["devices"]["pv"]
        self.wt_config = self.config["devices"]["wt"]
        self.sc_config = self.config["devices"]["sc"]
        self.svc_config = self.config["devices"]["svc"]
        self.oltc_config = self.config["devices"]["oltc"]

        # 构建 BFS 序列
        self.bfs_order = self._build_bfs_order()

    def _build_bfs_order(self):
        order = []
        queue = [self.network.root_bus]
        visited = {self.network.root_bus}
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.network.children[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return order

    def run_distflow(self, p_inject_pu: np.ndarray, q_inject_pu: np.ndarray,
                     v_root: float = 1.0, max_iter: int = 30, tol: float = 1e-6
                     ) -> Tuple[np.ndarray, float]:
        """
        简化 DistFlow 前后扫潮流

        Args:
            p_inject_pu: (n_buses,) 各节点净有功注入 (pu，正为注入)
            q_inject_pu: (n_buses,) 各节点净无功注入 (pu)
            v_root: 根节点电压 (pu)

        Returns:
            v: (n_buses,) 电压幅值 (pu)
            loss_pu: 总有功损耗 (pu)
        """
        net = self.network
        n_br = net.n_branches

        u = np.ones(self.n_buses)
        u[net.root_bus] = v_root ** 2

        P_br = np.zeros(n_br)
        Q_br = np.zeros(n_br)
        I_sq = np.zeros(n_br)

        for _ in range(max_iter):
            u_old = u.copy()

            # 后向扫：叶→根，计算支路功率
            for node in reversed(self.bfs_order):
                if node == net.root_bus:
                    continue
                parent = net.parent[node]
                br = net.branch_idx[(parent, node)]

                P_br[br] = -p_inject_pu[node]
                Q_br[br] = -q_inject_pu[node]

                for child in net.children[node]:
                    cbr = net.branch_idx[(node, child)]
                    P_br[br] += P_br[cbr] + I_sq[cbr] * net.r_pu[cbr]
                    Q_br[br] += Q_br[cbr] + I_sq[cbr] * net.x_pu[cbr]

            # 前向扫：根→叶，计算电压
            for node in self.bfs_order:
                if node == net.root_bus:
                    continue
                parent = net.parent[node]
                br = net.branch_idx[(parent, node)]
                r, x = net.r_pu[br], net.x_pu[br]
                I_sq[br] = (P_br[br] ** 2 + Q_br[br] ** 2) / max(u[parent], 0.01)
                u[node] = u[parent] - 2 * (r * P_br[br] + x * Q_br[br]) + (r ** 2 + x ** 2) * I_sq[br]
                u[node] = max(u[node], 0.5)

            if np.max(np.abs(u - u_old)) < tol:
                break

        total_loss = sum(I_sq[l] * net.r_pu[l] for l in range(n_br))
        v = np.sqrt(np.maximum(u, 0))
        return v, total_loss

    def evaluate_actions(self, scenario_data: Dict,
                         oltc_tap: np.ndarray,
                         sc_stage: np.ndarray,
                         pv_q_mvar: np.ndarray,
                         wt_q_mvar: np.ndarray,
                         svc_q_mvar: np.ndarray,
                         ) -> Dict:
        """
        评估完整动作序列

        Args:
            scenario_data: 包含 pv, wt 容量因子的字典（来自 VVCDataProcessor.load_scenario_data()）
                           可选包含 load_factor: (96,) 时变负荷因子
            oltc_tap:   (96,) 整数档位
            sc_stage:   (96, n_sc) 整数档位
            pv_q_mvar:  (96, n_pv) MVar
            wt_q_mvar:  (96, n_wt) MVar
            svc_q_mvar: (96, n_svc) MVar

        Returns:
            评估结果字典（voltage, loss 等）
        """
        net = self.network
        pv_buses = self.pv_config["buses"]
        wt_buses = self.wt_config["buses"]
        sc_buses = self.sc_config["buses"]
        svc_buses = self.svc_config["buses"]
        pv_cap = self.pv_config["capacity"]
        wt_cap = self.wt_config["capacity"]
        sc_q_per_stage = self.sc_config["q_per_stage"]

        # 时变负荷因子：与优化器保持一致，默认全1
        load_factor = scenario_data.get("load_factor", np.ones(self.n_periods))

        all_v = np.zeros((self.n_periods, self.n_buses))
        all_loss_pu = np.zeros(self.n_periods)

        for t in range(self.n_periods):
            # 构建节点注入
            p_inject = np.zeros(self.n_buses)
            q_inject = np.zeros(self.n_buses)

            # 时变负荷（与优化器 misocp_var_opt / socp_var_opt 一致）
            lf = load_factor[t] if t < len(load_factor) else 1.0
            p_inject -= net.p_load_pu * lf
            q_inject -= net.q_load_pu * lf

            # PV有功 + 无功
            for i, bus in enumerate(pv_buses):
                p_mw = scenario_data["pv"][t, i] * pv_cap[i]
                p_inject[bus] += p_mw / self.s_base
                q_inject[bus] += pv_q_mvar[t, i] / self.s_base

            # WT有功 + 无功
            for i, bus in enumerate(wt_buses):
                p_mw = scenario_data["wt"][t, i] * wt_cap[i]
                p_inject[bus] += p_mw / self.s_base
                q_inject[bus] += wt_q_mvar[t, i] / self.s_base

            # SC无功
            for i, bus in enumerate(sc_buses):
                q_sc = sc_stage[t, i] * sc_q_per_stage
                q_inject[bus] += q_sc / self.s_base

            # SVC无功
            for i, bus in enumerate(svc_buses):
                q_inject[bus] += svc_q_mvar[t, i] / self.s_base

            # 根节点电压
            tap = oltc_tap[t]
            v_root = self.oltc_config["v0"] + tap * self.oltc_config["tap_step"]

            v, loss_pu = self.run_distflow(p_inject, q_inject, v_root=v_root)
            all_v[t] = v
            all_loss_pu[t] = loss_pu

        loss_kw = all_loss_pu * self.s_base * 1000  # pu → kW

        return {
            "voltage": all_v,              # (96, n_buses)
            "loss_per_period_kw": loss_kw, # (96,)
            "loss_total_kw": float(np.sum(loss_kw)),
            "loss_average_kw": float(np.mean(loss_kw)),
            "v_min": float(np.min(all_v)),
            "v_max": float(np.max(all_v)),
            "v_mean": float(np.mean(all_v)),
        }