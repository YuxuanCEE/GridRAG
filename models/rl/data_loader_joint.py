# -*- coding: utf-8 -*-
"""
Task C (Joint) 数据处理器 — 接口与 data_loader_vvc.VVCDataProcessor 对齐

数据来源 (路径由 network + scenario_id 自动确定):
  - 优化结果 JSON:  opt_results/joint/{net}_joint_scenario_{sid}_results.json
  - 场景 CSV:       data/profiles/scenario_{sid}_{bus}.csv
  - EV CSV:         data/profiles/ev_profiles_{sid}_{bus}.csv

场景 CSV 列名格式:
  scenario_id, timestamp, node_{原始节点号}_PV, ..., node_{原始节点号}_wind, ...
  注意: 原始节点号 ≠ 0-indexed bus 编号, 必须用 config["columns"] 做映射

EV CSV 列名格式:
  datetime, station_0_load_kw, station_0_soc, station_1_load_kw, station_1_soc, ...
"""

import json
import csv
import re
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_networks import get_network_config, get_network_instance, get_result_filename

# ---- EV 列名正则 (station 编号本身就是 0-indexed, 与代码一致) ----
_RE_EV_LOAD = re.compile(r'^station_(\d+)_load_kw$', re.IGNORECASE)
_RE_EV_SOC = re.compile(r'^station_(\d+)_soc$', re.IGNORECASE)


class JointDataProcessor:
    """Joint 优化 BC 数据处理器 (对标 VVCDataProcessor)"""

    def __init__(self, network_name: str, scenario_id: str, project_root):
        self.network_name = network_name
        self.scenario_id = scenario_id
        self.project_root = Path(project_root)

        self.config = get_network_config(network_name, project_root)
        self.network = get_network_instance(network_name, self.config)

        # 网络拓扑编号 (ieee33 → 33)
        self.bus_num = str(self.network.n_buses)

        # ---- 设备计数 ----
        dev = self.config["devices"]
        self.pv_buses = dev["pv"]["buses"]
        self.wt_buses = dev["wt"]["buses"]
        self.sc_buses = dev["sc"]["buses"]
        self.svc_buses = dev.get("svc", {}).get("buses", [])
        self.ess_buses = dev["ess"]["buses"]

        # ★ CSV 列名映射 (直接从 config 读取, 与 VVC loader 保持一致)
        self.pv_columns = dev["pv"].get("columns", [])
        self.wt_columns = dev["wt"].get("columns", [])

        self.n_pv = len(self.pv_buses)
        self.n_wt = len(self.wt_buses)
        self.n_sc = len(self.sc_buses)
        self.n_svc = len(self.svc_buses)
        self.n_ess = len(self.ess_buses)
        self.n_ev = self.network.n_ev_stations

        # OLTC
        oltc = dev["oltc"]
        self.tap_min = oltc["tap_min"]
        self.tap_max = oltc["tap_max"]
        self.n_taps = self.tap_max - self.tap_min + 1

        # SC
        self.n_sc_stages = dev["sc"]["n_stages"]

        # ESS
        ess = dev["ess"]
        self.ess_capacity = ess["capacity_mwh"]
        self.ess_max_ch_rate = ess["max_charge_rate"]
        self.ess_max_dis_rate = ess["max_discharge_rate"]
        self.ess_soc_min = ess["soc_min"]
        self.ess_soc_max = ess["soc_max"]
        self.ess_soc_init = ess["soc_init"]
        self.eta_ch = ess["efficiency_charge"]
        self.eta_dis = ess["efficiency_discharge"]

        # PV / WT
        self.pv_capacity = dev["pv"]["capacity"]
        self.wt_capacity = dev["wt"].get("capacity", [0.5] * self.n_wt)

        # EV
        self.ev_stations = self.network.ev_stations if self.n_ev > 0 else []
        self.ev_capacity = [s["capacity_mwh"] for s in self.ev_stations]
        self.ev_max_power = [s["max_power_mw"] for s in self.ev_stations]
        self.max_cut_ratio = dev.get("ev_stations", {}).get("max_cut_ratio", 0.8)
        self.ev_eta = dev.get("ev_stations", {}).get("charge_efficiency", 0.95)

        # 其他
        self.n_periods = self.config["optimization"]["joint"].get("n_periods", 96)
        self.s_base = self.config["network"]["s_base"]
        self.delta_t = self.config["optimization"]["joint"].get("delta_t", 0.25)
        self.prices = self._build_prices()

        # ---- 特征 / 动作维度 ----
        self.n_features = (
            self.n_pv + self.n_wt + 1 + self.n_ev
            + self.n_ess + self.n_ev
            + 1 + 2 + 1 + self.n_sc
        )
        self.n_cont_actions = (
            self.n_svc + self.n_pv + self.n_wt
            + self.n_ess + self.n_pv + self.n_ev
        )

        self.input_scaler = StandardScaler()

    # ============================================================ 电价
    def _build_prices(self):
        pc = self.config["price"]
        p = np.zeros(self.n_periods)
        for t in range(self.n_periods):
            h = (t * 15 // 60) + 1
            if h in pc["peak_hours"]:
                p[t] = pc["peak_price"]
            elif h in pc["valley_hours"]:
                p[t] = pc["valley_price"]
            else:
                p[t] = pc["flat_price"]
        return p

    # ============================================================ 负荷曲线
    def _get_load_curve(self) -> np.ndarray:
        """
        生成典型日负荷变化曲线（标准化因子）
        
        ★ 必须与 data/data_loader.py → DataLoader.get_load_curve() 完全一致
        优化器通过该曲线缩放base负荷: p_load[t] = p_load_pu * load_factor[t]
        """
        hours = np.linspace(0, 24, self.n_periods, endpoint=False)
        
        # 典型居民/商业混合负荷曲线
        load_curve = (
            0.4  # 基础负荷
            + 0.15 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 22)  # 白天高峰
            + 0.25 * np.exp(-((hours - 12) ** 2) / 8)  # 中午高峰
            + 0.2 * np.exp(-((hours - 19) ** 2) / 4)   # 晚高峰
            - 0.1 * np.exp(-((hours - 4) ** 2) / 4)    # 凌晨低谷
        )
        
        # 归一化到0.3-1.0范围
        load_curve = 0.3 + 0.7 * (load_curve - load_curve.min()) / (load_curve.max() - load_curve.min())
        
        return load_curve.astype(np.float64)

    # ============================================================ 文件路径
    def _scenario_csv_path(self) -> Path:
        return (self.project_root / "data" / "profiles"
                / f"scenario_{self.scenario_id}_{self.bus_num}.csv")

    def _ev_csv_path(self) -> Path:
        return (self.project_root / "data" / "profiles"
                / f"ev_profiles_{self.scenario_id}_{self.bus_num}.csv")

    # ============================================================ 通用 CSV 读取
    def _read_csv(self, path: Path) -> dict:
        """
        读 CSV → {列名: [float列表]}
        使用 utf-8-sig 自动剥离 BOM
        """
        data = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return data
            for col in reader.fieldnames:
                data[col.strip()] = []
            for row in reader:
                for col in reader.fieldnames:
                    key = col.strip()
                    try:
                        data[key].append(float(row[col]))
                    except (ValueError, TypeError):
                        data[key].append(0.0)
        return data

    # ============================================================ 加载场景
    def load_scenario_data(self) -> dict:
        """
        加载场景 CSV → dict:
            pv:   {bus_id(0-indexed): np.array(T,)}
            wt:   {bus_id(0-indexed): np.array(T,)}
            load_factor: np.array(T,)

        ★ 关键: 用 config 中的 columns 字段 (如 "node_18_PV") 直接匹配 CSV 列名,
          再用 pv_buses[i] (0-indexed) 作为 dict key, 避免 1-indexed vs 0-indexed 错配
        """
        csv_path = self._scenario_csv_path()
        if not csv_path.exists():
            print(f"  警告: 场景文件不存在: {csv_path}，使用默认值")
            return self._default_scenario_data()

        data = self._read_csv(csv_path)
        result = {"pv": {}, "wt": {}, "load_factor": self._get_load_curve()}

        # ---- PV: 通过 config columns 精确匹配 ----
        for i, col_name in enumerate(self.pv_columns):
            if i >= len(self.pv_buses):
                break
            if col_name in data:
                bus = self.pv_buses[i]  # 0-indexed bus id
                result["pv"][bus] = np.array(data[col_name][:self.n_periods], dtype=float)
            else:
                print(f"    警告: PV 列 '{col_name}' 在CSV中未找到")

        # ---- WT: 通过 config columns 精确匹配 ----
        for i, col_name in enumerate(self.wt_columns):
            if i >= len(self.wt_buses):
                break
            if col_name in data:
                bus = self.wt_buses[i]  # 0-indexed bus id
                result["wt"][bus] = np.array(data[col_name][:self.n_periods], dtype=float)
            else:
                print(f"    警告: WT 列 '{col_name}' 在CSV中未找到")

        # 诊断输出
        found_pv = sorted(result["pv"].keys())
        found_wt = sorted(result["wt"].keys())
        missing_pv = [self.pv_columns[i] for i in range(self.n_pv)
                      if i < len(self.pv_columns) and self.pv_buses[i] not in result["pv"]]
        missing_wt = [self.wt_columns[i] for i in range(self.n_wt)
                      if i < len(self.wt_columns) and self.wt_buses[i] not in result["wt"]]

        print(f"  场景CSV: {csv_path.name}")
        print(f"    PV: 匹配成功 {len(found_pv)}/{self.n_pv} "
              f"(buses={found_pv}, columns={self.pv_columns[:len(found_pv)]})")
        if missing_pv:
            print(f"    PV 缺失列: {missing_pv}")
        print(f"    WT: 匹配成功 {len(found_wt)}/{self.n_wt} "
              f"(buses={found_wt}, columns={self.wt_columns[:len(found_wt)]})")
        if missing_wt:
            print(f"    WT 缺失列: {missing_wt}")

        return result

    # ============================================================ 加载 EV
    def load_ev_data(self) -> dict:
        """
        加载 EV CSV → dict[int, dict]:
            {k: {"load_kw": np.array(T,), "soc": np.array(T,)}}
        station 编号本身就是 0-indexed, 与 CSV 一致
        """
        csv_path = self._ev_csv_path()
        if not csv_path.exists():
            print(f"  警告: EV文件不存在: {csv_path}，使用默认值")
            return self._default_ev_data()

        raw = self._read_csv(csv_path)
        result = {}
        for k in range(self.n_ev):
            result[k] = {
                "load_kw": np.zeros(self.n_periods),
                "soc": np.zeros(self.n_periods),
            }

        for col_name, values in raw.items():
            col = col_name.strip()
            arr = np.array(values[:self.n_periods], dtype=float)

            m_load = _RE_EV_LOAD.match(col)
            if m_load:
                k = int(m_load.group(1))
                if k < self.n_ev:
                    result[k]["load_kw"] = arr
                continue

            m_soc = _RE_EV_SOC.match(col)
            if m_soc:
                k = int(m_soc.group(1))
                if k < self.n_ev:
                    result[k]["soc"] = arr
                continue

        loaded = [k for k in range(self.n_ev) if np.any(result[k]["load_kw"] > 0)]
        print(f"  EV CSV: {csv_path.name}")
        print(f"    充电站: {self.n_ev} 个, 有数据的: {loaded}")

        return result

    # ============================================================ 默认数据
    def _default_scenario_data(self):
        T = self.n_periods
        pv_d = {}
        for i, bus in enumerate(self.pv_buses):
            cap = self.pv_capacity[i] if i < len(self.pv_capacity) else 1.0
            t_arr = np.arange(T)
            pv_d[bus] = cap * np.clip(np.sin(np.pi * (t_arr - 20) / 48), 0, 1)
        wt_d = {}
        for i, bus in enumerate(self.wt_buses):
            cap = self.wt_capacity[i] if i < len(self.wt_capacity) else 0.5
            wt_d[bus] = np.full(T, cap * 0.3)
        return {"pv": pv_d, "wt": wt_d, "load_factor": self._get_load_curve()}

    def _default_ev_data(self):
        result = {}
        for k in range(self.n_ev):
            mp = self.ev_max_power[k] if k < len(self.ev_max_power) else 1.0
            result[k] = {
                "load_kw": np.full(self.n_periods, mp * 1000 * 0.3),
                "soc": np.zeros(self.n_periods),
            }
        return result

    # ============================================================ 加载优化结果
    def load_optimization_results(self) -> dict:
        rd = self.project_root / "opt_results" / "joint"
        fn = get_result_filename(self.network_name, "joint", self.scenario_id, "results")
        fp = rd / fn
        if not fp.exists():
            raise FileNotFoundError(f"优化结果不存在: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", data)

    # ============================================================ 特征构建辅助
    def _pv_arr(self, scenario_data):
        """(T, n_pv) 归一化 (值 / 容量)"""
        T = self.n_periods
        out = np.zeros((T, self.n_pv))
        if scenario_data is None or "pv" not in scenario_data:
            return out
        pv_d = scenario_data["pv"]
        for i, bus in enumerate(self.pv_buses):
            cap = self.pv_capacity[i] if i < len(self.pv_capacity) else 1.0
            if isinstance(pv_d, dict) and bus in pv_d:
                out[:, i] = np.array(pv_d[bus])[:T] / max(cap, 1e-6)
            elif isinstance(pv_d, np.ndarray) and pv_d.ndim == 2 and i < pv_d.shape[1]:
                out[:, i] = pv_d[:T, i] / max(cap, 1e-6)
        return out

    def _wt_arr(self, scenario_data):
        """(T, n_wt) 归一化 (值 / 容量)"""
        T = self.n_periods
        out = np.zeros((T, self.n_wt))
        if scenario_data is None or "wt" not in scenario_data:
            return out
        wt_d = scenario_data["wt"]
        for i, bus in enumerate(self.wt_buses):
            cap = self.wt_capacity[i] if i < len(self.wt_capacity) else 1.0
            if isinstance(wt_d, dict) and bus in wt_d:
                out[:, i] = np.array(wt_d[bus])[:T] / max(cap, 1e-6)
            elif isinstance(wt_d, np.ndarray) and wt_d.ndim == 2 and i < wt_d.shape[1]:
                out[:, i] = wt_d[:T, i] / max(cap, 1e-6)
        return out

    def _load_factor_arr(self, scenario_data):
        T = self.n_periods
        if scenario_data and "load_factor" in scenario_data:
            return np.array(scenario_data["load_factor"])[:T]
        return self._get_load_curve()

    def _ev_demand_arr(self, ev_data):
        """(T, n_ev) 归一化: load_kw / 1000 / max_power_mw"""
        T = self.n_periods
        out = np.zeros((T, self.n_ev))
        if ev_data is None:
            return out
        for k in range(self.n_ev):
            mp = self.ev_max_power[k] if k < len(self.ev_max_power) else 1.0
            if k in ev_data and "load_kw" in ev_data[k]:
                out[:, k] = np.array(ev_data[k]["load_kw"])[:T] / 1000.0 / max(mp, 1e-6)
        return out

    # ============================================================ 构建特征
    def build_features(self, results, scenario_data=None, ev_data=None):
        """(n_periods, n_features)"""
        T = self.n_periods
        feat = np.zeros((T, self.n_features))
        idx = 0

        feat[:, idx:idx + self.n_pv] = self._pv_arr(scenario_data)
        idx += self.n_pv
        feat[:, idx:idx + self.n_wt] = self._wt_arr(scenario_data)
        idx += self.n_wt
        feat[:, idx] = self._load_factor_arr(scenario_data)
        idx += 1
        feat[:, idx:idx + self.n_ev] = self._ev_demand_arr(ev_data)
        idx += self.n_ev

        # ESS SOC
        if "ess" in results and results["ess"].get("soc_mwh"):
            soc = np.array(results["ess"]["soc_mwh"])[:T]
            if soc.ndim == 1:
                soc = soc.reshape(-1, 1)
            for k in range(min(self.n_ess, soc.shape[1] if soc.ndim > 1 else 1)):
                cap = self.ess_capacity[k] if k < len(self.ess_capacity) else 1.0
                feat[:, idx + k] = soc[:, k] / max(cap, 1e-6)
        idx += self.n_ess

        # EV energy
        if "ev" in results and results["ev"].get("energy_mwh"):
            ev_e = np.array(results["ev"]["energy_mwh"])[:T]
            if ev_e.ndim == 1:
                ev_e = ev_e.reshape(-1, 1)
            for k in range(min(self.n_ev, ev_e.shape[1] if ev_e.ndim > 1 else 1)):
                cap = self.ev_capacity[k] if k < len(self.ev_capacity) else 1.0
                feat[:, idx + k] = ev_e[:, k] / max(cap, 1e-6)
        idx += self.n_ev

        # price
        mp = max(self.prices) if max(self.prices) > 0 else 1.0
        feat[:, idx] = self.prices[:T] / mp
        idx += 1

        # time encoding
        ta = np.arange(T)
        feat[:, idx] = np.sin(2 * np.pi * ta / T)
        feat[:, idx + 1] = np.cos(2 * np.pi * ta / T)
        idx += 2

        # prev OLTC
        if "oltc" in results and results["oltc"].get("tap_position"):
            taps = np.array(results["oltc"]["tap_position"])[:T]
            pt = np.zeros(T)
            pt[0] = taps[0]
            pt[1:] = taps[:-1]
            feat[:, idx] = (pt - self.tap_min) / max(self.n_taps - 1, 1)
        idx += 1

        # prev SC
        idx += self.n_sc

        return feat

    # ============================================================ 动作标签
    def build_actions(self, results):
        """从优化结果提取动作标签"""
        T = self.n_periods

        if "oltc" in results and results["oltc"].get("tap_position"):
            taps = np.array(results["oltc"]["tap_position"])[:T]
            oltc = np.clip((taps - self.tap_min).astype(int), 0, self.n_taps - 1)
        else:
            oltc = np.full(T, self.n_taps // 2, dtype=int)

        sc = np.zeros((T, self.n_sc), dtype=int)

        cont = np.zeros((T, self.n_cont_actions))
        ci = self.n_svc + self.n_pv + self.n_wt

        if "ess" in results:
            ch = np.array(results["ess"]["charge_mw"])[:T]
            dis = np.array(results["ess"]["discharge_mw"])[:T]
            if ch.ndim == 1:
                ch = ch.reshape(-1, 1)
            if dis.ndim == 1:
                dis = dis.reshape(-1, 1)
            for k in range(min(self.n_ess, ch.shape[1])):
                cap = self.ess_capacity[k] if k < len(self.ess_capacity) else 1.0
                mch = cap * self.ess_max_ch_rate
                mdi = cap * self.ess_max_dis_rate
                for t in range(T):
                    if dis[t, k] > ch[t, k] + 1e-8:
                        cont[t, ci + k] = min(dis[t, k] / max(mdi, 1e-8), 1.0)
                    elif ch[t, k] > 1e-8:
                        cont[t, ci + k] = -min(ch[t, k] / max(mch, 1e-8), 1.0)
        ci += self.n_ess

        ci += self.n_pv

        if "ev" in results and results["ev"].get("cut_ratio"):
            cr = np.array(results["ev"]["cut_ratio"])[:T]
            if cr.ndim == 1:
                cr = cr.reshape(-1, 1)
            for k in range(min(self.n_ev, cr.shape[1])):
                cont[:, ci + k] = cr[:, k] / max(self.max_cut_ratio, 1e-6)
        ci += self.n_ev

        return oltc, sc, cont

    # ============================================================ 增强
    def augment(self, feat, oltc, sc, cont, n_augments):
        af, ao, asc, ac = [feat], [oltc], [sc], [cont]
        ng = self.n_pv + self.n_wt
        for _ in range(n_augments):
            f = feat.copy()
            f += np.random.normal(0, 0.01, f.shape)
            f[:, :ng] *= np.random.uniform(0.8, 1.2, (1, ng))
            f[:, ng] *= np.random.uniform(0.95, 1.05)
            f = np.clip(f, -5, 5)
            c = cont.copy() + np.random.normal(0, 0.01, cont.shape)
            c = np.clip(c, -1, 1)
            af.append(f)
            ao.append(oltc.copy())
            asc.append(sc.copy())
            ac.append(c)
        return (np.concatenate(af), np.concatenate(ao),
                np.concatenate(asc), np.concatenate(ac))

    # ============================================================ BC 数据集
    def prepare_bc_dataset(self, n_augments=50):
        results = self.load_optimization_results()
        sd = self.load_scenario_data()
        ed = self.load_ev_data()
        feat = self.build_features(results, sd, ed)
        oltc, sc, cont = self.build_actions(results)
        feat, oltc, sc, cont = self.augment(feat, oltc, sc, cont, n_augments)
        self.input_scaler.fit(feat)
        return self.input_scaler.transform(feat).astype(np.float32), oltc, sc, cont

    # ============================================================ 单步特征
    def build_single_step_features(self, scenario_data, ev_data,
                                   ess_soc, ev_energy,
                                   prev_oltc_norm, prev_sc_norm, t):
        """构建第 t 步的特征向量 (n_features,)"""
        f = np.zeros(self.n_features)
        idx = 0

        pv_all = self._pv_arr(scenario_data)
        if t < len(pv_all):
            f[idx:idx + self.n_pv] = pv_all[t]
        idx += self.n_pv

        wt_all = self._wt_arr(scenario_data)
        if t < len(wt_all):
            f[idx:idx + self.n_wt] = wt_all[t]
        idx += self.n_wt

        lf = self._load_factor_arr(scenario_data)
        f[idx] = lf[t] if t < len(lf) else 1.0
        idx += 1

        ev_dem = self._ev_demand_arr(ev_data)
        if t < len(ev_dem):
            f[idx:idx + self.n_ev] = ev_dem[t]
        idx += self.n_ev

        for k in range(self.n_ess):
            cap = self.ess_capacity[k] if k < len(self.ess_capacity) else 1.0
            f[idx + k] = ess_soc[k] / max(cap, 1e-6)
        idx += self.n_ess

        for k in range(self.n_ev):
            cap = self.ev_capacity[k] if k < len(self.ev_capacity) else 1.0
            f[idx + k] = ev_energy[k] / max(cap, 1e-6)
        idx += self.n_ev

        mp = max(self.prices) if max(self.prices) > 0 else 1.0
        f[idx] = self.prices[t] / mp if t < len(self.prices) else 0
        idx += 1

        f[idx] = np.sin(2 * np.pi * t / self.n_periods)
        f[idx + 1] = np.cos(2 * np.pi * t / self.n_periods)
        idx += 2

        f[idx] = prev_oltc_norm
        idx += 1

        if isinstance(prev_sc_norm, np.ndarray):
            f[idx:idx + self.n_sc] = prev_sc_norm[:self.n_sc]
        idx += self.n_sc

        return f

    # ============================================================ meta
    def get_meta(self):
        return {
            "n_features": self.n_features,
            "n_oltc_actions": self.n_taps,
            "n_sc": self.n_sc,
            "n_sc_stages": self.n_sc_stages,
            "n_pv": self.n_pv,
            "n_wt": self.n_wt,
            "n_svc": self.n_svc,
            "n_ess": self.n_ess,
            "n_ev": self.n_ev,
            "n_cont_actions": self.n_cont_actions,
        }


# ================================================================ DataLoader
def create_joint_data_loaders(processor, batch_size=32, n_augments=50, val_ratio=0.2):
    feat, oltc, sc, cont = processor.prepare_bc_dataset(n_augments)
    n = len(feat)
    nv = int(n * val_ratio)
    idx = np.random.permutation(n)
    ti, vi = idx[:n - nv], idx[n - nv:]

    def _mk(ii, shuf):
        ds = TensorDataset(
            torch.FloatTensor(feat[ii]),
            torch.LongTensor(oltc[ii]),
            torch.LongTensor(sc[ii]),
            torch.FloatTensor(cont[ii]),
        )
        return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuf)

    meta = processor.get_meta()
    meta["n_train"] = len(ti)
    meta["n_val"] = len(vi)
    return _mk(ti, True), _mk(vi, False), meta