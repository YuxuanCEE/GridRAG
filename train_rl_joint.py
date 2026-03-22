#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL Baseline 主入口 - Task C (Joint)

接口与 train_rl_vvc.py 完全对齐:
  - 同场景完整流程 / 单独训练 / 推理 / 评估 / 跨场景测试

数据路径由 --network + --scenario-id 自动确定:
  场景:  data/profiles/scenario_{sid}_{bus}.csv
  EV:    data/profiles/ev_profiles_{sid}_{bus}.csv

使用方法:
    # 完整流程（同场景）
    python train_rl_joint.py --network ieee33 --scenario-id 004

    # 跨场景测试
    python train_rl_joint.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test

    # 仅训练
    python train_rl_joint.py --network ieee33 --scenario-id 004 --mode train

    # 仅推理（用004模型在005上）
    python train_rl_joint.py --network ieee33 --scenario-id 005 --mode predict --model-scenario 004
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_networks import get_network_config, get_result_filename
from models.rl.data_loader_joint import JointDataProcessor, create_joint_data_loaders
from models.rl.bc_agent_joint import JointBCAgent
from models.rl.env_joint import JointEnvironment


def _convert_numpy(obj):
    """递归转换 numpy 对象为 JSON 可序列化类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    return obj


def get_device(gpu_id=None):
    if gpu_id == -1:
        print("使用设备: CPU (手动指定)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        gid = gpu_id if gpu_id is not None else 0
        if gid < torch.cuda.device_count():
            d = torch.device(f"cuda:{gid}")
            print(f"使用设备: GPU {gid} - {torch.cuda.get_device_name(gid)}")
            return d
        print(f"GPU {gid} 不存在，使用 GPU 0")
        return torch.device("cuda:0")
    print("使用设备: CPU")
    return torch.device("cpu")


# ========================================================================
#  训练
# ========================================================================
def train_model(network_name, scenario_id, n_epochs=100, n_augments=50,
                batch_size=32, learning_rate=1e-3, hidden_dim=256,
                gpu_id=None, verbose=True):

    print("\n" + "=" * 70)
    print(f"RL Baseline (BC) 训练 - Joint - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    device = get_device(gpu_id)

    # 1. 数据
    print("\n[1/4] 准备数据...")
    processor = JointDataProcessor(network_name, scenario_id, PROJECT_ROOT)
    train_loader, val_loader, meta = create_joint_data_loaders(
        processor, batch_size=batch_size, n_augments=n_augments, val_ratio=0.2)

    print(f"  训练样本: {meta['n_train']}, 验证样本: {meta['n_val']}")
    print(f"  输入特征: {meta['n_features']}")
    print(f"  OLTC类别: {meta['n_oltc_actions']}, SC: {meta['n_sc']}×{meta['n_sc_stages']+1}, "
          f"ESS: {meta['n_ess']}, EV: {meta['n_ev']}, 连续: {meta['n_cont_actions']}")

    # 2. 创建智能体
    print("\n[2/4] 创建BC智能体...")
    agent = JointBCAgent(meta, device=str(device), hidden_dim=hidden_dim,
                         learning_rate=learning_rate)
    n_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    # 3. 训练
    print("\n[3/4] 开始训练...")
    t0 = time.time()
    train_results = agent.train(train_loader, val_loader, n_epochs=n_epochs,
                                early_stopping_patience=20, verbose=verbose)
    train_time = time.time() - t0

    # 4. 保存
    print("\n[4/4] 保存模型...")
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{network_name}_joint_scenario_{scenario_id}_model.pt"

    save_meta = {
        **meta,
        "network_name": network_name,
        "scenario_id": scenario_id,
        "hidden_dim": hidden_dim,
        "input_scaler_mean": processor.input_scaler.mean_.tolist(),
        "input_scaler_scale": processor.input_scaler.scale_.tolist(),
    }
    agent.save_model(model_path, meta=save_meta)

    return {
        "train_time": train_time,
        "best_val_loss": train_results["best_val_loss"],
        "best_epoch": train_results["best_epoch"],
        "model_path": str(model_path),
        "n_params": n_params,
        "device": str(device),
    }


# ========================================================================
#  推理 (顺序预测)
# ========================================================================
def predict(network_name, scenario_id, model_scenario_id=None,
            gpu_id=None, verbose=True):

    if model_scenario_id is None:
        model_scenario_id = scenario_id
    is_cross = (model_scenario_id != scenario_id)

    print("\n" + "=" * 70)
    if is_cross:
        print(f"RL Joint 跨场景推理 - 模型:{model_scenario_id} → 测试:{scenario_id}")
    else:
        print(f"RL Joint 推理 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    device = get_device(gpu_id)

    # 加载模型
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    model_path = output_dir / f"{network_name}_joint_scenario_{model_scenario_id}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    meta = ckpt["meta"]

    # 恢复 Scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["input_scaler_mean"])
    scaler.scale_ = np.array(meta["input_scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # 创建智能体并加载权重
    agent = JointBCAgent(meta, device=str(device),
                         hidden_dim=meta.get("hidden_dim", 256))
    agent.actor.load_state_dict(ckpt["actor_state_dict"])

    # 加载测试场景数据（路径由 network + scenario_id 自动确定）
    processor = JointDataProcessor(network_name, scenario_id, PROJECT_ROOT)
    scenario_data = processor.load_scenario_data()
    ev_data = processor.load_ev_data()

    T = processor.n_periods

    # ---- 顺序预测 ----
    print("  顺序预测 (96步)...")
    inference_start = time.time()

    # 初始化状态
    ess_soc = np.array([processor.ess_soc_init * c for c in processor.ess_capacity])
    ev_energy = np.zeros(processor.n_ev)
    if processor.n_ev > 0:
        for k, s in enumerate(processor.ev_stations):
            ev_energy[k] = s.get("arrival_energy_mwh", 0)

    prev_oltc_norm = 0.5
    prev_sc_norm = np.zeros(processor.n_sc)

    # 输出数组
    all_oltc_tap = np.zeros(T, dtype=int)
    all_sc_stage = np.zeros((T, processor.n_sc), dtype=int)
    all_ess_ch = np.zeros((T, processor.n_ess))
    all_ess_dis = np.zeros((T, processor.n_ess))
    all_ev_cut = np.zeros((T, processor.n_ev))
    all_ev_power = np.zeros((T, processor.n_ev))
    all_ess_soc = np.zeros((T, processor.n_ess))
    all_ev_energy = np.zeros((T, processor.n_ev))

    # EV 原始需求 (MW)
    ev_demand_norm = processor._ev_demand_arr(ev_data)
    ev_max_pw = np.array([processor.ev_max_power[k] if k < len(processor.ev_max_power)
                          else 1.0 for k in range(processor.n_ev)])
    ev_req_mw = ev_demand_norm * ev_max_pw.reshape(1, -1)

    for t in range(T):
        # 构建单步特征
        feat = processor.build_single_step_features(
            scenario_data, ev_data, ess_soc, ev_energy,
            prev_oltc_norm, prev_sc_norm, t)
        feat_norm = scaler.transform(feat.reshape(1, -1)).astype(np.float32)

        # 模型预测
        raw = agent.predict(feat_norm)
        oltc_idx = int(np.clip(raw["oltc_idx"][0], 0, processor.n_taps - 1))
        sc_st = np.clip(raw["sc_stage"][0], 0, processor.n_sc_stages).astype(int)
        cont = raw["continuous"][0]

        # 解码连续动作
        ci = processor.n_svc + processor.n_pv + processor.n_wt
        ess_act = np.clip(cont[ci:ci + processor.n_ess], -1, 1)
        ci += processor.n_ess
        ci += processor.n_pv  # PV curtailment (跳过)
        ev_cut_frac = np.clip(cont[ci:ci + processor.n_ev], 0, 1)
        ev_cut_actual = ev_cut_frac * processor.max_cut_ratio

        # ESS charge/discharge
        ch_mw = np.zeros(processor.n_ess)
        dis_mw = np.zeros(processor.n_ess)
        for k in range(processor.n_ess):
            cap = processor.ess_capacity[k]
            if ess_act[k] >= 0:
                dis_mw[k] = ess_act[k] * cap * processor.ess_max_dis_rate
            else:
                ch_mw[k] = abs(ess_act[k]) * cap * processor.ess_max_ch_rate

        # 更新 ESS SOC
        for k in range(processor.n_ess):
            cap = processor.ess_capacity[k]
            ess_soc[k] += (processor.eta_ch * ch_mw[k]
                           - dis_mw[k] / processor.eta_dis) * processor.delta_t
            ess_soc[k] = np.clip(ess_soc[k],
                                 cap * processor.ess_soc_min,
                                 cap * processor.ess_soc_max)

        # 更新 EV 能量
        for k in range(processor.n_ev):
            mw = ev_req_mw[t, k] * (1 - ev_cut_actual[k])
            ev_energy[k] += processor.ev_eta * mw * processor.delta_t
            cap = processor.ev_capacity[k] if k < len(processor.ev_capacity) else 10.0
            ev_energy[k] = np.clip(ev_energy[k], 0, cap)
            all_ev_power[t, k] = mw

        # 记录
        tap = processor.tap_min + oltc_idx
        all_oltc_tap[t] = tap
        all_sc_stage[t] = sc_st
        all_ess_ch[t] = ch_mw
        all_ess_dis[t] = dis_mw
        all_ev_cut[t] = ev_cut_actual
        all_ess_soc[t] = ess_soc.copy()
        all_ev_energy[t] = ev_energy.copy()

        # 更新 prev
        prev_oltc_norm = oltc_idx / max(processor.n_taps - 1, 1)
        prev_sc_norm = sc_st / max(processor.n_sc_stages, 1)

    inference_time = time.time() - inference_start

    # ---- 潮流评估 ----
    print("  运行简化潮流评估...")
    env = JointEnvironment(network_name, PROJECT_ROOT)
    eval_results = env.evaluate_actions(scenario_data, ev_data, {
        "oltc_tap": all_oltc_tap,
        "sc_stage": all_sc_stage,
        "ess_charge_mw": all_ess_ch,
        "ess_discharge_mw": all_ess_dis,
        "ev_cut_ratio": all_ev_cut,
    })

    # ---- 构建结果 ----
    dev_cfg = processor.config["devices"]
    sc_cfg = dev_cfg["sc"]
    pv_cfg = dev_cfg["pv"]
    wt_cfg = dev_cfg["wt"]
    svc_cfg = dev_cfg.get("svc", {})
    ess_cfg = dev_cfg["ess"]

    oltc_changes = int(np.sum(np.abs(np.diff(all_oltc_tap)) > 0.5))
    sc_n_actions = []
    for k in range(processor.n_sc):
        sc_n_actions.append(int(np.sum(np.abs(np.diff(all_sc_stage[:, k])) > 0.5)))

    full_results = {
        "scenario_id": scenario_id,
        "network": network_name,
        "task": "joint_rl",
        "model_scenario": model_scenario_id,
        "oltc": {
            "tap_position": all_oltc_tap.tolist(),
            "n_actions": oltc_changes,
        },
        "sc": {
            "buses": sc_cfg["buses"],
            "stage": all_sc_stage.tolist(),
            "n_actions": sc_n_actions,
        },
        "ess": {
            "buses": ess_cfg["buses"],
            "charge_mw": all_ess_ch.tolist(),
            "discharge_mw": all_ess_dis.tolist(),
            "soc_mwh": all_ess_soc.tolist(),
        },
        "ev": {
            "power_mw": all_ev_power.tolist(),
            "energy_mwh": all_ev_energy.tolist(),
            "cut_ratio": all_ev_cut.tolist(),
        },
        "pv_reactive": {
            "buses": pv_cfg["buses"],
            "q_mvar": np.zeros((T, processor.n_pv)).tolist(),
        },
        "wt_reactive": {
            "buses": wt_cfg["buses"],
            "q_mvar": np.zeros((T, processor.n_wt)).tolist(),
        },
        "svc_reactive": {
            "buses": svc_cfg.get("buses", []),
            "q_mvar": np.zeros((T, processor.n_svc)).tolist(),
        },
        "grid": {
            "power_mw": eval_results["grid_power_mw"].tolist(),
        },
        "voltage": {
            "values": eval_results["voltage"].tolist(),
            "min": eval_results["v_min"],
            "max": eval_results["v_max"],
            "mean": eval_results["v_mean"],
        },
        "loss": {
            "per_period_kw": eval_results["loss_per_period_kw"].tolist(),
            "total_kw": eval_results["loss_total_kw"],
            "average_kw": eval_results["loss_average_kw"],
        },
        "statistics": {
            "inference_time_ms": inference_time * 1000,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # 保存
    if is_cross:
        out_name = f"{network_name}_joint_scenario_{scenario_id}_from_{model_scenario_id}_rl_results.json"
    else:
        out_name = f"{network_name}_joint_scenario_{scenario_id}_rl_results.json"
    out_path = output_dir / out_name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_convert_numpy(full_results), f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {out_path}")

    return {
        "inference_time": inference_time,
        "results": full_results,
        "scenario_data": scenario_data,
        "ev_data": ev_data,
    }


# ========================================================================
#  目标函数计算（与 socp_joint.py 的目标函数对齐）
# ========================================================================
def _compute_joint_objective(rl_res, config):
    """
    根据 RL 结果计算 Task C 综合优化目标值
    与 socp_joint.py._add_objective() 的计算方式保持一致

    包含: 网损成本 + 电压惩罚 + 弃光成本 + 购电成本 + ESS成本 + 开关成本 + EV惩罚
    """
    from config_networks import PRICE_CONFIG

    opt_cfg = config["optimization"]["joint"]
    price_cfg = config["price"]
    dev_cfg = config["devices"]
    s_base = config["network"]["s_base"]
    delta_t = opt_cfg.get("delta_t", 0.25)
    n_periods = opt_cfg.get("n_periods", 96)

    loss_cost = opt_cfg.get("loss_cost", 500)
    voltage_penalty_coef = opt_cfg.get("voltage_penalty_coef", 1000)
    curtailment_cost = opt_cfg.get("pv_curtailment_cost", 700)
    weights = opt_cfg.get("weights", {})

    # 构建分时电价和时段权重
    price_data = np.zeros(n_periods)
    period_type = np.zeros(n_periods, dtype=int)
    for t in range(n_periods):
        hour = (t * 15 // 60) + 1
        if hour in price_cfg["peak_hours"]:
            price_data[t] = price_cfg["peak_price"]
            period_type[t] = 2
        elif hour in price_cfg["valley_hours"]:
            price_data[t] = price_cfg["valley_price"]
            period_type[t] = 0
        else:
            price_data[t] = price_cfg["flat_price"]
            period_type[t] = 1

    def get_weight(t, name):
        w = weights.get(name, [0.25, 0.25, 0.25])
        pt = period_type[t]
        return {0: w[2], 1: w[1], 2: w[0]}[pt]  # 峰, 平, 谷

    total = 0.0

    # 1. 网损成本（无权重，与修复后的socp_joint一致）
    loss_kw = np.array(rl_res["loss"]["per_period_kw"])
    for t in range(n_periods):
        loss_mw = loss_kw[t] / 1000.0
        total += loss_mw * loss_cost * delta_t

    # 2. 电压惩罚（保留权重）
    v_min = config["network"]["v_min"]
    v_max = config["network"]["v_max"]
    if "voltage" in rl_res and "values" in rl_res["voltage"]:
        v = np.array(rl_res["voltage"]["values"])
        for t in range(min(n_periods, v.shape[0])):
            w = get_weight(t, "voltage")
            over = np.maximum(v[t] - v_max, 0).sum()
            under = np.maximum(v_min - v[t], 0).sum()
            total += w * voltage_penalty_coef * (over + under)

    # 3. 购电成本（无权重）
    if "grid" in rl_res:
        p_grid = np.array(rl_res["grid"]["power_mw"])
        for t in range(min(n_periods, len(p_grid))):
            total += max(p_grid[t], 0) * price_data[t] * delta_t

    # 4. ESS运行成本（无权重）
    ess_cost_per_mwh = dev_cfg["ess"].get("cost_per_mwh", 40)
    if "ess" in rl_res:
        ch = np.array(rl_res["ess"].get("charge_mw", []))
        dis = np.array(rl_res["ess"].get("discharge_mw", []))
        if ch.size > 0 and dis.size > 0:
            for t in range(min(n_periods, ch.shape[0])):
                ess_mw = ch[t].sum() + dis[t].sum()
                total += ess_mw * ess_cost_per_mwh * delta_t

    # 5. EV惩罚
    ev_cfg = dev_cfg.get("ev_stations", {})
    if "ev" in rl_res and ev_cfg.get("enabled", False):
        ev_penalty_interrupt = ev_cfg.get("penalty_interruption", 50)
        ev_penalty_tier1 = opt_cfg.get("ev_curtailment_penalty_tier1", 200)
        ev_penalty_tier2 = opt_cfg.get("ev_curtailment_penalty_tier2", 800)
        ev_penalty_tier3 = opt_cfg.get("ev_curtailment_penalty_tier3", 1500)

        # 中断惩罚
        if "interruptions" in rl_res["ev"]:
            ints = np.array(rl_res["ev"]["interruptions"])
            for t in range(min(n_periods, ints.shape[0])):
                w = get_weight(t, "ev_satisfaction")
                total += w * ints[t].sum() * ev_penalty_interrupt

        # 削减惩罚 (简化: 用 cut_ratio 按阶梯分段)
        if "cut_ratio" in rl_res["ev"] and "power_mw" in rl_res["ev"]:
            cr = np.array(rl_res["ev"]["cut_ratio"])
            ev_pw = np.array(rl_res["ev"]["power_mw"])
            for t in range(min(n_periods, cr.shape[0])):
                for k in range(cr.shape[1] if cr.ndim > 1 else 1):
                    ratio = cr[t, k] if cr.ndim > 1 else cr[t]
                    # 原始需求 = actual / (1 - cut_ratio)
                    actual = ev_pw[t, k] if ev_pw.ndim > 1 else ev_pw[t]
                    if ratio < 1.0 - 1e-6:
                        required = actual / (1.0 - ratio)
                    else:
                        required = actual
                    t1 = min(ratio, 0.30)
                    t2 = min(max(ratio - 0.30, 0), 0.30)
                    t3 = min(max(ratio - 0.60, 0), 0.20)
                    total += required * t1 * ev_penalty_tier1 * delta_t
                    total += required * t2 * ev_penalty_tier2 * delta_t
                    total += required * t3 * ev_penalty_tier3 * delta_t

    return total


# ========================================================================
#  评估
# ========================================================================
def evaluate(network_name, scenario_id, model_scenario_id=None,
             scenario_data=None, ev_data=None, verbose=True):

    if model_scenario_id is None:
        model_scenario_id = scenario_id
    is_cross = (model_scenario_id != scenario_id)

    print("\n" + "=" * 70)
    if is_cross:
        print(f"RL Joint 跨场景评估 - 模型:{model_scenario_id} → 测试:{scenario_id}")
    else:
        print(f"RL Joint 评估 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    if is_cross:
        rl_path = output_dir / f"{network_name}_joint_scenario_{scenario_id}_from_{model_scenario_id}_rl_results.json"
    else:
        rl_path = output_dir / f"{network_name}_joint_scenario_{scenario_id}_rl_results.json"

    if not rl_path.exists():
        raise FileNotFoundError(f"RL结果不存在: {rl_path}")
    with open(rl_path, "r", encoding="utf-8") as f:
        rl_res = json.load(f)

    config = get_network_config(network_name, PROJECT_ROOT)

    # 1) 约束检查
    print("\n[1/2] 检查约束违反度...")
    violations = _check_constraints(rl_res, config)
    if verbose:
        _print_violation_report(violations)

    # 2) 与优化器对比
    print("\n[2/2] 与优化器结果对比...")
    comparison = None
    try:
        gt_dir = PROJECT_ROOT / "opt_results" / "joint"
        gt_fn = get_result_filename(network_name, "joint", scenario_id, "results")
        gt_path = gt_dir / gt_fn
        if not gt_path.exists():
            raise FileNotFoundError(f"GT不存在: {gt_path}")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_res = gt_data.get("results", gt_data)

        gt_loss = gt_res["loss"]["average_kw"]
        rl_loss = rl_res["loss"]["average_kw"]
        gt_cost = gt_res.get("cost", {}).get("total", 0)

        # ★ 计算 RL 结果对应的综合优化目标值
        rl_cost = _compute_joint_objective(rl_res, config)

        comparison = {
            "gt_loss_kw": gt_loss,
            "rl_loss_kw": rl_loss,
            "loss_mae_kw": abs(rl_loss - gt_loss),
            "loss_relative_error_pct": (rl_loss - gt_loss) / max(abs(gt_loss), 1e-6) * 100,
            "gt_cost": gt_cost,
            "rl_cost": rl_cost,
            "cost_relative_error_pct": (rl_cost - gt_cost) / max(abs(gt_cost), 1e-6) * 100 if gt_cost > 0 else 0,
        }
        print(f"  优化器网损: {gt_loss:.2f} kW")
        print(f"  RL网损:     {rl_loss:.2f} kW")
        print(f"  网损相对误差: {comparison['loss_relative_error_pct']:.2f}%")
        if gt_cost > 0:
            print(f"  优化器总成本: {gt_cost:.2f}")
            print(f"  RL总成本:     {rl_cost:.2f}")
            print(f"  成本相对误差: {comparison['cost_relative_error_pct']:.2f}%")
    except FileNotFoundError as e:
        print(f"  警告: 无法加载GT进行对比 - {e}")

    return {"violations": violations, "comparison": comparison}


def _check_constraints(rl_res, config):
    """检查约束违反"""
    v_min_lim = config["network"]["v_min"]
    v_max_lim = config["network"]["v_max"]
    ess_cfg = config["devices"]["ess"]

    viol = {
        "voltage": 0, "voltage_total": 0,
        "ess_soc": 0, "ess_soc_total": 0,
        "details": [],
    }

    # 电压约束
    if "voltage" in rl_res and "values" in rl_res["voltage"]:
        v = np.array(rl_res["voltage"]["values"])
        vv = int(np.sum((v < v_min_lim) | (v > v_max_lim)))
        viol["voltage"] = vv
        viol["voltage_total"] = v.size

    # ESS SOC 约束
    ess_soc_total = 0
    if "ess" in rl_res and "soc_mwh" in rl_res["ess"]:
        soc = np.array(rl_res["ess"]["soc_mwh"])
        if soc.ndim == 1:
            soc = soc.reshape(-1, 1)
        cap_list = ess_cfg["capacity_mwh"]
        for k in range(soc.shape[1]):
            cap = cap_list[k] if k < len(cap_list) else 1.0
            lo = cap * ess_cfg["soc_min"]
            hi = cap * ess_cfg["soc_max"]
            vv = int(np.sum((soc[:, k] < lo - 1e-4) | (soc[:, k] > hi + 1e-4)))
            viol["ess_soc"] += vv
            ess_soc_total += soc.shape[0]
    viol["ess_soc_total"] = ess_soc_total

    total = viol["voltage"] + viol["ess_soc"]
    total_possible = viol["voltage_total"] + ess_soc_total
    viol["summary"] = {
        "total_violations": total,
        "total_checks": max(total_possible, 1),
        "total_violation_percentage": total / max(total_possible, 1) * 100,
    }
    return viol


def _print_violation_report(viol):
    s = viol["summary"]
    print(f"  电压越限: {viol['voltage']} / {viol['voltage_total']}")
    print(f"  ESS SOC越限: {viol['ess_soc']} / {viol['ess_soc_total']}")
    print(f"  总违反率: {s['total_violation_percentage']:.2f}%")


# ========================================================================
#  跨场景测试
# ========================================================================
def run_cross_scenario_test(network_name, train_scenario, test_scenario,
                            n_epochs=100, n_augments=50, skip_train=False,
                            gpu_id=None, verbose=True):
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# RL Joint 跨场景测试 - {network_name.upper()}")
    print(f"# 训练场景: {train_scenario} → 测试场景: {test_scenario}")
    print("#" * 70)

    # 1. 训练
    if not skip_train:
        train_res = train_model(network_name, train_scenario, n_epochs=n_epochs,
                                n_augments=n_augments, gpu_id=gpu_id, verbose=verbose)
    else:
        print("\n跳过训练，使用已有模型")
        train_res = {"train_time": 0}

    # 2. 推理
    pred_res = predict(network_name, test_scenario,
                       model_scenario_id=train_scenario, gpu_id=gpu_id, verbose=verbose)

    # 3. 评估
    eval_res = evaluate(network_name, test_scenario,
                        model_scenario_id=train_scenario,
                        scenario_data=pred_res.get("scenario_data"),
                        ev_data=pred_res.get("ev_data"),
                        verbose=verbose)

    total_time = time.time() - total_start

    # 汇总
    print("\n" + "=" * 70)
    print("跨场景测试 关键指标汇总")
    print("=" * 70)
    print(f"\n【时间消耗】")
    print(f"  训练时间: {train_res.get('train_time', 0):.2f} s")
    print(f"  推理时间: {pred_res['inference_time'] * 1000:.2f} ms")
    print(f"  总时间:   {total_time:.2f} s")

    s = eval_res["violations"]["summary"]
    print(f"\n【约束违反率】 {s['total_violation_percentage']:.2f}%")

    if eval_res["comparison"]:
        c = eval_res["comparison"]
        print(f"\n【目标误差】")
        print(f"  GT网损: {c['gt_loss_kw']:.2f} kW | RL网损: {c['rl_loss_kw']:.2f} kW")
        print(f"  网损相对误差: {c['loss_relative_error_pct']:.2f}%")
        if c.get("gt_cost", 0) > 0:
            print(f"  GT总成本: {c['gt_cost']:.2f} | RL总成本: {c['rl_cost']:.2f}")
            print(f"  成本相对误差: {c['cost_relative_error_pct']:.2f}%")

    # 保存汇总
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = output_dir / f"{network_name}_joint_cross_{train_scenario}_to_{test_scenario}_summary.json"
    sd = _convert_numpy({
        "experiment_type": "cross_scenario_test",
        "network": network_name,
        "task": "joint",
        "train_scenario": train_scenario,
        "test_scenario": test_scenario,
        "timestamp": datetime.now().isoformat(),
        "time_consumption": {
            "train_time_s": train_res.get("train_time", 0),
            "inference_time_ms": pred_res["inference_time"] * 1000,
            "total_time_s": total_time,
        },
        "constraint_violations": eval_res["violations"],
        "comparison": eval_res["comparison"],
    })
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(sd, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {sp}")
    return sd


# ========================================================================
#  同场景完整流程
# ========================================================================
def run_full_pipeline(network_name, scenario_id, n_epochs=100, n_augments=50,
                      gpu_id=None, verbose=True):
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# RL Joint 完整流程 - {network_name.upper()} - 场景 {scenario_id}")
    print("#" * 70)

    train_res = train_model(network_name, scenario_id, n_epochs=n_epochs,
                            n_augments=n_augments, gpu_id=gpu_id, verbose=verbose)

    pred_res = predict(network_name, scenario_id, gpu_id=gpu_id, verbose=verbose)

    eval_res = evaluate(network_name, scenario_id,
                        scenario_data=pred_res.get("scenario_data"),
                        ev_data=pred_res.get("ev_data"),
                        verbose=verbose)

    total_time = time.time() - total_start

    # 汇总
    print("\n" + "=" * 70)
    print("关键指标汇总")
    print("=" * 70)
    print(f"\n【时间消耗】")
    print(f"  训练: {train_res['train_time']:.2f} s | 推理: {pred_res['inference_time']*1000:.2f} ms")
    print(f"【模型】参数量: {train_res['n_params']:,} | 最佳epoch: {train_res['best_epoch']}")

    s = eval_res["violations"]["summary"]
    print(f"【约束违反率】 {s['total_violation_percentage']:.2f}%")

    if eval_res["comparison"]:
        c = eval_res["comparison"]
        print(f"【网损误差】 相对: {c['loss_relative_error_pct']:.2f}%")
        if c.get("gt_cost", 0) > 0:
            print(f"【总成本】 GT: {c['gt_cost']:.2f} | RL: {c['rl_cost']:.2f} | 误差: {c['cost_relative_error_pct']:.2f}%")

    # 保存汇总
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = output_dir / f"{network_name}_joint_scenario_{scenario_id}_rl_summary.json"
    sd = _convert_numpy({
        "experiment_type": "same_scenario_test",
        "network": network_name,
        "task": "joint",
        "scenario_id": scenario_id,
        "timestamp": datetime.now().isoformat(),
        "time_consumption": {
            "train_time_s": train_res["train_time"],
            "inference_time_ms": pred_res["inference_time"] * 1000,
            "total_time_s": total_time,
        },
        "model_info": {
            "n_params": train_res["n_params"],
            "best_val_loss": train_res["best_val_loss"],
            "best_epoch": train_res["best_epoch"],
        },
        "constraint_violations": eval_res["violations"],
        "comparison": eval_res["comparison"],
    })
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(sd, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {sp}")
    return sd


# ========================================================================
#  主入口
# ========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RL Baseline (BC) for Joint Task (Task C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_rl_joint.py --network ieee33 --scenario-id 004
  python train_rl_joint.py --network ieee33 --scenario-id 004 --mode train
  python train_rl_joint.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test
  python train_rl_joint.py --network ieee33 --scenario-id 005 --mode predict --model-scenario 004
        """)

    parser.add_argument("--network", "-n", type=str, default="ieee123",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"])
    parser.add_argument("--scenario-id", "-s", type=str, default="006")
    parser.add_argument("--mode", "-m", type=str, default="cross-test",
                        choices=["full", "train", "predict", "evaluate", "cross-test"])
    parser.add_argument("--gpu", "-g", type=int, default=0)

    # 跨场景
    parser.add_argument("--train-scenario", type=str, default="006")
    parser.add_argument("--test-scenario", type=str, default="005")
    parser.add_argument("--model-scenario", type=str, default=None)
    parser.add_argument("--skip-train", action="store_true")

    # 超参数
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--augments", "-a", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet
    gpu_id = args.gpu

    if args.mode == "full":
        run_full_pipeline(args.network, args.scenario_id,
                          n_epochs=args.epochs, n_augments=args.augments,
                          gpu_id=gpu_id, verbose=verbose)

    elif args.mode == "train":
        train_model(args.network, args.scenario_id,
                    n_epochs=args.epochs, n_augments=args.augments,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim, gpu_id=gpu_id, verbose=verbose)

    elif args.mode == "predict":
        model_sc = args.model_scenario or args.scenario_id
        predict(args.network, args.scenario_id,
                model_scenario_id=model_sc, gpu_id=gpu_id, verbose=verbose)

    elif args.mode == "evaluate":
        model_sc = args.model_scenario or args.scenario_id
        evaluate(args.network, args.scenario_id,
                 model_scenario_id=model_sc, verbose=verbose)

    elif args.mode == "cross-test":
        train_sc = args.train_scenario or args.scenario_id
        test_sc = args.test_scenario
        if test_sc is None:
            print("错误: cross-test 需要 --test-scenario")
            sys.exit(1)
        run_cross_scenario_test(args.network, train_sc, test_sc,
                                n_epochs=args.epochs, n_augments=args.augments,
                                skip_train=args.skip_train,
                                gpu_id=gpu_id, verbose=verbose)


if __name__ == "__main__":
    main()