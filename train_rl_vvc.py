#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL Baseline 主入口 - Task A (VVC)

功能:
- 行为克隆训练（单场景）
- 推理预测（支持跨场景）
- 约束违反度 & 目标误差评估

使用方法:
    # 完整流程（同场景）
    python train_rl_vvc.py --network ieee33 --scenario-id 004

    # 跨场景测试
    python train_rl_vvc.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test

    # 仅训练
    python train_rl_vvc.py --network ieee33 --scenario-id 004 --mode train

    # 仅推理（用004模型在005上）
    python train_rl_vvc.py --network ieee33 --scenario-id 005 --mode predict --model-scenario 004
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

from config_networks import get_network_config
from models.rl.data_loader_vvc import VVCDataProcessor, create_vvc_data_loaders
from models.rl.network import create_actor
from models.rl.bc_agent import BCAgent
from models.rl.env_vvc import VVCEnvironment
from models.rl.utils import (
    VVCConstraintChecker, print_vvc_violation_report, postprocess_actions
)


def _convert_numpy(obj):
    """递归转换 numpy 对象为 JSON 可序列化类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    return obj


def get_device(gpu_id: int = None) -> torch.device:
    if gpu_id == -1:
        print("使用设备: CPU (手动指定)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        gpu_id = gpu_id if gpu_id is not None else 0
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"使用设备: GPU {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
            return device
        print(f"GPU {gpu_id} 不存在，使用 GPU 0")
        return torch.device("cuda:0")
    print("使用设备: CPU")
    return torch.device("cpu")


# ========== 训练 ==========
def train_model(network_name: str, scenario_id: str,
                n_epochs: int = 100, n_augments: int = 50,
                batch_size: int = 16, learning_rate: float = 1e-3,
                d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                gpu_id: int = None, verbose: bool = True) -> dict:

    print("\n" + "=" * 70)
    print(f"RL Baseline (BC) 训练 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    device = get_device(gpu_id)

    # 1. 数据
    print("\n[1/4] 准备数据...")
    processor = VVCDataProcessor(network_name, scenario_id, PROJECT_ROOT)
    train_loader, val_loader, meta = create_vvc_data_loaders(
        processor, batch_size=batch_size, n_augments=n_augments, val_ratio=0.2)

    print(f"  训练样本: {meta['n_train']}, 验证样本: {meta['n_val']}")
    print(f"  输入特征: {meta['n_features']}")
    print(f"  OLTC类别数: {meta['n_oltc_actions']}, SC单元: {meta['n_sc']}, "
          f"PV: {meta['n_pv']}, WT: {meta['n_wt']}, SVC: {meta['n_svc']}")

    # 2. 创建智能体
    print("\n[2/4] 创建BC智能体...")
    agent = BCAgent(meta, device=device, d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers, learning_rate=learning_rate)
    n_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    # 3. 训练
    print("\n[3/4] 开始训练...")
    train_start = time.time()
    train_results = agent.train(train_loader, val_loader, n_epochs=n_epochs,
                                early_stopping_patience=20, verbose=verbose)
    train_time = time.time() - train_start

    # 4. 保存
    print("\n[4/4] 保存模型...")
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{network_name}_vvc_scenario_{scenario_id}_model.pt"

    save_meta = {
        **meta,
        "network_name": network_name,
        "scenario_id": scenario_id,
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
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


# ========== 推理 ==========
def predict(network_name: str, scenario_id: str,
            model_scenario_id: str = None,
            gpu_id: int = None, verbose: bool = True) -> dict:

    if model_scenario_id is None:
        model_scenario_id = scenario_id
    is_cross = model_scenario_id != scenario_id

    print("\n" + "=" * 70)
    if is_cross:
        print(f"RL Baseline 跨场景推理 - 模型:{model_scenario_id} → 测试:{scenario_id}")
    else:
        print(f"RL Baseline 推理 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    device = get_device(gpu_id)

    # 加载模型 checkpoint
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    model_path = output_dir / f"{network_name}_vvc_scenario_{model_scenario_id}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    meta = ckpt["meta"]

    # 加载测试场景数据
    processor = VVCDataProcessor(network_name, scenario_id, PROJECT_ROOT)
    scenario_data = processor.load_scenario_data()
    input_tensor = processor.prepare_input_tensor(scenario_data)  # (96, F)

    # 使用训练时的 scaler 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["input_scaler_mean"])
    scaler.scale_ = np.array(meta["input_scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    input_norm = scaler.transform(input_tensor).astype(np.float32)

    # 创建智能体并加载权重
    agent = BCAgent(meta, device=device,
                    d_model=meta.get("d_model", 128),
                    n_heads=meta.get("n_heads", 4),
                    n_layers=meta.get("n_layers", 3))
    agent.actor.load_state_dict(ckpt["actor_state_dict"])

    # 推理
    inference_start = time.time()
    raw_actions = agent.predict(input_norm)
    inference_time = time.time() - inference_start

    # 后处理
    config = get_network_config(network_name, PROJECT_ROOT)
    actions = postprocess_actions(raw_actions, config)

    # 环境评估（潮流计算）
    print("  运行简化潮流评估...")
    env = VVCEnvironment(network_name, PROJECT_ROOT)
    eval_results = env.evaluate_actions(
        scenario_data, actions["oltc_tap"], actions["sc_stage"],
        actions["pv_q_mvar"], actions["wt_q_mvar"], actions["svc_q_mvar"])

    # 构建并保存结果
    pv_cfg = config["devices"]["pv"]
    wt_cfg = config["devices"]["wt"]
    sc_cfg = config["devices"]["sc"]
    svc_cfg = config["devices"]["svc"]

    # 统计动作次数
    oltc_n_actions = int(np.sum(np.abs(np.diff(actions["oltc_tap"])) > 0.5))
    sc_n_actions = []
    for k in range(len(sc_cfg["buses"])):
        sc_n_actions.append(int(np.sum(np.abs(np.diff(actions["sc_stage"][:, k])) > 0.5)))

    full_results = {
        "scenario_id": scenario_id,
        "network": network_name,
        "task": "vvc_rl",
        "model_scenario": model_scenario_id,
        "oltc": {
            "tap": actions["oltc_tap"].tolist(),
            "n_actions": oltc_n_actions,
        },
        "sc": {
            "buses": sc_cfg["buses"],
            "stage": actions["sc_stage"].tolist(),
            "q_mvar": actions["sc_q_mvar"].tolist(),
            "n_actions": sc_n_actions,
        },
        "pv_reactive": {
            "buses": pv_cfg["buses"],
            "q_mvar": actions["pv_q_mvar"].tolist(),
        },
        "wt_reactive": {
            "buses": wt_cfg["buses"],
            "q_mvar": actions["wt_q_mvar"].tolist(),
        },
        "svc_reactive": {
            "buses": svc_cfg["buses"],
            "q_mvar": actions["svc_q_mvar"].tolist(),
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
        out_name = f"{network_name}_vvc_scenario_{scenario_id}_from_{model_scenario_id}_rl_results.json"
    else:
        out_name = f"{network_name}_vvc_scenario_{scenario_id}_rl_results.json"
    out_path = output_dir / out_name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {out_path}")

    return {
        "inference_time": inference_time,
        "results": full_results,
        "scenario_data": scenario_data,
    }


# ========== 评估 ==========
def evaluate(network_name: str, scenario_id: str,
             model_scenario_id: str = None,
             scenario_data: dict = None,
             verbose: bool = True) -> dict:

    if model_scenario_id is None:
        model_scenario_id = scenario_id
    is_cross = model_scenario_id != scenario_id

    print("\n" + "=" * 70)
    if is_cross:
        print(f"RL Baseline 跨场景评估 - 模型:{model_scenario_id} → 测试:{scenario_id}")
    else:
        print(f"RL Baseline 评估 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    checker = VVCConstraintChecker(network_name, scenario_id, PROJECT_ROOT)

    # 确定RL结果路径
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    if is_cross:
        rl_path = output_dir / f"{network_name}_vvc_scenario_{scenario_id}_from_{model_scenario_id}_rl_results.json"
    else:
        rl_path = output_dir / f"{network_name}_vvc_scenario_{scenario_id}_rl_results.json"

    rl_results = checker.load_rl_results(rl_path)

    # 加载 PV/WT 容量因子（用于无功约束检查）
    if scenario_data is None:
        processor = VVCDataProcessor(network_name, scenario_id, PROJECT_ROOT)
        scenario_data = processor.load_scenario_data()
    pv_cf = scenario_data["pv"]
    wt_cf = scenario_data["wt"]

    # 1. 约束违反度
    print("\n[1/2] 检查约束违反度...")
    violations = checker.check_all(rl_results, pv_cf=pv_cf, wt_cf=wt_cf)
    if verbose:
        print_vvc_violation_report(violations)

    # 2. 与优化器结果对比
    print("\n[2/2] 与优化器结果对比...")
    comparison = None
    try:
        comparison = checker.compare_with_ground_truth(rl_results)
        print(f"  优化器网损: {comparison['gt_loss_kw']:.2f} kW")
        print(f"  RL网损:     {comparison['rl_loss_kw']:.2f} kW")
        print(f"  相对误差:   {comparison['relative_error_pct']:.2f}%")
        print(f"  MAE:        {comparison['mae_kw']:.2f} kW")
    except FileNotFoundError as e:
        print(f"  警告: 无法加载GT进行对比 - {e}")

    return {"violations": violations, "comparison": comparison}


# ========== 跨场景测试 ==========
def run_cross_scenario_test(network_name: str, train_scenario: str,
                            test_scenario: str, n_epochs: int = 100,
                            n_augments: int = 50, skip_train: bool = False,
                            gpu_id: int = None, verbose: bool = True) -> dict:
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# RL Baseline 跨场景测试 - {network_name.upper()}")
    print(f"# 训练场景: {train_scenario} → 测试场景: {test_scenario}")
    print("#" * 70)

    # 1. 训练
    if not skip_train:
        train_res = train_model(network_name, train_scenario, n_epochs=n_epochs,
                                n_augments=n_augments, gpu_id=gpu_id, verbose=verbose)
    else:
        print(f"\n跳过训练，使用已有模型")
        train_res = {"train_time": 0}

    # 2. 推理
    pred_res = predict(network_name, test_scenario,
                       model_scenario_id=train_scenario, gpu_id=gpu_id, verbose=verbose)

    # 3. 评估
    eval_res = evaluate(network_name, test_scenario,
                        model_scenario_id=train_scenario,
                        scenario_data=pred_res.get("scenario_data"),
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
        print(f"  相对误差: {c['relative_error_pct']:.2f}% | MAE: {c['mae_kw']:.2f} kW")

    # 保存汇总
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{network_name}_vvc_cross_{train_scenario}_to_{test_scenario}_summary.json"
    summary_data = _convert_numpy({
        "experiment_type": "cross_scenario_test",
        "network": network_name,
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
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")
    return summary_data


# ========== 同场景完整流程 ==========
def run_full_pipeline(network_name: str, scenario_id: str,
                      n_epochs: int = 100, n_augments: int = 50,
                      gpu_id: int = None, verbose: bool = True) -> dict:
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# RL Baseline 完整流程 - {network_name.upper()} - 场景 {scenario_id}")
    print("#" * 70)

    train_res = train_model(network_name, scenario_id, n_epochs=n_epochs,
                            n_augments=n_augments, gpu_id=gpu_id, verbose=verbose)

    pred_res = predict(network_name, scenario_id, gpu_id=gpu_id, verbose=verbose)

    eval_res = evaluate(network_name, scenario_id,
                        scenario_data=pred_res.get("scenario_data"), verbose=verbose)

    total_time = time.time() - total_start

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
        print(f"【目标误差】 相对: {c['relative_error_pct']:.2f}% | MAE: {c['mae_kw']:.2f} kW")

    # 保存
    output_dir = PROJECT_ROOT / "opt_results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{network_name}_vvc_scenario_{scenario_id}_rl_summary.json"
    summary_data = _convert_numpy({
        "experiment_type": "same_scenario_test",
        "network": network_name,
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
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")
    return summary_data


# ========== 主入口 ==========
def main():
    parser = argparse.ArgumentParser(
        description="RL Baseline (BC) for VVC Task (Task A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_rl_vvc.py --network ieee33 --scenario-id 004
  python train_rl_vvc.py --network ieee33 --scenario-id 004 --mode train
  python train_rl_vvc.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test
  python train_rl_vvc.py --network ieee33 --scenario-id 005 --mode predict --model-scenario 004
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
    parser.add_argument("--augments", "-a", type=int, default=1000)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
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
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, gpu_id=gpu_id, verbose=verbose)

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