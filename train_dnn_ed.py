#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Baseline 主入口 - Task B (ED)

功能:
- 训练DNN模型（每个场景单独训练）
- 执行推理预测（支持跨场景）
- 计算约束违反度
- 输出关键指标：训练时间、推理时间、成本误差

使用方法:
    # 完整流程（同场景训练和测试）
    python train_dnn_ed.py --network ieee33 --scenario-id 004

    # 指定GPU训练
    python train_dnn_ed.py --network ieee33 --scenario-id 004 --gpu 0

    # 仅训练
    python train_dnn_ed.py --network ieee33 --scenario-id 004 --mode train

    # 跨场景测试：用004模型在005上推理
    python train_dnn_ed.py --network ieee33 --scenario-id 005 --mode predict --model-scenario 004

    # 评估005推理结果
    python train_dnn_ed.py --network ieee33 --scenario-id 005 --mode evaluate

    # 完整跨场景实验
    python train_dnn_ed.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dnn.data_loader_ed import EDDataProcessor, create_data_loaders
from models.dnn.dnn_model import create_model, EDTransformerModel
from models.dnn.trainer import EDTrainer
from models.dnn.predictor import EDPredictor
from utils.constraint_violation import ConstraintViolationChecker, print_violation_report


def get_device(gpu_id: int = None) -> torch.device:
    """
    获取计算设备

    Args:
        gpu_id: GPU编号，None表示自动选择，-1表示强制CPU

    Returns:
        torch.device
    """
    if gpu_id == -1:
        print("使用设备: CPU (手动指定)")
        return torch.device("cpu")

    if torch.cuda.is_available():
        if gpu_id is None:
            gpu_id = 0

        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
            print(f"使用设备: GPU {gpu_id} - {gpu_name} ({gpu_memory:.1f} GB)")
            return device
        else:
            print(f"警告: GPU {gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
            print(f"使用设备: GPU 0")
            return torch.device("cuda:0")
    else:
        print("使用设备: CPU (CUDA不可用)")
        return torch.device("cpu")


def train_model(network_name: str,
                scenario_id: str,
                n_epochs: int = 100,
                batch_size: int = 16,
                n_augments: int = 50,
                learning_rate: float = 1e-3,
                d_model: int = 128,
                n_heads: int = 4,
                n_layers: int = 3,
                gpu_id: int = None,
                verbose: bool = True) -> dict:
    """
    训练DNN模型

    Args:
        network_name: 网络名称 (ieee13, ieee33, ieee69, ieee123)
        scenario_id: 场景编号
        n_epochs: 训练轮数
        batch_size: 批次大小
        n_augments: 数据增强数量
        learning_rate: 学习率
        d_model: Transformer隐藏维度
        n_heads: 注意力头数
        n_layers: Encoder层数
        gpu_id: GPU编号
        verbose: 是否打印详细信息

    Returns:
        训练结果字典
    """
    print("\n" + "=" * 70)
    print(f"DNN Baseline 训练 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    # 获取设备
    device = get_device(gpu_id)

    # 1. 准备数据
    print("\n[1/4] 准备数据...")
    processor = EDDataProcessor(network_name, scenario_id, PROJECT_ROOT)

    train_loader, val_loader, meta = create_data_loaders(
        processor,
        batch_size=batch_size,
        n_augments=n_augments,
        val_ratio=0.2,
    )

    print(f"  训练样本: {meta['n_train']}")
    print(f"  验证样本: {meta['n_val']}")
    print(f"  输入特征: {meta['n_features']}")
    print(f"  连续输出: {meta['n_continuous']}")
    print(f"  二进制输出: {meta['n_binary']}")

    # 2. 创建模型
    print("\n[2/4] 创建模型...")
    model = create_model(
        meta,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {n_params:,}")
    print(f"  Transformer配置: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # 3. 训练
    print("\n[3/4] 开始训练...")

    trainer = EDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
    )

    train_start = time.time()
    train_results = trainer.train(
        n_epochs=n_epochs,
        early_stopping_patience=20,
        verbose=verbose,
    )
    train_time = time.time() - train_start

    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    output_dir = PROJECT_ROOT / "opt_results" / "dnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{network_name}_ed_scenario_{scenario_id}_model.pt"

    # 保存额外元信息（包括scaler参数用于推理）
    save_meta = {
        **meta,
        "network_name": network_name,
        "scenario_id": scenario_id,
        "n_augments": n_augments,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "input_scaler_mean": processor.input_scaler.mean_.tolist() if processor.input_scaler else None,
        "input_scaler_scale": processor.input_scaler.scale_.tolist() if processor.input_scaler else None,
        "output_scaler_mean": processor.output_scaler.mean_.tolist() if processor.output_scaler else None,
        "output_scaler_scale": processor.output_scaler.scale_.tolist() if processor.output_scaler else None,
    }

    trainer.save_model(model_path, meta=save_meta)

    return {
        "train_time": train_time,
        "best_val_loss": train_results["best_val_loss"],
        "best_epoch": train_results["best_epoch"],
        "model_path": str(model_path),
        "n_params": n_params,
        "device": str(device),
    }


def predict(network_name: str,
            scenario_id: str,
            model_scenario_id: str = None,
            gpu_id: int = None,
            verbose: bool = True) -> dict:
    """
    执行预测

    Args:
        network_name: 网络名称
        scenario_id: 测试场景编号（用于加载输入数据）
        model_scenario_id: 模型训练场景编号（如果为None则与scenario_id相同）
        gpu_id: GPU编号
        verbose: 是否打印详细信息

    Returns:
        预测结果和推理时间
    """
    if model_scenario_id is None:
        model_scenario_id = scenario_id

    is_cross_scenario = model_scenario_id != scenario_id

    print("\n" + "=" * 70)
    if is_cross_scenario:
        print(f"DNN Baseline 跨场景预测 - {network_name.upper()}")
        print(f"  模型来源: 场景 {model_scenario_id}")
        print(f"  测试场景: 场景 {scenario_id}")
    else:
        print(f"DNN Baseline 预测 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    # 获取设备
    device = get_device(gpu_id)

    # 指定模型路径
    model_path = PROJECT_ROOT / "opt_results" / "dnn" / f"{network_name}_ed_scenario_{model_scenario_id}_model.pt"

    predictor = EDPredictor(
        network_name,
        scenario_id,
        model_path=model_path,
        model_scenario_id=model_scenario_id,
        project_root=PROJECT_ROOT,
        device=device,
    )

    inference_start = time.time()
    results = predictor.run(save=True)
    inference_time = time.time() - inference_start

    return {
        "inference_time": inference_time,
        "results": results,
        "model_scenario_id": model_scenario_id,
        "test_scenario_id": scenario_id,
        "is_cross_scenario": is_cross_scenario,
    }


def evaluate(network_name: str,
             scenario_id: str,
             model_scenario_id: str = None,
             include_power_flow: bool = False,
             verbose: bool = True) -> dict:
    """
    评估DNN预测结果
    """
    if model_scenario_id is None:
        model_scenario_id = scenario_id

    is_cross_scenario = model_scenario_id != scenario_id

    print("\n" + "=" * 70)
    if is_cross_scenario:
        print(f"DNN Baseline 跨场景评估 - {network_name.upper()}")
        print(f"  模型来源: 场景 {model_scenario_id}")
        print(f"  测试场景: 场景 {scenario_id}")
    else:
        print(f"DNN Baseline 评估 - {network_name.upper()} - 场景 {scenario_id}")
    print("=" * 70)

    checker = ConstraintViolationChecker(network_name, scenario_id, PROJECT_ROOT)

    # 确定要加载的DNN结果文件
    if is_cross_scenario:
        dnn_results_path = PROJECT_ROOT / "opt_results" / "dnn" / \
                           f"{network_name}_ed_scenario_{scenario_id}_from_{model_scenario_id}_dnn_results.json"
    else:
        dnn_results_path = PROJECT_ROOT / "opt_results" / "dnn" / \
                           f"{network_name}_ed_scenario_{scenario_id}_dnn_results.json"

    # 约束违反度检查
    print("\n[1/2] 检查约束违反度...")
    dnn_results = checker.load_dnn_results(dnn_results_path)
    violations = checker.check_all(results=dnn_results, include_power_flow=include_power_flow)

    if verbose:
        print_violation_report(violations)

    # 与优化器结果对比
    print("\n[2/2] 与优化器结果对比...")
    try:
        comparison = checker.compare_with_ground_truth(dnn_results=dnn_results)

        print("\n成本对比:")
        print(f"  DNN预测成本: {comparison['cost_comparison']['dnn_cost_yuan']:.2f} 元")
        print(f"  优化器成本 (场景{scenario_id}): {comparison['cost_comparison']['gt_cost_yuan']:.2f} 元")
        print(f"  相对误差: {comparison['cost_comparison']['relative_error_percent']:.2f}%")

        print("\nESS调度误差:")
        print(f"  充电功率RMSE: {comparison['ess_comparison']['charge_rmse_mw']:.4f} MW")
        print(f"  放电功率RMSE: {comparison['ess_comparison']['discharge_rmse_mw']:.4f} MW")

        print("\n购电功率误差:")
        print(f"  RMSE: {comparison['grid_comparison']['pgrid_rmse_mw']:.4f} MW")

    except FileNotFoundError as e:
        print(f"  警告: 无法加载优化器结果进行对比 - {e}")
        comparison = None

    return {
        "violations": violations,
        "comparison": comparison,
        "is_cross_scenario": is_cross_scenario,
    }


def run_cross_scenario_test(network_name: str,
                            train_scenario_id: str,
                            test_scenario_id: str,
                            n_epochs: int = 100,
                            n_augments: int = 50,
                            include_power_flow: bool = False,
                            skip_train: bool = False,
                            gpu_id: int = None,
                            verbose: bool = True) -> dict:
    """
    运行跨场景测试：在一个场景训练，在另一个场景测试

    这用于验证"新场景旧模型失效"的假设

    Args:
        network_name: 网络名称
        train_scenario_id: 训练场景编号
        test_scenario_id: 测试场景编号
        n_epochs: 训练轮数
        n_augments: 数据增强数量
        include_power_flow: 是否运行潮流检查
        skip_train: 是否跳过训练（使用已有模型）
        gpu_id: GPU编号
        verbose: 是否打印详细信息

    Returns:
        完整结果
    """
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# DNN Baseline 跨场景测试")
    print(f"# 网络: {network_name.upper()}")
    print(f"# 训练场景: {train_scenario_id}")
    print(f"# 测试场景: {test_scenario_id}")
    print("#" * 70)

    # 1. 训练（在训练场景上）
    if not skip_train:
        train_results = train_model(
            network_name, train_scenario_id,
            n_epochs=n_epochs,
            n_augments=n_augments,
            gpu_id=gpu_id,
            verbose=verbose,
        )
    else:
        print(f"\n跳过训练，使用已有模型: 场景 {train_scenario_id}")
        model_path = PROJECT_ROOT / "opt_results" / "dnn" / f"{network_name}_ed_scenario_{train_scenario_id}_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        train_results = {"train_time": 0, "model_path": str(model_path)}

    # 2. 预测（在测试场景上，使用训练场景的模型）
    predict_results = predict(
        network_name,
        test_scenario_id,
        model_scenario_id=train_scenario_id,
        gpu_id=gpu_id,
        verbose=verbose
    )

    # 3. 评估（与测试场景的优化器结果对比）
    eval_results = evaluate(
        network_name,
        test_scenario_id,
        model_scenario_id=train_scenario_id,
        include_power_flow=include_power_flow,
        verbose=verbose,
    )

    total_time = time.time() - total_start

    # 汇总输出
    print("\n" + "=" * 70)
    print("跨场景测试 关键指标汇总")
    print("=" * 70)

    print(f"\n【实验设置】")
    print(f"  训练场景: {train_scenario_id}")
    print(f"  测试场景: {test_scenario_id}")

    print(f"\n【时间消耗】")
    print(f"  训练时间: {train_results.get('train_time', 0):.2f} s")
    print(f"  推理时间: {predict_results['inference_time'] * 1000:.2f} ms")
    print(f"  总时间: {total_time:.2f} s")

    print(f"\n【约束违反度】")
    violations = eval_results["violations"]
    summary = violations.get("summary", {})
    print(f"  ESS SOC边界: {violations['constraints']['ess_soc_bounds']['total_violation_percentage']:.2f}%")
    print(f"  ESS功率边界: {violations['constraints']['ess_power_bounds']['total_violation_percentage']:.2f}%")
    print(f"  ESS充放电互斥: {violations['constraints']['ess_mutex']['violation_percentage']:.2f}%")
    print(f"  SOC动态最大偏差: {violations['constraints']['ess_soc_dynamics']['max_deviation_mwh']:.4f} MWh")
    print(f"  --------------------------------")
    print(f"  【总违反比例: {summary.get('total_violation_percentage', 0):.2f}%】")

    if eval_results["comparison"]:
        print(f"\n【成本误差】")
        comp = eval_results["comparison"]
        print(f"  DNN成本: {comp['cost_comparison']['dnn_cost_yuan']:.2f} 元")
        print(f"  优化器成本 (场景{test_scenario_id}): {comp['cost_comparison']['gt_cost_yuan']:.2f} 元")
        print(f"  相对误差: {comp['cost_comparison']['relative_error_percent']:.2f}%")

    print("\n" + "=" * 70)

    # 保存结果
    output_dir = PROJECT_ROOT / "opt_results" / "dnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{network_name}_ed_cross_test_{train_scenario_id}_to_{test_scenario_id}_summary.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    summary_data = {
        "experiment_type": "cross_scenario_test",
        "network": network_name,
        "train_scenario_id": train_scenario_id,
        "test_scenario_id": test_scenario_id,
        "timestamp": datetime.now().isoformat(),
        "time_consumption": {
            "train_time_s": train_results.get("train_time", 0),
            "inference_time_ms": predict_results["inference_time"] * 1000,
            "total_time_s": total_time,
        },
        "constraint_violations": convert_numpy(violations),
        "cost_comparison": convert_numpy(eval_results["comparison"]) if eval_results["comparison"] else None,
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n汇总结果已保存至: {summary_path}")

    return summary_data


def run_full_pipeline(network_name: str,
                      scenario_id: str,
                      n_epochs: int = 100,
                      n_augments: int = 50,
                      include_power_flow: bool = False,
                      gpu_id: int = None,
                      verbose: bool = True) -> dict:
    """
    运行完整流程：训练 → 预测 → 评估（同场景）
    """
    total_start = time.time()

    print("\n" + "#" * 70)
    print(f"# DNN Baseline 完整流程（同场景）")
    print(f"# 网络: {network_name.upper()}")
    print(f"# 场景: {scenario_id}")
    print("#" * 70)

    # 1. 训练
    train_results = train_model(
        network_name, scenario_id,
        n_epochs=n_epochs,
        n_augments=n_augments,
        gpu_id=gpu_id,
        verbose=verbose,
    )

    # 2. 预测
    predict_results = predict(
        network_name, scenario_id,
        gpu_id=gpu_id,
        verbose=verbose
    )

    # 3. 评估
    eval_results = evaluate(
        network_name, scenario_id,
        include_power_flow=include_power_flow,
        verbose=verbose,
    )

    total_time = time.time() - total_start

    # 汇总输出
    print("\n" + "=" * 70)
    print("关键指标汇总")
    print("=" * 70)

    print(f"\n【时间消耗】")
    print(f"  训练时间: {train_results['train_time']:.2f} s")
    print(f"  推理时间: {predict_results['inference_time'] * 1000:.2f} ms")
    print(f"  总时间: {total_time:.2f} s")

    print(f"\n【模型信息】")
    print(f"  参数量: {train_results['n_params']:,}")
    print(f"  最佳验证损失: {train_results['best_val_loss']:.4f}")
    print(f"  最佳epoch: {train_results['best_epoch']}")
    print(f"  训练设备: {train_results.get('device', 'N/A')}")

    print(f"\n【约束违反度】")
    violations = eval_results["violations"]
    summary = violations.get("summary", {})
    print(f"  ESS SOC边界: {violations['constraints']['ess_soc_bounds']['total_violation_percentage']:.2f}%")
    print(f"  ESS功率边界: {violations['constraints']['ess_power_bounds']['total_violation_percentage']:.2f}%")
    print(f"  ESS充放电互斥: {violations['constraints']['ess_mutex']['violation_percentage']:.2f}%")
    print(f"  SOC动态最大偏差: {violations['constraints']['ess_soc_dynamics']['max_deviation_mwh']:.4f} MWh")
    print(f"  --------------------------------")
    print(f"  【总违反比例: {summary.get('total_violation_percentage', 0):.2f}%】")

    if eval_results["comparison"]:
        print(f"\n【成本误差】")
        comp = eval_results["comparison"]
        print(f"  DNN成本: {comp['cost_comparison']['dnn_cost_yuan']:.2f} 元")
        print(f"  优化器成本: {comp['cost_comparison']['gt_cost_yuan']:.2f} 元")
        print(f"  相对误差: {comp['cost_comparison']['relative_error_percent']:.2f}%")

    print("\n" + "=" * 70)

    # 保存完整结果
    output_dir = PROJECT_ROOT / "opt_results" / "dnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{network_name}_ed_scenario_{scenario_id}_summary.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    summary_data = {
        "experiment_type": "same_scenario_test",
        "network": network_name,
        "scenario_id": scenario_id,
        "timestamp": datetime.now().isoformat(),
        "time_consumption": {
            "train_time_s": train_results["train_time"],
            "inference_time_ms": predict_results["inference_time"] * 1000,
            "total_time_s": total_time,
        },
        "model_info": {
            "n_params": train_results["n_params"],
            "best_val_loss": train_results["best_val_loss"],
            "best_epoch": train_results["best_epoch"],
            "device": train_results.get("device", "N/A"),
        },
        "constraint_violations": convert_numpy(violations),
        "cost_comparison": convert_numpy(eval_results["comparison"]) if eval_results["comparison"] else None,
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n汇总结果已保存至: {summary_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description="DNN Baseline for ED Task (Task B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（同场景训练和测试）
  python train_dnn_ed.py --network ieee33 --scenario-id 004

  # 指定GPU 0训练
  python train_dnn_ed.py --network ieee33 --scenario-id 004 --gpu 0

  # 指定GPU 1训练
  python train_dnn_ed.py --network ieee33 --scenario-id 004 --gpu 1

  # 强制使用CPU
  python train_dnn_ed.py --network ieee33 --scenario-id 004 --gpu -1

  # 仅训练
  python train_dnn_ed.py --network ieee33 --scenario-id 004 --mode train

  # 跨场景测试：用004模型在005上推理并评估
  python train_dnn_ed.py --network ieee33 --train-scenario 004 --test-scenario 005 --mode cross-test

  # 分步骤跨场景测试：
  # Step 1: 训练
  python train_dnn_ed.py -n ieee33 -s 004 -m train
  # Step 2: 用004模型预测005
  python train_dnn_ed.py -n ieee33 -s 005 -m predict --model-scenario 004
  # Step 3: 评估005结果
  python train_dnn_ed.py -n ieee33 -s 005 -m evaluate --model-scenario 004
        """
    )

    parser.add_argument("--network", "-n", type=str, default="ieee123",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"],
                        help="网络系统")
    parser.add_argument("--scenario-id", "-s", type=str, default="004",
                        help="场景编号")
    parser.add_argument("--mode", "-m", type=str, default="cross-test",
                        choices=["full", "train", "predict", "evaluate", "cross-test"],
                        help="运行模式")

    # GPU参数
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU编号 (默认: 0, -1表示强制CPU)")

    # 跨场景测试参数
    parser.add_argument("--train-scenario", type=str, default="004",
                        help="训练场景编号 (用于cross-test模式)")
    parser.add_argument("--test-scenario", type=str, default="005",
                        help="测试场景编号 (用于cross-test模式)")
    parser.add_argument("--model-scenario", type=str, default="004",
                        help="模型来源场景编号 (用于predict/evaluate模式的跨场景测试)")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，使用已有模型 (用于cross-test模式)")

    # 训练参数
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--augments", "-a", type=int, default=100,
                        help="数据增强数量")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer隐藏维度")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Encoder层数")

    # 其他
    parser.add_argument("--include-power-flow", action="store_true",
                        help="是否运行潮流计算检查电压约束")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="减少输出信息")

    args = parser.parse_args()

    verbose = not args.quiet

    # 默认使用GPU 0（如果可用）
    gpu_id = args.gpu if args.gpu is not None else 0

    if args.mode == "full":
        run_full_pipeline(
            args.network,
            args.scenario_id,
            n_epochs=args.epochs,
            n_augments=args.augments,
            include_power_flow=args.include_power_flow,
            gpu_id=gpu_id,
            verbose=verbose,
        )

    elif args.mode == "train":
        train_model(
            args.network,
            args.scenario_id,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            n_augments=args.augments,
            learning_rate=args.learning_rate,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            gpu_id=gpu_id,
            verbose=verbose,
        )

    elif args.mode == "predict":
        predict(
            args.network,
            args.scenario_id,
            model_scenario_id=args.model_scenario,
            gpu_id=gpu_id,
            verbose=verbose
        )

    elif args.mode == "evaluate":
        evaluate(
            args.network,
            args.scenario_id,
            model_scenario_id=args.model_scenario,
            include_power_flow=args.include_power_flow,
            verbose=verbose,
        )

    elif args.mode == "cross-test":
        # 跨场景测试模式
        train_scenario = args.train_scenario or args.scenario_id
        test_scenario = args.test_scenario

        if test_scenario is None:
            print("错误: cross-test模式需要指定 --test-scenario")
            print("示例: python train_dnn_ed.py -n ieee33 --train-scenario 004 --test-scenario 005 -m cross-test")
            sys.exit(1)

        run_cross_scenario_test(
            args.network,
            train_scenario,
            test_scenario,
            n_epochs=args.epochs,
            n_augments=args.augments,
            include_power_flow=args.include_power_flow,
            skip_train=args.skip_train,
            gpu_id=gpu_id,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()