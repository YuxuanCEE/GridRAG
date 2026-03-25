# -*- coding: utf-8 -*-
"""
离线构建database特征库

遍历所有场景文件，提取特征并保存

使用方法:
    python scripts/build_database_features.py --network ieee33
"""

import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Model.retrieval.feature_extractor import ScenarioFeatureExtractor


def build_database_features(network_name: str,
                            profiles_dir: Path = None,
                            output_dir: Path = None):
    """
    构建database特征库

    Args:
        network_name: 网络名称 (ieee13/33/69/123)
        profiles_dir: 场景文件目录
        output_dir: 输出目录
    """
    if profiles_dir is None:
        profiles_dir = PROJECT_ROOT / "data" / "profiles"

    if output_dir is None:
        output_dir = PROJECT_ROOT / "opt_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定节点数
    network_map = {"ieee13": 13, "ieee33": 33, "ieee69": 69, "ieee123": 123}
    n_buses = network_map.get(network_name, 33)

    # 搜索场景文件
    scenario_pattern = f"scenario_*_{n_buses}.csv"
    ev_pattern = f"ev_profiles_*_{n_buses}.csv"

    scenario_files = sorted(profiles_dir.glob(scenario_pattern))

    if len(scenario_files) == 0:
        print(f"⚠️  未找到场景文件: {profiles_dir / scenario_pattern}")
        return

    print(f"\n{'=' * 60}")
    print(f"构建Database特征库")
    print(f"{'=' * 60}")
    print(f"网络: {network_name}")
    print(f"场景数: {len(scenario_files)}")
    print(f"输出目录: {output_dir}")

    # 初始化特征提取器
    extractor = ScenarioFeatureExtractor()

    # 提取所有场景特征
    scenario_ids = []
    features_list = []

    print(f"\n提取场景特征...")

    for scenario_file in tqdm(scenario_files, desc="处理场景"):
        # 提取场景ID（如 "001"）
        filename = scenario_file.stem  # scenario_001_33
        parts = filename.split('_')
        if len(parts) >= 2:
            scenario_id = parts[1]  # "001"
        else:
            scenario_id = filename

        # 查找对应的EV文件
        ev_file = profiles_dir / f"ev_profiles_{scenario_id}_{n_buses}.csv"

        if not ev_file.exists():
            ev_file = None

        try:
            # 提取特征
            features = extractor.extract_from_scenario_file(
                scenario_file=scenario_file,
                ev_file=ev_file,
                network_name=network_name
            )

            scenario_ids.append(scenario_id)
            features_list.append(features)

        except Exception as e:
            print(f"\n⚠️  场景{scenario_id}处理失败: {e}")
            continue

    if len(features_list) == 0:
        print("❌ 没有成功提取任何特征")
        return

    # 转换为numpy数组
    features_array = np.vstack(features_list)

    # 保存database
    database = {
        "scenario_ids": scenario_ids,
        "features": features_array,
        "metadata": {
            "network": network_name,
            "n_scenarios": len(scenario_ids),
            "feature_dim": features_array.shape[1],
            "extractor_config": {
                "temporal_dim": extractor.temporal_dim,
            }
        }
    }

    # 修正：去除task参数
    output_file = output_dir / f"database_features_{network_name}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(database, f)

    print(f"\n{'=' * 60}")
    print(f"✅ 特征库构建完成!")
    print(f"{'=' * 60}")
    print(f"场景数: {len(scenario_ids)}")
    print(f"特征维度: {features_array.shape[1]}")
    print(f"保存位置: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="构建Database特征库")
    parser.add_argument("--network", "-n", type=str, default="ieee123",
                        choices=["ieee13", "ieee33", "ieee69", "ieee123"],
                        help="网络名称")
    parser.add_argument("--profiles-dir", type=str, default=None,
                        help="场景文件目录（默认: data/profiles）")
    parser.add_argument("--output-dir", type=str, default='database_feature',
                        help="输出目录（默认: opt_results）")

    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    build_database_features(
        network_name=args.network,
        profiles_dir=profiles_dir,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()