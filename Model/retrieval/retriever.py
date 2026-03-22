# -*- coding: utf-8 -*-
"""
场景检索器
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from .feature_extractor import ScenarioFeatureExtractor
from .distance_metrics import combined_distance


class ScenarioRetriever:
    """场景检索器"""

    def __init__(self, network_name: str, database_dir: Path = None):
        """
        初始化场景检索器

        Args:
            network_name: 网络名称 (e.g., "ieee33", "ieee123")
            database_dir: 特征库存储目录，默认为scripts/database_feature目录
        """
        if database_dir is None:
            # 修正：默认使用scripts/database_feature目录
            database_dir = Path(__file__).parent.parent.parent / "scripts" / "database_feature"

        self.network_name = network_name
        self.database_dir = Path(database_dir)
        self.database_file = self.database_dir / f"database_features_{network_name}.pkl"

        # 加载database
        self.database = self._load_database()

        # 初始化特征提取器
        self.extractor = ScenarioFeatureExtractor()

        # 特征权重（可调）
        self._init_feature_weights()

    def _load_database(self) -> Dict:
        """加载预计算的database特征"""
        if not self.database_file.exists():
            raise FileNotFoundError(
                f"Database特征文件不存在: {self.database_file}\n"
                f"请先运行 scripts/build_database_features.py 构建特征库"
            )

        with open(self.database_file, 'rb') as f:
            database = pickle.load(f)

        print(f"已加载database: {len(database['scenario_ids'])}个场景")
        print(f"特征维度: {database['features'].shape[1]}")

        return database

    def _init_feature_weights(self):
        """初始化特征权重（可根据经验调整）"""
        feature_dim = self.database['features'].shape[1]

        # 简单策略：统计特征（均值、峰谷差）权重更高
        # 假设前64维是时序特征，后面是统计特征
        weights = np.ones(feature_dim)

        # 提高统计特征权重
        if feature_dim > 64:
            weights[64:] = 2.0  # 统计特征权重加倍

        self.feature_weights = weights

    def retrieve(self, scenario_file: Path, ev_file: Path = None,
                 top_k: int = 1) -> List[Tuple[str, float]]:
        """
        检索最相似场景

        Args:
            scenario_file: 新场景DER数据文件
            ev_file: 新场景EV数据文件（可选）
            top_k: 返回前k个相似场景

        Returns:
            [(scenario_id, distance), ...] 按相似度排序
        """
        # 提取新场景特征
        print(f"提取新场景特征: {scenario_file.name}")
        query_features = self.extractor.extract_from_scenario_file(
            scenario_file=scenario_file,
            ev_file=ev_file,
            network_name=self.network_name
        )

        # 计算距离
        distances = combined_distance(
            query=query_features,
            database=self.database['features'],
            weights=self.feature_weights,
            euclidean_weight=0.7,
            cosine_weight=0.3
        )

        # 排序
        sorted_indices = np.argsort(distances)[:top_k]

        results = []
        for idx in sorted_indices:
            scenario_id = self.database['scenario_ids'][idx]
            dist = distances[idx]
            results.append((scenario_id, float(dist)))

        return results

    def retrieve_top1(self, scenario_file: Path, ev_file: Path = None) -> str:
        """
        检索最相似场景（返回场景ID）

        Args:
            scenario_file: 新场景文件
            ev_file: EV文件（可选）

        Returns:
            最相似的场景ID（如 "001"）
        """
        results = self.retrieve(scenario_file, ev_file, top_k=1)

        best_scenario_id, best_distance = results[0]

        print(f"\n检索结果:")
        print(f"  最相似场景: {best_scenario_id}")
        print(f"  距离: {best_distance:.4f}")

        return best_scenario_id

    def retrieve_with_info(self, scenario_file: Path, ev_file: Path = None):
        """检索最相似场景并返回额外信息（distance与query_features）。

        Returns:
            best_scenario_id: str
            best_distance: float
            query_features: np.ndarray
        """
        print(f"提取新场景特征: {scenario_file.name}")
        query_features = self.extractor.extract_from_scenario_file(
            scenario_file=scenario_file,
            ev_file=ev_file,
            network_name=self.network_name
        )

        distances = combined_distance(
            query=query_features,
            database=self.database['features'],
            weights=self.feature_weights,
            euclidean_weight=0.7,
            cosine_weight=0.3
        )

        best_idx = int(np.argmin(distances))
        best_scenario_id = self.database['scenario_ids'][best_idx]
        best_distance = float(distances[best_idx])

        print(f"\n检索结果:")
        print(f"  最相似场景: {best_scenario_id}")
        print(f"  距离: {best_distance:.4f}")

        return best_scenario_id, best_distance, query_features


def test_retriever():
    """测试检索器"""
    from pathlib import Path

    # 测试参数
    network_name = "ieee33"

    # 输入文件
    scenario_file = Path("data/online_inf/scenario_online_33.csv")
    ev_file = Path("data/online_inf/ev_profiles_online_33.csv")

    if not scenario_file.exists():
        print(f"测试文件不存在: {scenario_file}")
        return

    try:
        # 修正：去除task参数
        retriever = ScenarioRetriever(network_name)

        # 检索
        best_scenario_id = retriever.retrieve_top1(scenario_file, ev_file)

        print(f"\n✅ 检索成功! 最匹配场景: {best_scenario_id}")

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("请先运行离线特征构建脚本")


if __name__ == "__main__":
    test_retriever()