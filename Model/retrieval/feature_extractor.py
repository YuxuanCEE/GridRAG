# -*- coding: utf-8 -*-
"""
场景特征提取器
方案A：1D-CNN + 统计特征（轻量级，适合在线推理）

特征维度：
- 时序特征（1D-CNN）: 64维
- 统计特征: ~40维
- 元信息: ~5维
总计: ~110维
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


class StatisticalFeatureExtractor:
    """统计特征提取器"""

    def extract(self, timeseries: np.ndarray) -> np.ndarray:
        """
        提取时序的统计特征

        Args:
            timeseries: (n_periods,) 或 (n_periods, n_nodes)

        Returns:
            统计特征向量
        """
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        features = []

        for i in range(timeseries.shape[1]):
            ts = timeseries[:, i]

            # 基础统计量
            features.extend([
                np.mean(ts),  # 均值
                np.std(ts),  # 标准差
                np.min(ts),  # 最小值
                np.max(ts),  # 最大值
                np.ptp(ts),  # 峰谷差
            ])

            # 分位数
            features.extend(np.percentile(ts, [25, 50, 75]))

            # 高阶统计量
            features.extend([
                stats.skew(ts),  # 偏度
                stats.kurtosis(ts),  # 峰度
            ])

            # 时序特性
            if len(ts) > 1:
                diff = np.diff(ts)
                features.extend([
                    np.mean(np.abs(diff)),  # 平均变化率
                    np.std(diff),  # 变化率标准差
                ])

                # 自相关（lag=1, 4, 24对应15min, 1h, 6h）
                for lag in [1, 4, 24]:
                    if len(ts) > lag:
                        acf = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                        features.append(acf if not np.isnan(acf) else 0.0)
                    else:
                        features.append(0.0)
            else:
                features.extend([0.0] * 5)

        return np.array(features, dtype=np.float32)


class TemporalFeatureExtractor:
    """1D-CNN时序特征提取器（轻量级）"""

    def __init__(self, output_dim: int = 64):
        """
        Args:
            output_dim: 输出特征维度
        """
        self.output_dim = output_dim

        # 简化版CNN：使用numpy手工实现卷积池化
        # 实际可替换为PyTorch模型，但为了轻量级，这里用统计池化近似

    def extract(self, timeseries: np.ndarray) -> np.ndarray:
        """
        提取时序特征（简化版：多尺度统计池化）

        Args:
            timeseries: (n_periods,) 或 (n_periods, n_nodes)

        Returns:
            时序特征向量 (output_dim,)
        """
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        n_periods, n_nodes = timeseries.shape
        features = []

        # 多尺度池化（模拟CNN的感受野）
        window_sizes = [4, 8, 16, 24]  # 对应1h, 2h, 4h, 6h

        for win_size in window_sizes:
            for node_idx in range(n_nodes):
                ts = timeseries[:, node_idx]

                # 滑动窗口统计
                n_windows = max(1, n_periods // win_size)
                for i in range(n_windows):
                    start = i * win_size
                    end = min((i + 1) * win_size, n_periods)
                    window = ts[start:end]

                    if len(window) > 0:
                        features.extend([
                            np.mean(window),
                            np.std(window),
                            np.max(window),
                        ])

        # 填充或截断到目标维度
        features = np.array(features, dtype=np.float32)

        if len(features) < self.output_dim:
            features = np.pad(features, (0, self.output_dim - len(features)))
        elif len(features) > self.output_dim:
            features = features[:self.output_dim]

        return features


class ScenarioFeatureExtractor:
    """场景特征提取器（综合版）"""

    def __init__(self, temporal_dim: int = 64):
        """
        Args:
            temporal_dim: 时序特征维度
        """
        self.stat_extractor = StatisticalFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(output_dim=temporal_dim)
        self.temporal_dim = temporal_dim

    def extract_from_scenario_file(self, scenario_file: Path,
                                   ev_file: Path = None,
                                   network_name: str = "ieee33") -> np.ndarray:
        """
        从场景文件提取特征

        Args:
            scenario_file: DER场景文件路径
            ev_file: EV场景文件路径（可选）
            network_name: 网络名称（用于识别列格式）

        Returns:
            特征向量 (feature_dim,)
        """
        # 加载DER数据
        df_der = pd.read_csv(scenario_file)

        # 识别PV和WT列
        pv_cols = [col for col in df_der.columns if 'PV' in col]
        wt_cols = [col for col in df_der.columns if 'wind' in col]

        # 提取时序数据
        der_timeseries = []

        for col in pv_cols + wt_cols:
            if col in df_der.columns:
                der_timeseries.append(df_der[col].values)

        if len(der_timeseries) == 0:
            raise ValueError(f"未找到DER数据列：{scenario_file}")

        der_timeseries = np.column_stack(der_timeseries)  # (n_periods, n_ders)

        # 加载EV数据（如果有）
        ev_timeseries = []
        if ev_file and ev_file.exists():
            df_ev = pd.read_csv(ev_file)
            load_cols = [col for col in df_ev.columns if 'load_kw' in col]
            soc_cols = [col for col in df_ev.columns if 'soc' in col]

            for col in load_cols + soc_cols:
                if col in df_ev.columns:
                    ev_timeseries.append(df_ev[col].values)

            if len(ev_timeseries) > 0:
                ev_timeseries = np.column_stack(ev_timeseries)

        # 提取特征
        features = self.extract_from_timeseries(
            der_timeseries=der_timeseries,
            ev_timeseries=ev_timeseries if len(ev_timeseries) > 0 else None
        )

        return features

    def extract_from_timeseries(self, der_timeseries: np.ndarray,
                                ev_timeseries: np.ndarray = None) -> np.ndarray:
        """
        从时序数据提取特征

        Args:
            der_timeseries: DER功率时序 (n_periods, n_ders)
            ev_timeseries: EV充电时序 (n_periods, n_evs) [可选]

        Returns:
            特征向量 (feature_dim,)
        """
        features = []

        # 1. DER时序特征
        der_temporal_feat = self.temporal_extractor.extract(der_timeseries)
        features.append(der_temporal_feat)

        # 2. DER统计特征
        der_stat_feat = self.stat_extractor.extract(der_timeseries)
        features.append(der_stat_feat)

        # 3. EV特征（如果有）
        if ev_timeseries is not None:
            ev_stat_feat = self.stat_extractor.extract(ev_timeseries)
            features.append(ev_stat_feat)

        # 4. 元信息特征
        meta_features = [
            np.mean(der_timeseries),  # 总体DER水平
            np.max(der_timeseries),  # 峰值出力
            np.sum(der_timeseries > 0) / der_timeseries.size,  # 出力占比
        ]

        if ev_timeseries is not None:
            meta_features.extend([
                np.mean(ev_timeseries),  # 总体EV负荷
                np.max(ev_timeseries),  # 峰值负荷
            ])

        features.append(np.array(meta_features, dtype=np.float32))

        # 拼接所有特征
        feature_vector = np.concatenate(features)

        return feature_vector

    def get_feature_dim(self) -> int:
        """获取特征维度（需要先运行一次extract才能确定）"""
        # 近似估计：temporal(64) + stat_der(~20) + stat_ev(~20) + meta(5) ≈ 110
        return self.temporal_dim + 45  # 预估值


def test_feature_extractor():
    """测试特征提取器"""
    from pathlib import Path

    # 测试数据
    profiles_dir = Path("data/profiles")
    scenario_file = profiles_dir / "scenario_001_33.csv"
    ev_file = profiles_dir / "ev_profiles_001_33.csv"

    if not scenario_file.exists():
        print(f"测试文件不存在: {scenario_file}")
        return

    extractor = ScenarioFeatureExtractor()

    print("提取场景特征...")
    features = extractor.extract_from_scenario_file(
        scenario_file=scenario_file,
        ev_file=ev_file,
        network_name="ieee33"
    )

    print(f"特征维度: {len(features)}")
    print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"特征均值: {features.mean():.4f}")
    print(f"特征示例（前10维）: {features[:10]}")


if __name__ == "__main__":
    test_feature_extractor()