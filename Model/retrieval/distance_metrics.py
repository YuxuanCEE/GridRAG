# -*- coding: utf-8 -*-
"""
距离度量函数
"""

import numpy as np


def weighted_euclidean_distance(query: np.ndarray,
                                database: np.ndarray,
                                weights: np.ndarray = None) -> np.ndarray:
    """
    加权欧氏距离

    Args:
        query: 查询特征 (feature_dim,)
        database: 数据库特征 (n_samples, feature_dim)
        weights: 特征权重 (feature_dim,) [可选]

    Returns:
        距离数组 (n_samples,)
    """
    if weights is None:
        weights = np.ones(query.shape[0])

    # 归一化权重
    weights = weights / weights.sum()

    # 广播计算
    diff = database - query  # (n_samples, feature_dim)
    weighted_diff = diff * np.sqrt(weights)  # 加权
    distances = np.linalg.norm(weighted_diff, axis=1)

    return distances


def cosine_similarity(query: np.ndarray,
                      database: np.ndarray) -> np.ndarray:
    """
    余弦相似度

    Args:
        query: 查询特征 (feature_dim,)
        database: 数据库特征 (n_samples, feature_dim)

    Returns:
        相似度数组 (n_samples,) 范围[0, 1]
    """
    # 归一化
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)

    # 点积
    similarity = np.dot(database_norm, query_norm)

    # 转换到[0, 1]范围（cosine范围[-1, 1]）
    similarity = (similarity + 1) / 2

    return similarity


def combined_distance(query: np.ndarray,
                      database: np.ndarray,
                      weights: np.ndarray = None,
                      euclidean_weight: float = 0.7,
                      cosine_weight: float = 0.3) -> np.ndarray:
    """
    组合距离：加权欧氏距离 + 余弦相似度

    Args:
        query: 查询特征
        database: 数据库特征
        weights: 特征权重
        euclidean_weight: 欧氏距离权重
        cosine_weight: 余弦相似度权重

    Returns:
        组合距离（越小越相似）
    """
    # 欧氏距离（归一化）
    euc_dist = weighted_euclidean_distance(query, database, weights)
    euc_dist_norm = euc_dist / (euc_dist.max() + 1e-8)

    # 余弦相似度（转为距离）
    cos_sim = cosine_similarity(query, database)
    cos_dist = 1 - cos_sim

    # 组合
    combined = euclidean_weight * euc_dist_norm + cosine_weight * cos_dist

    return combined