# -*- coding: utf-8 -*-
"""
DNN Baseline 模型 - Task B (ED)

模型结构: 输入层 → Transformer Encoder → 全连接层 → 输出层

输入: 时序特征 (PV, WT, 电价, 时间编码) + 静态特征
输出:
- 连续变量: ESS充放电功率, 购电功率
- 二进制变量: ESS模式, 联络线状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EDTransformerModel(nn.Module):
    """
    ED任务Transformer模型

    结构:
    1. 输入嵌入层: 将原始特征映射到d_model维度
    2. 位置编码
    3. Transformer Encoder
    4. 输出头:
       - 连续变量头: 全连接层
       - 二进制变量头: 全连接层 + Sigmoid
    """

    def __init__(self,
                 n_features: int,
                 n_continuous: int,
                 n_binary: int,
                 n_periods: int = 96,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            n_features: 输入特征维度
            n_continuous: 连续输出维度
            n_binary: 二进制输出维度
            n_periods: 时间步数
            d_model: Transformer隐藏维度
            n_heads: 注意力头数
            n_layers: Encoder层数
            d_ff: 前馈网络维度
            dropout: Dropout率
        """
        super().__init__()

        self.n_features = n_features
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.n_periods = n_periods
        self.d_model = d_model

        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_periods, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出头 - 连续变量
        self.continuous_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_continuous),
        )

        # 输出头 - 二进制变量
        self.binary_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_binary),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 (batch, n_periods, n_features)

        Returns:
            continuous: 连续输出 (batch, n_periods, n_continuous)
            binary_logits: 二进制输出logits (batch, n_periods, n_binary)
        """
        # 输入嵌入
        x = self.input_embedding(x)  # (batch, n_periods, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, n_periods, d_model)

        # 输出头
        continuous = self.continuous_head(x)  # (batch, n_periods, n_continuous)
        binary_logits = self.binary_head(x)  # (batch, n_periods, n_binary)

        return continuous, binary_logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测（推理模式）

        Args:
            x: 输入特征 (batch, n_periods, n_features)
            threshold: 二进制变量阈值

        Returns:
            continuous: 连续输出 (batch, n_periods, n_continuous)
            binary: 二进制输出 (batch, n_periods, n_binary)
        """
        self.eval()
        with torch.no_grad():
            continuous, binary_logits = self.forward(x)
            binary = (torch.sigmoid(binary_logits) > threshold).float()
        return continuous, binary


class EDLoss(nn.Module):
    """ED任务损失函数"""

    def __init__(self,
                 continuous_weight: float = 1.0,
                 binary_weight: float = 1.0):
        """
        Args:
            continuous_weight: 连续变量损失权重
            binary_weight: 二进制变量损失权重
        """
        super().__init__()
        self.continuous_weight = continuous_weight
        self.binary_weight = binary_weight

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                pred_continuous: torch.Tensor,
                pred_binary_logits: torch.Tensor,
                target_continuous: torch.Tensor,
                target_binary: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Returns:
            损失字典: {"total", "continuous", "binary"}
        """
        # 连续变量损失 (MSE)
        loss_continuous = self.mse_loss(pred_continuous, target_continuous)

        # 二进制变量损失 (BCE)
        loss_binary = self.bce_loss(pred_binary_logits, target_binary)

        # 总损失
        loss_total = self.continuous_weight * loss_continuous + self.binary_weight * loss_binary

        return {
            "total": loss_total,
            "continuous": loss_continuous,
            "binary": loss_binary,
        }


def create_model(meta: Dict,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1) -> EDTransformerModel:
    """
    根据元信息创建模型

    Args:
        meta: 数据元信息字典
        d_model: Transformer隐藏维度
        n_heads: 注意力头数
        n_layers: Encoder层数
        d_ff: 前馈网络维度
        dropout: Dropout率

    Returns:
        EDTransformerModel实例
    """
    model = EDTransformerModel(
        n_features=meta["n_features"],
        n_continuous=meta["n_continuous"],
        n_binary=meta["n_binary"],
        n_periods=meta["n_periods"],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )
    return model


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    n_periods = 96
    n_features = 20
    n_continuous = 7  # 3 ESS charge + 3 ESS discharge + 1 P_grid
    n_binary = 6  # 3 ESS mode + 3 tie status

    model = EDTransformerModel(
        n_features=n_features,
        n_continuous=n_continuous,
        n_binary=n_binary,
        n_periods=n_periods,
    )

    # 测试前向传播
    x = torch.randn(batch_size, n_periods, n_features)
    continuous, binary_logits = model(x)

    print(f"输入形状: {x.shape}")
    print(f"连续输出形状: {continuous.shape}")
    print(f"二进制输出形状: {binary_logits.shape}")

    # 测试损失
    criterion = EDLoss()
    target_continuous = torch.randn(batch_size, n_periods, n_continuous)
    target_binary = torch.randint(0, 2, (batch_size, n_periods, n_binary)).float()

    losses = criterion(continuous, binary_logits, target_continuous, target_binary)
    print(f"\n损失: total={losses['total']:.4f}, continuous={losses['continuous']:.4f}, binary={losses['binary']:.4f}")

    # 模型参数统计
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {n_params:,}")