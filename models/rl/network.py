# -*- coding: utf-8 -*-
"""
Actor-Critic 网络结构

Actor: 输入场景特征序列 → 输出混合动作（离散 + 连续）
Critic: 输入场景特征序列 → 输出状态价值估计（BC中可选）

网络采用 Transformer Encoder 架构，处理96时步的完整序列。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VVCActorNetwork(nn.Module):
    """
    VVC Actor 网络

    输入: (batch, T, n_features) 场景特征序列
    输出: 各类控制量
        - OLTC tap logits:   (batch, T, n_oltc_actions)
        - SC stage logits:   (batch, T, n_sc, n_sc_stages+1)
        - PV Q (normalized): (batch, T, n_pv)
        - WT Q (normalized): (batch, T, n_wt)
        - SVC Q (normalized):(batch, T, n_svc)
    """

    def __init__(self, n_features: int, n_periods: int,
                 n_oltc_actions: int, n_sc: int, n_sc_stages: int,
                 n_pv: int, n_wt: int, n_svc: int,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        self.n_features = n_features
        self.n_periods = n_periods
        self.n_oltc_actions = n_oltc_actions
        self.n_sc = n_sc
        self.n_sc_stages = n_sc_stages  # 不含0，总类别数 = n_sc_stages + 1
        self.n_pv = n_pv
        self.n_wt = n_wt
        self.n_svc = n_svc

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=n_periods + 10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 共享中间层
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ====== 离散动作头 ======
        # OLTC tap: (B, T, n_oltc_actions) logits
        self.oltc_head = nn.Linear(d_model, n_oltc_actions)

        # SC stage: (B, T, n_sc * (n_sc_stages+1)) → reshape
        n_sc_classes = n_sc_stages + 1
        self.sc_head = nn.Linear(d_model, n_sc * n_sc_classes) if n_sc > 0 else None

        # ====== 连续动作头 ======
        # PV Q: (B, T, n_pv) → tanh
        self.pv_q_head = nn.Linear(d_model, n_pv) if n_pv > 0 else None

        # WT Q: (B, T, n_wt) → tanh
        self.wt_q_head = nn.Linear(d_model, n_wt) if n_wt > 0 else None

        # SVC Q: (B, T, n_svc) → tanh
        self.svc_q_head = nn.Linear(d_model, n_svc) if n_svc > 0 else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, T, n_features)
        Returns:
            dict of action outputs
        """
        # 编码
        h = self.input_proj(x)         # (B, T, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)            # (B, T, d_model)
        h = self.shared_fc(h)          # (B, T, d_model)

        outputs = {}

        # OLTC
        outputs["oltc_logits"] = self.oltc_head(h)  # (B, T, n_oltc_actions)

        # SC
        if self.sc_head is not None:
            sc_raw = self.sc_head(h)  # (B, T, n_sc * n_classes)
            B, T, _ = sc_raw.shape
            n_classes = self.n_sc_stages + 1
            outputs["sc_logits"] = sc_raw.view(B, T, self.n_sc, n_classes)
        else:
            outputs["sc_logits"] = None

        # PV Q
        if self.pv_q_head is not None:
            outputs["pv_q"] = torch.tanh(self.pv_q_head(h))  # (B, T, n_pv)
        else:
            outputs["pv_q"] = None

        # WT Q
        if self.wt_q_head is not None:
            outputs["wt_q"] = torch.tanh(self.wt_q_head(h))  # (B, T, n_wt)
        else:
            outputs["wt_q"] = None

        # SVC Q
        if self.svc_q_head is not None:
            outputs["svc_q"] = torch.tanh(self.svc_q_head(h))  # (B, T, n_svc)
        else:
            outputs["svc_q"] = None

        return outputs


class VVCCriticNetwork(nn.Module):
    """
    VVC Critic 网络（BC中可选，为框架完整性保留）

    输入: (batch, T, n_features)
    输出: (batch, T, 1) 状态价值
    """

    def __init__(self, n_features: int, n_periods: int,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=n_periods + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.value_head(h)  # (B, T, 1)


def create_actor(meta: dict, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1) -> VVCActorNetwork:
    """根据 meta 信息创建 Actor 网络"""
    return VVCActorNetwork(
        n_features=meta["n_features"],
        n_periods=meta["n_periods"],
        n_oltc_actions=meta["n_oltc_actions"],
        n_sc=meta["n_sc"],
        n_sc_stages=meta["n_sc_stages"],
        n_pv=meta["n_pv"],
        n_wt=meta["n_wt"],
        n_svc=meta["n_svc"],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )