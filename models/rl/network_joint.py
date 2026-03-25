# -*- coding: utf-8 -*-
"""Task C (Joint) Actor 网络 — MLP + 离散/连续混合输出头"""

import torch
import torch.nn as nn


class JointActor(nn.Module):
    """MLP Actor: OLTC 分类头 + SC 分类头 + 连续动作头"""

    def __init__(self, n_features: int, n_oltc_actions: int,
                 n_sc: int, n_sc_stages: int, n_cont_actions: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_sc = n_sc

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.oltc_head = nn.Linear(hidden_dim, n_oltc_actions)
        self.sc_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n_sc_stages + 1) for _ in range(n_sc)]
        )
        self.cont_head = nn.Linear(hidden_dim, n_cont_actions)

    def forward(self, x):
        h = self.encoder(x)
        return {
            "oltc_logits": self.oltc_head(h),
            "sc_logits": [head(h) for head in self.sc_heads],
            "continuous": self.cont_head(h),
        }


def create_joint_actor(meta: dict, hidden_dim: int = 256) -> JointActor:
    return JointActor(
        n_features=meta["n_features"],
        n_oltc_actions=meta["n_oltc_actions"],
        n_sc=meta["n_sc"],
        n_sc_stages=meta["n_sc_stages"],
        n_cont_actions=meta["n_cont_actions"],
        hidden_dim=hidden_dim,
    )