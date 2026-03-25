# -*- coding: utf-8 -*-
"""
行为克隆 (Behavioral Cloning) 智能体

纯离线模仿学习：
- 离散动作使用交叉熵损失
- 连续动作使用MSE损失
- 支持早停、学习率衰减
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional

from models.rl.network import VVCActorNetwork, create_actor


class BCAgent:
    """行为克隆智能体"""

    def __init__(self, meta: Dict, device: torch.device = None,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 learning_rate: float = 1e-3,
                 discrete_weight: float = 1.0,
                 continuous_weight: float = 1.0):
        """
        Args:
            meta: 数据元信息（来自 VVCDataProcessor.prepare_dataset）
            device: 计算设备
            d_model, n_heads, n_layers: Transformer 超参数
            learning_rate: 学习率
            discrete_weight: 离散动作损失权重
            continuous_weight: 连续动作损失权重
        """
        self.meta = meta
        self.device = device or torch.device("cpu")
        self.discrete_weight = discrete_weight
        self.continuous_weight = continuous_weight

        # 创建 Actor
        self.actor = create_actor(meta, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.actor.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)

    def _compute_loss(self, outputs: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """计算 BC 混合损失"""
        losses = {}

        # OLTC 交叉熵
        oltc_logits = outputs["oltc_logits"]  # (B, T, n_classes)
        oltc_target = batch["oltc_class"].to(self.device)  # (B, T)
        B, T, C = oltc_logits.shape
        losses["oltc_ce"] = F.cross_entropy(oltc_logits.reshape(B * T, C), oltc_target.reshape(B * T))

        # SC 交叉熵
        if outputs["sc_logits"] is not None:
            sc_logits = outputs["sc_logits"]  # (B, T, n_sc, n_classes)
            sc_target = batch["sc_class"].to(self.device)  # (B, T, n_sc)
            n_sc = sc_logits.shape[2]
            n_cls = sc_logits.shape[3]
            sc_loss = 0
            for k in range(n_sc):
                sc_loss += F.cross_entropy(
                    sc_logits[:, :, k, :].reshape(B * T, n_cls),
                    sc_target[:, :, k].reshape(B * T)
                )
            losses["sc_ce"] = sc_loss / max(n_sc, 1)
        else:
            losses["sc_ce"] = torch.tensor(0.0, device=self.device)

        # PV Q MSE
        if outputs["pv_q"] is not None and self.meta["n_pv"] > 0:
            losses["pv_q_mse"] = F.mse_loss(outputs["pv_q"], batch["pv_q"].to(self.device))
        else:
            losses["pv_q_mse"] = torch.tensor(0.0, device=self.device)

        # WT Q MSE
        if outputs["wt_q"] is not None and self.meta["n_wt"] > 0:
            losses["wt_q_mse"] = F.mse_loss(outputs["wt_q"], batch["wt_q"].to(self.device))
        else:
            losses["wt_q_mse"] = torch.tensor(0.0, device=self.device)

        # SVC Q MSE
        if outputs["svc_q"] is not None and self.meta["n_svc"] > 0:
            losses["svc_q_mse"] = F.mse_loss(outputs["svc_q"], batch["svc_q"].to(self.device))
        else:
            losses["svc_q_mse"] = torch.tensor(0.0, device=self.device)

        # 汇总
        discrete_loss = losses["oltc_ce"] + losses["sc_ce"]
        continuous_loss = losses["pv_q_mse"] + losses["wt_q_mse"] + losses["svc_q_mse"]
        total_loss = self.discrete_weight * discrete_loss + self.continuous_weight * continuous_loss

        losses["discrete"] = discrete_loss.item()
        losses["continuous"] = continuous_loss.item()
        losses["total"] = total_loss.item()

        return total_loss, losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int = 100, early_stopping_patience: int = 20,
              verbose: bool = True) -> Dict:
        """
        训练 BC

        Returns:
            训练结果字典
        """
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(n_epochs):
            # ---- 训练 ----
            self.actor.train()
            epoch_loss = 0
            n_batches = 0
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                outputs = self.actor(inputs)
                loss, loss_dict = self._compute_loss(outputs, batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss_dict["total"]
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # ---- 验证 ----
            self.actor.eval()
            val_loss_sum = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input"].to(self.device)
                    outputs = self.actor(inputs)
                    _, loss_dict = self._compute_loss(outputs, batch)
                    val_loss_sum += loss_dict["total"]
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            history["val_loss"].append(avg_val_loss)

            self.scheduler.step()

            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.actor.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch:3d}/{n_epochs} | "
                      f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                      f"Best: {best_val_loss:.4f} (epoch {best_epoch})")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  早停于 epoch {epoch}, 最佳 epoch {best_epoch}")
                break

        # 恢复最优权重
        if best_state is not None:
            self.actor.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
        }

    @torch.no_grad()
    def predict(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        推理：输入标准化后的特征 → 输出原始动作

        Args:
            input_tensor: (96, n_features) 或 (1, 96, n_features)

        Returns:
            动作字典（已反归一化为实际值）
        """
        self.actor.eval()
        if input_tensor.ndim == 2:
            input_tensor = input_tensor[np.newaxis, :]
        x = torch.from_numpy(input_tensor).float().to(self.device)
        outputs = self.actor(x)

        # OLTC tap
        oltc_logits = outputs["oltc_logits"][0].cpu().numpy()  # (T, n_actions)
        oltc_class = np.argmax(oltc_logits, axis=-1)           # (T,)
        oltc_tap = oltc_class + self.meta["tap_min"]           # class → 实际 tap

        # SC stage
        if outputs["sc_logits"] is not None:
            sc_logits = outputs["sc_logits"][0].cpu().numpy()  # (T, n_sc, n_classes)
            sc_stage = np.argmax(sc_logits, axis=-1)           # (T, n_sc)
        else:
            sc_stage = np.zeros((self.meta["n_periods"], 0), dtype=np.int64)

        # SC Q (由 stage 确定)
        sc_q_mvar = sc_stage.astype(np.float64) * self.meta["sc_q_per_stage"]

        # PV Q
        if outputs["pv_q"] is not None and self.meta["n_pv"] > 0:
            pv_q_norm = outputs["pv_q"][0].cpu().numpy()  # (T, n_pv) in [-1,1]
            pv_q_mvar = np.zeros_like(pv_q_norm)
            for i, q_max in enumerate(self.meta["pv_q_max"]):
                pv_q_mvar[:, i] = pv_q_norm[:, i] * q_max
        else:
            pv_q_mvar = np.zeros((self.meta["n_periods"], 0))

        # WT Q
        if outputs["wt_q"] is not None and self.meta["n_wt"] > 0:
            wt_q_norm = outputs["wt_q"][0].cpu().numpy()
            wt_q_mvar = np.zeros_like(wt_q_norm)
            for i, q_max in enumerate(self.meta["wt_q_max"]):
                wt_q_mvar[:, i] = wt_q_norm[:, i] * q_max
        else:
            wt_q_mvar = np.zeros((self.meta["n_periods"], 0))

        # SVC Q
        if outputs["svc_q"] is not None and self.meta["n_svc"] > 0:
            svc_q_norm = outputs["svc_q"][0].cpu().numpy()
            svc_q_mvar = svc_q_norm * self.meta["svc_q_max"]
        else:
            svc_q_mvar = np.zeros((self.meta["n_periods"], 0))

        return {
            "oltc_tap": oltc_tap.astype(np.int64),
            "sc_stage": sc_stage.astype(np.int64),
            "sc_q_mvar": sc_q_mvar,
            "pv_q_mvar": pv_q_mvar,
            "wt_q_mvar": wt_q_mvar,
            "svc_q_mvar": svc_q_mvar,
        }

    def save_model(self, path: Path, meta: Dict = None):
        save_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "meta": meta or self.meta,
        }
        torch.save(save_dict, path)
        print(f"  模型已保存: {path}")

    def load_model(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        if "meta" in ckpt:
            self.meta.update(ckpt["meta"])
        print(f"  模型已加载: {path}")