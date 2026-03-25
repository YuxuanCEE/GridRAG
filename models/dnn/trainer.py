# -*- coding: utf-8 -*-
"""
DNN Baseline 训练器 - Task B (ED)

功能:
- 模型训练和验证
- 学习率调度
- 早停机制
- 模型保存和加载
- 训练日志
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

from .dnn_model import EDTransformerModel, EDLoss, create_model


class EDTrainer:
    """ED任务DNN训练器"""

    def __init__(self,
                 model: EDTransformerModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device = None,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 continuous_weight: float = 1.0,
                 binary_weight: float = 1.0):
        """
        Args:
            model: EDTransformerModel实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            continuous_weight: 连续变量损失权重
            binary_weight: 二进制变量损失权重
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 损失函数
        self.criterion = EDLoss(
            continuous_weight=continuous_weight,
            binary_weight=binary_weight
        )

        # 学习率调度器（移除verbose参数，兼容新版PyTorch）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_continuous_loss": [],
            "train_binary_loss": [],
            "val_continuous_loss": [],
            "val_binary_loss": [],
            "learning_rate": [],
        }

        # 最佳模型
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0

        # 训练时间统计
        self.train_time = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_continuous_loss = 0.0
        total_binary_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # 数据移至设备
            input_data = batch["input"].to(self.device)
            target_continuous = batch["continuous"].to(self.device)
            target_binary = batch["binary"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            pred_continuous, pred_binary_logits = self.model(input_data)

            # 计算损失
            losses = self.criterion(
                pred_continuous, pred_binary_logits,
                target_continuous, target_binary
            )

            # 反向传播
            losses["total"].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += losses["total"].item()
            total_continuous_loss += losses["continuous"].item()
            total_binary_loss += losses["binary"].item()
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "continuous_loss": total_continuous_loss / n_batches,
            "binary_loss": total_binary_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0.0
        total_continuous_loss = 0.0
        total_binary_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_data = batch["input"].to(self.device)
            target_continuous = batch["continuous"].to(self.device)
            target_binary = batch["binary"].to(self.device)

            pred_continuous, pred_binary_logits = self.model(input_data)

            losses = self.criterion(
                pred_continuous, pred_binary_logits,
                target_continuous, target_binary
            )

            total_loss += losses["total"].item()
            total_continuous_loss += losses["continuous"].item()
            total_binary_loss += losses["binary"].item()
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "continuous_loss": total_continuous_loss / n_batches,
            "binary_loss": total_binary_loss / n_batches,
        }

    def train(self,
              n_epochs: int = 100,
              early_stopping_patience: int = 20,
              verbose: bool = True) -> Dict:
        """
        完整训练流程

        Args:
            n_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            verbose: 是否打印详细信息

        Returns:
            训练结果字典
        """
        start_time = time.time()
        patience_counter = 0

        if verbose:
            print(f"\n开始训练 (设备: {self.device})")
            print(f"  训练样本: {len(self.train_loader.dataset)}")
            print(f"  验证样本: {len(self.val_loader.dataset)}")
            print(f"  总轮数: {n_epochs}")
            print("-" * 60)

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 学习率调度
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]['lr']

            # 如果学习率变化，打印信息
            if verbose and current_lr != old_lr:
                print(f"  >> 学习率调整: {old_lr:.2e} -> {current_lr:.2e}")

            # 记录历史
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_continuous_loss"].append(train_metrics["continuous_loss"])
            self.history["train_binary_loss"].append(train_metrics["binary_loss"])
            self.history["val_continuous_loss"].append(val_metrics["continuous_loss"])
            self.history["val_binary_loss"].append(val_metrics["binary_loss"])
            self.history["learning_rate"].append(current_lr)

            # 检查是否为最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")

            # 早停检查
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n早停触发 (epoch {epoch + 1}), 最佳epoch: {self.best_epoch + 1}")
                break

        self.train_time = time.time() - start_time

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        if verbose:
            print("-" * 60)
            print(f"训练完成!")
            print(f"  总训练时间: {self.train_time:.2f}s")
            print(f"  最佳验证损失: {self.best_val_loss:.4f} (epoch {self.best_epoch + 1})")

        return {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_time": self.train_time,
            "history": self.history,
        }

    def save_model(self, save_path: Path, meta: Dict = None):
        """保存模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "n_features": self.model.n_features,
                "n_continuous": self.model.n_continuous,
                "n_binary": self.model.n_binary,
                "n_periods": self.model.n_periods,
                "d_model": self.model.d_model,
            },
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_time": self.train_time,
            "history": self.history,
            "meta": meta,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")

    @classmethod
    def load_model(cls, load_path: Path, device: torch.device = None) -> Tuple[EDTransformerModel, Dict]:
        """加载模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        # 创建模型
        config = checkpoint["model_config"]
        model = EDTransformerModel(
            n_features=config["n_features"],
            n_continuous=config["n_continuous"],
            n_binary=config["n_binary"],
            n_periods=config["n_periods"],
            d_model=config["d_model"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, checkpoint


if __name__ == "__main__":
    # 简单测试
    from torch.utils.data import TensorDataset

    # 模拟数据
    n_samples = 100
    n_periods = 96
    n_features = 20
    n_continuous = 7
    n_binary = 6

    X = torch.randn(n_samples, n_periods, n_features)
    Y_cont = torch.randn(n_samples, n_periods, n_continuous)
    Y_bin = torch.randint(0, 2, (n_samples, n_periods, n_binary)).float()

    # 创建数据加载器
    from torch.utils.data import DataLoader


    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y_cont, Y_bin):
            self.X = X
            self.Y_cont = Y_cont
            self.Y_bin = Y_bin

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {"input": self.X[idx], "continuous": self.Y_cont[idx], "binary": self.Y_bin[idx]}


    dataset = SimpleDataset(X, Y_cont, Y_bin)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 创建模型和训练器
    model = EDTransformerModel(n_features, n_continuous, n_binary, n_periods)
    trainer = EDTrainer(model, train_loader, val_loader)

    # 训练
    results = trainer.train(n_epochs=20, verbose=True)