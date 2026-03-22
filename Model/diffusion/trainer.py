# -*- coding: utf-8 -*-
"""Model.diffusion.trainer

精简版 Diffusion-TS 训练器。
仅保留 uncondition 训练 + EMA + checkpoint 保存/加载。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from ema_pytorch import EMA

from engine.lr_sch import ReduceLROnPlateauWithWarmup


def _cycle(dl):
    """无限循环 DataLoader。"""
    while True:
        for data in dl:
            yield data


class DiffusionTrainer:
    """Diffusion-TS 轻量训练器（仅 uncondition 模式）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.dl = _cycle(dataloader)
        self.dataloader = dataloader

        tcfg = config["training"]
        self.max_epochs = tcfg["max_epochs"]
        self.gradient_accumulate_every = tcfg["gradient_accumulate_every"]
        self.save_cycle = tcfg["save_cycle"]

        # Optimizer
        lr = tcfg["lr"]
        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            betas=(0.9, 0.96),
        )

        # EMA
        self.ema = EMA(
            self.model,
            beta=tcfg["ema_decay"],
            update_every=tcfg["ema_update_every"],
        ).to(self.device)

        # LR scheduler
        sch_cfg = tcfg["scheduler"]
        self.sch = ReduceLROnPlateauWithWarmup(
            optimizer=self.opt,
            factor=sch_cfg["factor"],
            patience=sch_cfg["patience"],
            min_lr=sch_cfg["min_lr"],
            threshold=sch_cfg["threshold"],
            threshold_mode=sch_cfg["threshold_mode"],
            warmup_lr=tcfg["warmup_lr"],
            warmup=tcfg["warmup_steps"],
            verbose=False,
        )

        self.step = 0
        self.milestone = 0

    def train(self, results_folder: str, verbose: bool = True) -> None:
        """执行训练循环。"""
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        device = self.device

        tic = time.time()
        step = 0
        with tqdm(
            initial=step,
            total=self.max_epochs,
            desc="Diffusion training",
            disable=not verbose,
        ) as pbar:
            while step < self.max_epochs:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f"loss: {total_loss:.6f}")
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(results_folder, self.milestone)

                pbar.update(1)

        # 训练完成后保存最终 checkpoint
        self.milestone += 1
        self.save(results_folder, self.milestone, verbose=verbose)
        if verbose:
            print(
                f"Training complete. {step} steps, "
                f"time: {time.time() - tic:.1f}s, "
                f"saved to {results_folder}"
            )

    def save(self, folder: Path, milestone: int, verbose: bool = False) -> None:
        path = folder / f"checkpoint-{milestone}.pt"
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, str(path))
        if verbose:
            print(f"  Saved checkpoint: {path}")

    def load(self, folder: str, milestone: int) -> None:
        path = Path(folder) / f"checkpoint-{milestone}.pt"
        data = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.opt.load_state_dict(data["opt"])
        self.step = data["step"]
        self.milestone = milestone

    def get_latest_milestone(self, folder: str) -> Optional[int]:
        """扫描目录找到最大 milestone 编号。"""
        folder = Path(folder)
        if not folder.exists():
            return None
        milestones = []
        for f in folder.glob("checkpoint-*.pt"):
            try:
                m = int(f.stem.split("-")[1])
                milestones.append(m)
            except (IndexError, ValueError):
                continue
        return max(milestones) if milestones else None

    @torch.no_grad()
    def sample(
        self, n_samples: int, shape: tuple, batch_size: int = 256
    ) -> np.ndarray:
        """从 EMA 模型生成 uncondition 样本。"""
        samples = np.empty([0, shape[0], shape[1]])
        remaining = n_samples
        while remaining > 0:
            bs = min(batch_size, remaining)
            s = self.ema.ema_model.generate_mts(batch_size=bs)
            samples = np.row_stack([samples, s.detach().cpu().numpy()])
            remaining -= bs
        return samples[:n_samples]
