# -*- coding: utf-8 -*-
"""Task C (Joint) 行为克隆智能体 — 接口与 bc_agent (VVC) 一致"""

import time
import numpy as np
import torch
import torch.nn.functional as F

from .network_joint import create_joint_actor


class JointBCAgent:
    """Joint BC Agent（接口与 VVC BCAgent 对齐）"""

    def __init__(self, meta: dict, device="cpu",
                 hidden_dim: int = 256, learning_rate: float = 1e-3):
        self.meta = meta
        self.device = torch.device(device)
        self.actor = create_joint_actor(meta, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------ train
    def train(self, train_loader, val_loader, n_epochs=100,
              early_stopping_patience=20, verbose=True):
        best_val, best_ep, best_state = float("inf"), 0, None
        patience = 0
        history = {"train_loss": [], "val_loss": []}
        t0 = time.time()

        for ep in range(n_epochs):
            # ---- train ----
            self.actor.train()
            t_loss, nb = 0.0, 0
            for batch in train_loader:
                feat, o_lbl, s_lbl, c_lbl = [b.to(self.device) for b in batch]
                out = self.actor(feat)
                loss = self._loss(out, o_lbl, s_lbl, c_lbl)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer.step()
                t_loss += loss.item(); nb += 1
            avg_t = t_loss / max(nb, 1)

            # ---- val ----
            self.actor.eval()
            v_loss, nv = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    feat, o_lbl, s_lbl, c_lbl = [b.to(self.device) for b in batch]
                    out = self.actor(feat)
                    v_loss += self._loss(out, o_lbl, s_lbl, c_lbl).item(); nv += 1
            avg_v = v_loss / max(nv, 1)

            history["train_loss"].append(avg_t)
            history["val_loss"].append(avg_v)

            if avg_v < best_val:
                best_val, best_ep = avg_v, ep + 1
                best_state = {k: v.cpu().clone() for k, v in self.actor.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if verbose and ((ep + 1) % 10 == 0 or ep == 0):
                print(f"  Epoch {ep+1:3d}/{n_epochs}: "
                      f"train={avg_t:.4f} val={avg_v:.4f} "
                      f"best={best_val:.4f}@{best_ep} "
                      f"time={time.time()-t0:.1f}s")

            if patience >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {ep+1}")
                break

        if best_state is not None:
            self.actor.load_state_dict(best_state)
        return {"best_val_loss": best_val, "best_epoch": best_ep, "history": history}

    # ---------------------------------------------------------------- predict
    def predict(self, input_norm: np.ndarray) -> dict:
        """开环预测: (N, F) → dict of arrays"""
        self.actor.eval()
        with torch.no_grad():
            x = torch.FloatTensor(input_norm).to(self.device)
            out = self.actor(x)
        return {
            "oltc_idx": out["oltc_logits"].argmax(-1).cpu().numpy(),
            "sc_stage": np.stack([lg.argmax(-1).cpu().numpy()
                                  for lg in out["sc_logits"]], axis=-1),
            "continuous": out["continuous"].cpu().numpy(),
        }

    # ------------------------------------------------------------- save/load
    def save_model(self, path, meta=None):
        data = {"actor_state_dict": self.actor.state_dict()}
        if meta is not None:
            data["meta"] = meta
        torch.save(data, path)
        print(f"  模型已保存: {path}")

    # ----------------------------------------------------------------- loss
    def _loss(self, out, o_lbl, s_lbl, c_lbl):
        l_o = F.cross_entropy(out["oltc_logits"], o_lbl)
        l_s = sum(F.cross_entropy(out["sc_logits"][i], s_lbl[:, i])
                  for i in range(self.meta["n_sc"]))
        l_c = F.mse_loss(out["continuous"], c_lbl)
        return l_o + l_s + 5.0 * l_c