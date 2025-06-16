from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..inference.filters import is_toxic
from .adaptive_weighting import adaptive_weights
from .checkpointing import CheckpointManager
from .reward import total_reward
from .sft_trainer import SFTDataset, SFTModel, construct_sft_samples


@dataclass
class GRIPMetrics:
    kl: float
    reward: float


class GRIPTrainer:
    """Minimal GRIP RL trainer."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 5,
        batch_size: int = 8,
        lr: float = 1e-3,
        epochs: int = 3,
        ckpt_dir: str = "checkpoints",
    ) -> None:
        self.samples = construct_sft_samples(df, seq_len)
        self.dataset = SFTDataset(self.samples)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = SFTModel()
        self.ref_model = SFTModel()
        self.ref_model.load_state_dict(self.model.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.lr = lr
        self.epochs = epochs
        self.ckpt = CheckpointManager(ckpt_dir, k=2)
        self.metrics = GRIPMetrics(kl=0.0, reward=0.0)

    def _kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(p, q)

    def fit(self) -> GRIPMetrics:
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )
        for step in range(1, self.epochs + 1):
            rewards: List[float] = []
            kls: List[float] = []
            for x, y in self.loader:
                opt.zero_grad()
                pred = self.model(x)
                with torch.no_grad():
                    ref = self.ref_model(x)
                kl = self._kl(pred, ref)
                kls.append(float(kl.item()))
                batch_rewards = []
                for pv, tv in zip(pred.tolist(), y.squeeze().tolist()):
                    text = f"<think>{pv:.2f}</think><answer>{pv:.2f}</answer>"
                    if is_toxic(text):
                        r = -1.0
                    else:
                        r = total_reward(text, f"{pv:.2f}", [pv], [tv])
                    batch_rewards.append(r)
                rewards.append(float(np.mean(batch_rewards)))
                weights = torch.tensor(
                    adaptive_weights(batch_rewards), dtype=torch.float32
                )
                loss = -(torch.tensor(batch_rewards, dtype=torch.float32) @ weights)
                loss = loss + 0.1 * kl
                loss.backward()
                opt.step()
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            mean_kl = float(np.mean(kls)) if kls else 0.0
            self.ckpt.save(self.model.state_dict(), mean_reward, step)
            self.metrics = GRIPMetrics(kl=mean_kl, reward=mean_reward)
        return self.metrics
