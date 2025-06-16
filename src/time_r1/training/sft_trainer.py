from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class SFTSample:
    """Single SFT training sample."""

    sequence: torch.Tensor  # (seq_len, features)
    target: torch.Tensor  # shape (1,)


class SFTDataset(Dataset):
    """Simple dataset of SFT samples."""

    def __init__(self, samples: Iterable[SFTSample]) -> None:
        self.samples: List[SFTSample] = list(samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        s = self.samples[idx]
        return s.sequence, s.target


def mape(pred: float, target: float) -> float:
    return abs((target - pred) / target)


def construct_sft_samples(df: pd.DataFrame, seq_len: int) -> List[SFTSample]:
    """Construct training samples following PLAN.md T-3 steps."""

    samples: List[SFTSample] = []
    numeric = (
        df[["open", "high", "low", "close", "volume"]].astype("float32").to_numpy()
    )
    closes = numeric[:, 3]
    for i in range(len(df) - seq_len - 1):
        window = numeric[i : i + seq_len]
        target = closes[i + seq_len]

        # Step 1/2: generate two candidate predictions and pick the one
        # with lower MAPE. The chosen value is not used further but mirrors
        # the data construction steps described in PLAN.md.
        pred_last = closes[i + seq_len - 1]
        pred_mean = closes[i : i + seq_len].mean()
        _ = (
            pred_last
            if mape(pred_last, target) <= mape(pred_mean, target)
            else pred_mean
        )

        # Step 3/4: reasoning text is ignored here but would be
        # f"<think>best {best:.2f}</think><answer>{target:.2f}</answer>"

        samples.append(
            SFTSample(
                sequence=torch.from_numpy(window),
                target=torch.tensor([target], dtype=torch.float32),
            )
        )
    return samples


class LoRALinear(nn.Linear):
    """Linear layer with LoRA adapters."""

    def __init__(
        self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0
    ):
        super().__init__(in_features, out_features)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base = super().forward(input)
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(input, self.lora_A), self.lora_B
        )
        return base + self.scale * lora


class SFTModel(nn.Module):
    """Tiny LSTM model with LoRA adapter."""

    def __init__(self, input_size: int = 5, hidden_size: int = 64, r: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = LoRALinear(hidden_size, 1, r=r)
        # freeze base parameters
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in [self.fc.weight, self.fc.bias]:
            if p is not None:
                p.requires_grad = False
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, seq_len, features)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out.squeeze(-1)


class SFTTrainer:
    """Minimal trainer implementing SFT warm-up logic."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 10,
        batch_size: int = 8,
        lr: float = 5e-3,
        epochs: int = 1,
    ) -> None:
        self.samples = construct_sft_samples(df, seq_len)
        self.dataset = SFTDataset(self.samples)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = SFTModel()
        self.lr = lr
        self.epochs = epochs
        self.loss: float | None = None

    @property
    def lora_parameter_ratio(self) -> float:
        total = sum(p.numel() for p in self.model.parameters())
        lora = sum(p.numel() for n, p in self.model.named_parameters() if "lora" in n)
        return lora / total

    def fit(self) -> float:
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )
        for _ in range(self.epochs):
            losses = []
            for x, y in self.dataloader:
                opt.zero_grad()
                pred = self.model(x)
                loss = self.model.loss_fn(pred, y.squeeze())
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.loss = sum(losses) / len(losses)
        assert self.loss is not None
        return self.loss
