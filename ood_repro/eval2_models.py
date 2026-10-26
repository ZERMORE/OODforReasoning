"""Small OFFLINE sequence classifiers; padding-masked, with a shared length feature."""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    def __init__(self, kind, channels=1, hidden=16):
        super().__init__()
        if kind not in {"deepsets", "gru", "cnn", "attention", "attention_nopos"}:
            raise ValueError(kind)
        if hidden < 4 or hidden % 2:
            raise ValueError("hidden must be even and >=4")
        self.kind, self.hidden = kind, hidden
        self.input = nn.Linear(channels, hidden)
        if kind == "gru":
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        elif kind == "cnn":
            self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        elif kind.startswith("attention"):
            layer = nn.TransformerEncoderLayer(hidden, 2, hidden * 2, dropout=0., batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, lengths, length_feature):
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        state = torch.tanh(self.input(x)) * mask[..., None]
        if self.kind == "gru":
            packed = pack_padded_sequence(state, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.encoder(packed)
            pooled = hidden[-1]
        else:
            if self.kind == "cnn":
                state = torch.tanh(self.conv1(state.transpose(1, 2))).transpose(1, 2) * mask[..., None]
                state = torch.tanh(self.conv2(state.transpose(1, 2))).transpose(1, 2)
            elif self.kind.startswith("attention"):
                if self.kind == "attention":
                    position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
                    frequency = torch.exp(torch.arange(0, self.hidden, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / self.hidden))
                    encoding = torch.zeros((x.shape[1], self.hidden), device=x.device, dtype=x.dtype)
                    encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
                    state = state + encoding[None] * mask[..., None]
                state = self.encoder(state, src_key_padding_mask=~mask)
            else:
                state = self.encoder(state)
            pooled = (state * mask[..., None]).sum(1) / lengths[:, None]
        return self.head(torch.cat([pooled, length_feature[:, None]], dim=1)).squeeze(-1)
