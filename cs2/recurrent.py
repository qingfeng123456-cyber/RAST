"""Full-prefix LSTM and GRU count baselines.

All methods use the existing legal autoregressive count decoder. Only the
encoder family or the conditional RPS training term changes. Existing retained
model definitions and their seeded construction are left untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .improved import conditional_ranked_score
from .models import Autoregressive


def _rps_weight(cfg):
    weight = float(cfg.get("conditional_rps_weight", .25))
    if weight < 0:
        raise ValueError("conditional_rps_weight must be nonnegative")
    return weight


class ConditionalRPSMixin:
    """Exact joint NLL plus the same ordered-count score as the final model."""

    def decode(self, ctx, n, y=None, return_aux=False):
        state, remaining = ctx, n.clone()
        previous = torch.zeros_like(n)
        nll = torch.zeros_like(n, dtype=torch.float32)
        rps = torch.zeros_like(n, dtype=torch.float32)
        output = []
        options = torch.arange(6, device=n.device)[None]
        for region in range(self.k - 1):
            ids = torch.full_like(n, region)
            embedded = self.region(ids) + self.count(previous) + self.remaining(remaining)
            state = self.cell(embedded, state)
            logits = self.head(state).float().masked_fill(options > remaining[:, None], -1e9)
            value = y[:, region] if y is not None else torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            nll = nll + F.cross_entropy(logits, value, reduction="none")
            if y is not None:
                rps = rps + conditional_ranked_score(logits, value)
            output.append(value)
            previous, remaining = value, remaining - value
        output.append(remaining)
        result = torch.stack(output, -1), nll
        return (*result, rps) if return_aux else result

    def loss(self, x, y, n):
        _, nll, rps = self.decode(self.encoder(x), n, y, return_aux=True)
        return (nll + self.rps_weight * rps).mean()


class RecurrentHistoryEncoder(nn.Module):
    """Encode every recorded valid frame with a conventional stacked RNN.

    Prefixes have right padding only. Their recorded validity channel, rather
    than feature magnitude, determines packed lengths. No history is cropped.
    """

    def __init__(self, features, k, cfg, kind):
        super().__init__()
        if features != 24 * k + 23:
            raise ValueError(f"Full-prefix schema requires {24*k+23} features; received {features}")
        self.valid_index = 2 * k + 5
        self.elapsed_index = 2 * k + 6 + 18 * k
        self.kind = kind
        self.dim = int(cfg.get("transformer_dim", cfg.get("hidden_dim", 192)))
        self.hidden_dim = int(cfg.get(f"{kind}_hidden_dim", 320 if kind == "lstm" else 384))
        self.layers = int(cfg.get("rnn_layers", 3))
        if min(self.dim, self.hidden_dim, self.layers) < 1:
            raise ValueError("Recurrent widths and layer count must be positive")
        dropout = float(cfg.get("dropout", .1))
        self.input = nn.Sequential(nn.Linear(features, self.dim), nn.LayerNorm(self.dim), nn.GELU())
        self.use_time_encoding = bool(cfg.get("rnn_time_encoding", True))
        self.position = nn.Linear(8, self.dim) if self.use_time_encoding else None
        rnn_class = {"lstm": nn.LSTM, "gru": nn.GRU}.get(kind)
        if rnn_class is None:
            raise ValueError(f"Unknown recurrent kind: {kind}")
        self.rnn = rnn_class(self.dim, self.hidden_dim, num_layers=self.layers,
                             dropout=dropout if self.layers > 1 else 0., batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.dim), nn.GELU())

    def forward(self, x):
        lengths = (x[:, :, self.valid_index] > .5).sum(-1).clamp_min(1).cpu()
        frames = self.input(x)
        if self.position is not None:
            elapsed = x[:, :, self.elapsed_index:self.elapsed_index+1] * 240
            scales = x.new_tensor([1., 5., 20., 100.])
            phase = elapsed / scales
            frames = frames + self.position(torch.cat([phase.sin(), phase.cos()], -1))
        packed = nn.utils.rnn.pack_padded_sequence(frames, lengths, batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        hidden = state[0] if self.kind == "lstm" else state
        return self.output(hidden[-1])


class RecurrentCount(ConditionalRPSMixin, Autoregressive):
    exact_nll = True
    # Packed-RNN BF16 support depends on the target CUDA/cuDNN build. FP32 is
    # deliberately portable; retained Transformers keep their existing AMP.
    supports_amp = False

    def __init__(self, features, k, cfg, kind):
        nn.Module.__init__(self)
        self.k = k
        self.encoder = RecurrentHistoryEncoder(features, k, cfg, kind)
        dim = self.encoder.dim
        self.region = nn.Embedding(k, dim)
        self.count = nn.Embedding(6, dim)
        self.remaining = nn.Embedding(6, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.head = nn.Linear(dim, 6)
        self.rps_weight = _rps_weight(cfg)


class LSTMCount(RecurrentCount):
    def __init__(self, features, k, cfg):
        super().__init__(features, k, cfg, "lstm")


class GRUCount(RecurrentCount):
    def __init__(self, features, k, cfg):
        super().__init__(features, k, cfg, "gru")


def register_models(registry):
    """Register after definitions, supporting either public import order."""
    registry.update({"lstm_count": LSTMCount, "gru_count": GRUCount})


from .models import MODEL_CLASSES as _registry
register_models(_registry)
