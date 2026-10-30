from __future__ import annotations


import torch


from torch import nn


from torch.nn import functional as F


def log_coefficient(y):
    return torch.lgamma(y.sum(-1).float() + 1) - torch.lgamma(y.float() + 1).sum(-1)


def multinomial_counts(probabilities, n, samples):
    b, k = probabilities.shape
    draws = torch.multinomial(probabilities, samples * 5, replacement=True).reshape(b, samples, 5)
    valid = torch.arange(5, device=n.device)[None, None, :] < n[:, None, None]
    counts = torch.zeros(b, samples, k, device=n.device)
    counts.scatter_add_(2, draws, valid.expand(b, samples, 5).float())
    return counts


class HistoryEncoder(nn.Module):
    def __init__(self, features, dim):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, dim), nn.LayerNorm(dim), nn.GELU())
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x):
        lengths = (x.abs().sum(-1) > 0).sum(-1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.input(x), lengths, batch_first=True, enforce_sorted=False)
        return self.gru(packed)[1][-1]


class Autoregressive(nn.Module):
    """Exact count likelihood, with every conditional truncated to remaining mass."""
    exact_nll = True

    def __init__(self, features, k, cfg):
        super().__init__()
        dim = cfg["hidden_dim"]
        self.k = k
        self.encoder = HistoryEncoder(features, dim)
        self.region = nn.Embedding(k, dim)
        self.count = nn.Embedding(6, dim)
        self.remaining = nn.Embedding(6, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.head = nn.Linear(dim, 6)

    def decode(self, ctx, n, y=None):
        state, remaining = ctx, n.clone()
        previous = torch.zeros_like(n)
        output, nll = [], torch.zeros_like(n, dtype=torch.float32)
        options = torch.arange(6, device=ctx.device)[None]
        for region in range(self.k - 1):
            r = torch.full_like(n, region)
            state = self.cell(self.region(r) + self.count(previous) + self.remaining(remaining), state)
            logits = self.head(state).masked_fill(options > remaining[:, None], -1e9)
            value = y[:, region] if y is not None else torch.multinomial(logits.float().softmax(-1), 1).squeeze(-1)
            nll += F.cross_entropy(logits.float(), value, reduction="none")
            output.append(value)
            previous, remaining = value, remaining - value
        output.append(remaining)
        return torch.stack(output, -1), nll

    def nll(self, x, y, n):
        return self.decode(self.encoder(x), n, y)[1]

    def loss(self, x, y, n):
        return self.nll(x, y, n).mean()

    def sample(self, x, n, samples):
        ctx = self.encoder(x).repeat_interleave(samples, 0)
        return self.decode(ctx, n.repeat_interleave(samples))[0].reshape(len(x), samples, self.k).float()

    def infer(self, x, y, n, samples):
        ctx = self.encoder(x)
        draws = self.decode(ctx.repeat_interleave(samples,0), n.repeat_interleave(samples))[0].reshape(len(x),samples,self.k).float()
        return draws, self.decode(ctx,n,y)[1]


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.width, self.dropout = heads, dim // heads, dropout
        self.qkv = nn.Linear(dim, 3*dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, bias=None, causal=False):
        b, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0., is_causal=causal)
        return self.output(result.transpose(1, 2).reshape(b, length, dim))


class TacticalBlock(nn.Module):
    """Factorized spatial/causal temporal attention with learned residual gates.

    Temporal keys need no padding mask: right padding occurs strictly AFTER all
    real queries and causal attention cannot propagate it backwards. Spatial
    attention never mixes different times. Last-valid-time pooling discards pads.
    """
    def __init__(self, dim, heads, dropout, distances, structured):
        super().__init__()
        self.structured = structured
        self.spatial, self.temporal = Attention(dim, heads, dropout), Attention(dim, heads, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.spatial_scale = nn.Parameter(torch.full((dim,), .1))
        self.temporal_scale = nn.Parameter(torch.full((dim,), .1))
        self.ffn_scale = nn.Parameter(torch.full((dim,), .1))
        self.gate = nn.Linear(12, dim) if structured else None
        self.distance_weight = nn.Parameter(torch.zeros(heads)) if structured else None
        self.register_buffer("distances", distances)

    def forward(self, x, evidence):
        b, t, k, d = x.shape
        bias = -F.softplus(self.distance_weight)[None, :, None, None] * self.distances if self.structured and not getattr(self, "diagnostic_no_distance", False) and not getattr(self, "ablation_no_distance", False) else None
        spatial = self.spatial(self.norms[0](x).reshape(b*t, k, d), bias=bias).reshape(b, t, k, d)
        x = x + self.spatial_scale.tanh() * spatial
        temporal_input = self.norms[1](x).transpose(1, 2).reshape(b*k, t, d)
        temporal = self.temporal(temporal_input, causal=True).reshape(b, k, t, d).transpose(1, 2)
        gate = .5 + self.gate(evidence).sigmoid() if self.structured else 1.
        if getattr(self, "diagnostic_neutral_gate", False) or getattr(self, "ablation_neutral_gate", False):
            gate = 1.
        x = x + self.temporal_scale.tanh() * gate * temporal
        return x + self.ffn_scale.tanh() * self.ffn(self.norms[2](x))


class TacticalEncoder(nn.Module):
    def __init__(self, features, k, cfg, structured=True, use_skill=True, use_smoke=True):
        super().__init__()
        self.k, self.aux = k, 14
        self.smoke_start, self.memory_start = 2*k+6+14*k, 2*k+6+16*k
        self.extra_start, self.skill_start = 2*k+6+18*k, 2*k+6+18*k+2
        self.acoustic_start = 20*k+22
        self.use_skill, self.use_smoke = use_skill, use_smoke
        if features != 24*k+23:
            raise ValueError(f"Full-prefix feature schema requires {24*k+23} features, got {features}")
        dim = cfg.get("transformer_dim", cfg["hidden_dim"])
        heads = cfg.get("transformer_heads", 4)
        if dim % heads:
            raise ValueError("Transformer dimension must be divisible by number of heads")
        centers = torch.tensor(cfg["region_centers"], dtype=torch.float32)
        distances = torch.cdist(centers, centers)
        distances = distances / distances[distances > 0].median().clamp_min(1)
        self.regional_input, self.global_input = nn.Linear(24, dim), nn.Linear(23, dim)
        self.region = nn.Parameter(torch.randn(k, dim)*.02)
        self.position = nn.Linear(8, dim)
        self.blocks = nn.ModuleList([TacticalBlock(dim, heads, cfg.get("dropout", .1), distances, structured)
                                    for _ in range(cfg.get("transformer_layers", 2))])
        self.pool = nn.Sequential(nn.LayerNorm(2*dim), nn.Linear(2*dim, dim), nn.GELU())

    def forward(self, x, return_regions=False):
        b, t, _ = x.shape
        k = self.k
        own, exposed = x[:, :, :k], x[:, :, k:2*k]
        auxiliary = x[:, :, 2*k+6:self.smoke_start].reshape(b, t, k, 14)
        smoke = x[:, :, self.smoke_start:self.memory_start].reshape(b, t, k, 2)
        memory = x[:, :, self.memory_start:self.extra_start].reshape(b, t, k, 2)
        extra = x[:, :, self.extra_start:self.acoustic_start]
        acoustic = x[:, :, self.acoustic_start:self.acoustic_start+4*k].reshape(b, t, k, 4)
        if not self.use_smoke:
            smoke = torch.zeros_like(smoke)
        if not self.use_skill:
            extra = torch.cat([extra[:, :, :2], torch.zeros_like(extra[:, :, 2:])], -1)
        regional = torch.cat([own[..., None], exposed[..., None], auxiliary, smoke, memory, acoustic], -1)
        global_features = torch.cat([x[:, :, 2*k:2*k+6], extra, x[:, :, -1:]], -1)
        # Continuous time encodings preserve variable event-aligned sampling gaps.
        elapsed = extra[:, :, 0:1]*240
        scales = x.new_tensor([1., 5., 20., 100.])
        phase = elapsed / scales
        position = self.position(torch.cat([phase.sin(), phase.cos()], -1))
        nodes = self.regional_input(regional) + self.global_input(global_features)[:, :, None] + self.region + position[:, :, None]
        public_population = x[:, :, 2*k:2*k+2][:, :, None].expand(-1, -1, k, -1)
        audio_available = x[:, :, -1:][:, :, None].expand(-1, -1, k, -1)
        evidence = torch.cat([smoke, memory, exposed[..., None], acoustic, audio_available, public_population], -1)
        for block in self.blocks:
            nodes = block(nodes, evidence)
        last_index = (x[:, :, 2*k+5] > .5).sum(-1).clamp_min(1) - 1
        current = nodes[torch.arange(b, device=x.device), last_index]
        pooled = self.pool(torch.cat([current.mean(1), current.amax(1)], -1))
        return (pooled,current) if return_regions else pooled


class TacticalTransformer(Autoregressive):
    """Game-conditioned full-prefix encoder plus exact joint count decoder.

    A candidate research architecture; novelty and superiority require a
    literature search and controlled experiments, not a module name.
    """
    def __init__(self, features, k, cfg, structured=True, use_skill=True, use_smoke=True):
        decoder_cfg = dict(cfg, hidden_dim=cfg.get("transformer_dim", cfg["hidden_dim"]))
        super().__init__(features, k, decoder_cfg)
        self.encoder = TacticalEncoder(features, k, cfg, structured, use_skill, use_smoke)
        self.supports_amp = True


class VanillaTransformer(TacticalTransformer):
    def __init__(self, features, k, cfg):
        super().__init__(features, k, cfg, structured=False)


from .improved import ImprovedTransformer

MODEL_CLASSES = {"vanilla_transformer": VanillaTransformer, "improved_transformer": ImprovedTransformer}

# Supplemental registration leaves the retained definitions untouched. The
# module also registers on completion when imported directly before models.
from . import recurrent as _recurrent

if hasattr(_recurrent, "register_models"):
    _recurrent.register_models(MODEL_CLASSES)
