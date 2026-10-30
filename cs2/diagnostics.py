"""Frozen-checkpoint mechanism probes; all results are saved for plot-only reuse."""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .io import save_json

LOG = logging.getLogger("cs2")


def selection(data, cfg, output):
    path = output / "diagnostic_selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    test = np.flatnonzero(data["split"] == "test")
    table = pd.DataFrame({"index": test, "group": data["group"][test], "round": data["round"][test], "elapsed": data["elapsed"][test]})
    candidates = []
    for _, sub in table.groupby(["group", "round"], sort=True):
        # Outcome-independent, prespecified cutoffs, at most two per round.
        for target in (20., 60.):
            candidates.append(int(sub.iloc[np.abs(sub.elapsed-target).argmin()]["index"]))
    candidates = np.unique(candidates)
    rng = np.random.default_rng(1717)
    rng.shuffle(candidates)
    limit = cfg.get("diagnostic_samples", 256)
    # Round-robin matches prevents one long demo from dominating the selection.
    pools = [list(candidates[data["group"][candidates] == group]) for group in np.unique(data["group"][test])]
    chosen = []
    while len(chosen) < limit and any(pools):
        for pool in pools:
            if pool and len(chosen) < limit:
                chosen.append(pool.pop())
    chosen = sorted(int(i) for i in chosen)
    positions = {int(i): p for p, i in enumerate(test)}
    k = data["y"].shape[-1]
    rows = []
    for i in chosen:
        current = data["frames"][data["end"][i]-1]
        rows.append({"index": i, "test_position": positions[i], "group": str(data["group"][i]), "round": int(data["round"][i]),
                     "demo": str(data["demo"][i]),
                     "tick": int(data["tick"][i]), "n": int(data["n"][i]), "elapsed": float(data["elapsed"][i]),
                     "ct_alive": int(round(current[2*k]*5)), "t_alive": int(round(current[2*k+1]*5)),
                     "smoke_active": int(data["smoke_active"][i]), "contact_age": float(data["contact_age"][i]),
                     "audio_available": bool(current[-1] > .5), "ct_elo": float(data["ct_elo"][i]), "t_elo": float(data["t_elo"][i]),
                     "ct_elo_known": bool(current[20*k+8+5] > .5), "t_elo_known": bool(current[20*k+8+7+5] > .5),
                     "ct_team": str(data["ct_team"][i]), "t_team": str(data["t_team"][i])})
    # Predefined cases: active smoke, stale radar, population diversity. Never rank by model success.
    priority = sorted(range(len(rows)), key=lambda j: (-rows[j]["smoke_active"], abs(rows[j]["elapsed"]-40), rows[j]["index"]))
    cases = priority[:1]
    for n in range(1, 6):
        eligible = [j for j, row in enumerate(rows) if row["n"] == n and j not in cases]
        if eligible:
            cases.append(min(eligible, key=lambda j: abs(rows[j]["elapsed"]-40)))
    payload = {"selection_rule": "at most two nearest cutoffs to 20/60 live seconds per round, match-balanced seeded subsample; cases use input context/population, never prediction quality",
               "rows": rows, "case_rows": cases[:6], "seed": 1717, "split": "test", "matches": len(pools)}
    save_json(path, payload)
    return payload


def _rms(x):
    return x.detach().float().square().mean(-1).sqrt().cpu().numpy()


@torch.inference_mode()
def collect(model, name, data, cfg, device, location: Path, output: Path):
    if not hasattr(getattr(model, "encoder", None), "blocks") or name in {"tactical_multinomial", "tactical_mixture"}:
        return
    model.eval()
    selected = selection(data, cfg, output)
    folder = location / "diagnostics"
    folder.mkdir(exist_ok=True)
    encoder = model.encoder
    k, state, handles = encoder.k, {}, []
    def block_pre(layer, module, args):
        x, evidence = args
        state[layer] = {"carry": _rms(x[0, -1]), "input_rep": x[0, -1].mean(0).float().cpu().numpy(),
                        "evidence": evidence[0, -1].float().cpu().numpy(), "carry_tensor": x[0, -1].detach()}
        gate_active = module.structured and not getattr(module,"ablation_neutral_gate",False)
        gate = .5 + module.gate(evidence[0, -1]).sigmoid() if gate_active else torch.ones_like(x[0, -1])
        state[layer]["gate_tensor"] = gate
        state[layer]["gate"] = gate.float().mean(-1).cpu().numpy()
    def spatial_hook(layer, module, args, kwargs, result):
        block = encoder.blocks[layer]
        state[layer]["spatial_norm"] = _rms(result[-1]*block.spatial_scale.tanh())
        state[layer]["after_spatial"] = state[layer]["carry_tensor"] + result[-1]*block.spatial_scale.tanh()
        if not state.get("case", False):
            return
        x = args[0][-1:]
        q, kk, _ = module.qkv(x).reshape(1, k, 3, module.heads, module.width).permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q.float() @ kk.float().transpose(-2, -1)) / np.sqrt(module.width)
        bias = kwargs.get("bias")
        scores = logits if bias is None else logits + bias.float()
        state[layer]["spatial_attention"] = scores.softmax(-1)[0].cpu().numpy()
        state[layer]["spatial_no_bias"] = logits.softmax(-1)[0].cpu().numpy()
        state[layer]["distance_bias"] = np.zeros((module.heads, k, k), np.float32) if bias is None else bias[0].float().cpu().numpy()
    def temporal_hook(layer, module, args, result):
        block = encoder.blocks[layer]
        state[layer]["temporal_norm"] = _rms(result[:, -1]*block.temporal_scale.tanh()*state[layer]["gate_tensor"])
        state[layer]["after_temporal"] = state[layer]["after_spatial"] + result[:, -1]*block.temporal_scale.tanh()*state[layer]["gate_tensor"]
        if not state.get("case", False):
            return
        x = args[0]
        length = x.shape[1]
        qkv = module.qkv(x).reshape(k, length, 3, module.heads, module.width).permute(2, 0, 3, 1, 4)
        q, keys = qkv[0, :, :, -1:, :].float(), qkv[1].float()
        weights = ((q @ keys.transpose(-2, -1))/np.sqrt(module.width)).softmax(-1)
        state[layer]["temporal_attention"] = weights[:, :, 0].permute(1, 0, 2).cpu().numpy()
    def ffn_hook(layer, module, args, result):
        state[layer]["ffn_norm"] = _rms(result[0, -1]*encoder.blocks[layer].ffn_scale.tanh())
    def block_hook(layer, module, args, result):
        state[layer]["representation"] = result[0, -1].mean(0).float().cpu().numpy()
    for layer, block in enumerate(encoder.blocks):
        handles.extend([
            block.register_forward_pre_hook(lambda m, a, l=layer: block_pre(l, m, a)),
            block.spatial.register_forward_hook(lambda m, a, kw, o, l=layer: spatial_hook(l, m, a, kw, o), with_kwargs=True),
            block.temporal.register_forward_hook(lambda m, a, o, l=layer: temporal_hook(l, m, a, o)),
            block.ffn.register_forward_hook(lambda m, a, o, l=layer: ffn_hook(l, m, a, o)),
            block.register_forward_hook(lambda m, a, o, l=layer: block_hook(l, m, a, o))])
    embeddings, reps, gates, gate_channels, residuals, incoming, evidence, nlls, cases = [], [], [], [], [], [], [], [], {}
    try:
        for j, row in enumerate(selected["rows"]):
            i = int(row["index"])
            prefix = data["frames"][data["start"][i]:data["end"][i]]
            x = torch.as_tensor(prefix[None], device=device)
            n, y = torch.as_tensor(data["n"][i:i+1], device=device).long(), torch.as_tensor(data["y"][i:i+1], device=device).long()
            state.clear()
            state["case"] = j in selected["case_rows"]
            ctx = model.encode_context(x) if hasattr(model,"encode_context") else encoder(x)
            pooled = ctx[0] if isinstance(ctx,tuple) else ctx
            embeddings.append(pooled[0].float().cpu().numpy())
            if hasattr(model,"capture_readout"): model.capture_readout = state["case"]
            nlls.append(float(model.decode(ctx, n, y)[1][0]))
            reps.append(np.stack([state[0]["input_rep"]] + [state[l]["representation"] for l in range(len(encoder.blocks))]))
            gates.append(np.stack([state[l]["gate"] for l in range(len(encoder.blocks))]))
            gate_channels.append(np.stack([state[l]["gate_tensor"].float().cpu().numpy() for l in range(len(encoder.blocks))]))
            residuals.append(np.stack([np.stack([state[l][key] for key in ("carry", "spatial_norm", "temporal_norm", "ffn_norm")], -1) for l in range(len(encoder.blocks))]))
            incoming.append(np.stack([np.stack([state[l]["carry"], _rms(state[l]["after_spatial"]), _rms(state[l]["after_temporal"])], -1) for l in range(len(encoder.blocks))]))
            evidence.append(state[0]["evidence"])
            if state["case"]:
                for key in ("spatial_attention", "spatial_no_bias", "distance_bias", "temporal_attention"):
                    cases[f"case{j}_{key}"] = np.stack([state[l][key] for l in range(len(encoder.blocks))])
                cases[f"case{j}_prefix"] = prefix
                cases[f"case{j}_time"] = prefix[:, 2*k+4]*120
                cases[f"case{j}_truth"] = data["y"][i]
                if getattr(model,"capture_readout",False):
                    for key,value in model.readout_trace.items(): cases[f"case{j}_readout_{key}"] = value[0].float().cpu().numpy()
        np.savez_compressed(folder / "traces.npz", embeddings=np.stack(embeddings), representations=np.stack(reps), gates=np.stack(gates), gate_channels=np.stack(gate_channels),
                            residual_norms=np.stack(residuals), incoming_norms=np.stack(incoming), evidence=np.stack(evidence), exact_nll=np.asarray(nlls))
        np.savez_compressed(folder / "cases.npz", **cases)
    finally:
        if hasattr(model,"capture_readout"): model.capture_readout = False
        for handle in handles:
            handle.remove()
    if name in {"tactical_transformer","improved_transformer","population_transformer"}:
        interventions = []
        try:
            probe_rows = np.linspace(0, len(selected["rows"])-1, min(cfg.get("diagnostic_intervention_samples", 128), len(selected["rows"]))).round().astype(int)
            for j in probe_rows:
                row = selected["rows"][j]
                i = int(row["index"])
                x = torch.as_tensor(data["frames"][data["start"][i]:data["end"][i]][None], device=device)
                n, y = torch.as_tensor(data["n"][i:i+1], device=device).long(), torch.as_tensor(data["y"][i:i+1], device=device).long()
                values = dict(row, baseline_nll=nlls[j])
                for flag, key in (("diagnostic_neutral_gate", "neutral_gate_nll"), ("diagnostic_no_distance", "no_distance_nll")):
                    for block in encoder.blocks:
                        setattr(block, flag, True)
                    values[key] = float(model.nll(x, y, n)[0])
                    for block in encoder.blocks:
                        setattr(block, flag, False)
                if hasattr(model,"regional_readout"):
                    model.diagnostic_disable_readout = True
                    values["no_readout_nll"] = float(model.nll(x,y,n)[0])
                    model.diagnostic_disable_readout = False
                interventions.append(values)
            pd.DataFrame(interventions).to_csv(folder / "interventions.csv", index=False)
        finally:
            if hasattr(model,"regional_readout"): model.diagnostic_disable_readout = False
            for block in encoder.blocks:
                block.diagnostic_neutral_gate = block.diagnostic_no_distance = False
    checkpoint = location / "best.pt"
    from .gradient_audit import audit
    audit(model, name, data, cfg, device, folder, int(location.parent.name.split("_")[-1]))
    save_json(folder / "trace_manifest.json", {"model": name, "samples": len(embeddings), "cases": selected["case_rows"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "trace_precision": "float32, eval, dropout disabled",
        "attention": "actual Q/K softmax; temporal last query only, all prefix keys; local no-bias weights keep Q/K fixed",
        "residual_norm_order": ["carry", "spatial_update", "gated_temporal_update", "ffn_update"],
        "evidence_order": ["friendly_smoke_count", "friendly_smoke_age", "radar_age", "radar_ever", "exposed_count", "CT_steps", "T_steps", "footstep_age", "footstep_ever", "audio_available", "CT_alive", "T_alive"],
        "interpretation": "Mechanism traces and frozen computational interventions, not causal game effects or proof from attention alone."})
    LOG.info("Saved reusable module diagnostics: %s (%d cases)", name, len(embeddings))
