"""Tiny CPU checks for new full-history baselines; performs no model training."""
from __future__ import annotations

import argparse
from io import BytesIO
import importlib
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from cs2.models import MODEL_CLASSES


def config():
    return dict(hidden_dim=192, transformer_dim=192, transformer_heads=8,
                transformer_layers=4, dropout=.1, conditional_rps_weight=.25,
                rnn_layers=3, lstm_hidden_dim=320, gru_hidden_dim=384,
                rnn_time_encoding=True,
                region_centers=[[float(i % 4), float(i // 4), 0.] for i in range(12)])


def fixture(steps=5, batch=2):
    k = 12
    x = torch.randn(batch, steps, 24*k+23) * .05
    x[:, :, 2*k+5] = 1.
    x[:, :, 2*k+6+18*k] = torch.arange(steps)[None] / 480.
    if batch > 1 and steps == 5:
        x[1, 3:] = 0.
    n = torch.tensor([1, 5][:batch], dtype=torch.long)
    y = torch.zeros(batch, k, dtype=torch.long)
    y[0, 2] = 1
    if batch > 1:
        y[1, 1], y[1, 7] = 2, 3
    return x, y, n


def legal(draws, n):
    assert draws.ndim == 3 and draws.shape[0] == len(n) and draws.shape[-1] == 12
    assert bool(torch.isfinite(draws).all())
    assert bool((draws >= 0).all()) and bool((draws == draws.round()).all())
    assert torch.equal(draws.sum(-1).long(), n[:, None].expand(draws.shape[:2]))


def check_new(name, cfg, x, y, n, long_prefix):
    torch.manual_seed(2026)
    model = MODEL_CLASSES[name](311, 12, cfg)
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
                  supports_amp=bool(model.supports_amp), exact_nll=bool(model.exact_nll),
                  rps_weight=model.rps_weight)
    model.train()
    loss = model.loss(x, y, n)
    assert loss.ndim == 0 and bool(torch.isfinite(loss))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        ctx = model.encoder(x)
        assert ctx.shape == (2, 192)
        target, nll, rps = model.decode(ctx, n, y, return_aux=True)
        assert torch.equal(target, y)
        assert nll.shape == (2,) and bool(torch.isfinite(nll).all())
        torch.testing.assert_close(model.loss(x, y, n), (nll + model.rps_weight*rps).mean())
        torch.testing.assert_close(model.nll(x, y, n), nll)
        draws, infer_nll = model.infer(x, y, n, 16)
        legal(draws, n)
        torch.testing.assert_close(infer_nll, nll)
        legal(model.sample(x, n, 16), n)
        # Include the zero-population boundary as well as every dataset value.
        support_n = torch.arange(6, dtype=torch.long)
        support_x = x[:1].expand(6, -1, -1).contiguous()
        legal(model.sample(support_x, support_n, 8), support_n)
        padded = torch.cat([x, torch.zeros(len(x), 3, x.shape[-1])], 1)
        torch.testing.assert_close(model.nll(padded, y, n), nll, atol=1e-6, rtol=1e-6)
        # Nonzero unavailable padding must also be ignored by validity flags.
        dirty = padded.clone()
        dirty[:, 5:] = torch.randn_like(dirty[:, 5:])
        dirty[:, 5:, 29] = 0.
        torch.testing.assert_close(model.nll(dirty, y, n), nll, atol=1e-6, rtol=1e-6)
        alternate = y.roll(1, -1)
        torch.manual_seed(731)
        a = model.infer(x, y, n, 16)[0]
        torch.manual_seed(731)
        b = model.infer(x, alternate, n, 16)[0]
        assert torch.equal(a, b), "Sampled draws depend on supplied evaluation targets"
        state = BytesIO()
        torch.save(model.state_dict(), state)
        state.seek(0)
        restored = MODEL_CLASSES[name](311, 12, cfg).eval()
        restored.load_state_dict(torch.load(state, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored.nll(x, y, n), nll, atol=0, rtol=0)
        if long_prefix and name in {"lstm_count", "gru_count"}:
            long_x, long_y, long_n = fixture(1665, 1)
            start = time.perf_counter()
            values = model.nll(long_x, long_y, long_n)
            assert values.shape == (1,) and bool(torch.isfinite(values).all())
            result["long_prefix"] = dict(steps=1665, seconds=time.perf_counter()-start,
                                         finite_nll=True, device="cpu")
    result["checks"] = ["finite_loss", "backward_finite_all_parameters", "context_shape",
                        "exact_joint_nll", "objective_identity", "legal_free_running_draws",
                        "legal_known_populations_0_to_5",
                        "right_padding_invariance", "flagged_padding_invariance",
                        "evaluation_target_independence", "state_serialization"]
    result["status"] = "PASS"
    return result


def check_retained(reference, cfg, x, y, n):
    if not (reference / "models.py").is_file():
        return dict(status="SKIPPED", reason="Optional untouched reference package not supplied")
    package = types.ModuleType("reference_cs2")
    package.__path__ = [str(reference)]
    sys.modules["reference_cs2"] = package
    original = importlib.import_module("reference_cs2.models")
    results = {}
    for name in ["vanilla_transformer", "improved_transformer"]:
        torch.manual_seed(2026)
        current = MODEL_CLASSES[name](311, 12, cfg).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2026)
        baseline = original.MODEL_CLASSES[name](311, 12, cfg).eval()
        assert torch.equal(rng, torch.get_rng_state())
        assert current.state_dict().keys() == baseline.state_dict().keys()
        assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in current.state_dict().items())
        with torch.no_grad():
            torch.testing.assert_close(current.loss(x, y, n), baseline.loss(x, y, n), atol=0, rtol=0)
            torch.manual_seed(93)
            a, an = current.infer(x, y, n, 16)
            torch.manual_seed(93)
            b, bn = baseline.infer(x, y, n, 16)
            assert torch.equal(a, b) and torch.equal(an, bn)
        results[name] = dict(status="PASS", seeded_parameters_identical=True,
                             construction_rng_identical=True, loss_and_infer_identical=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "github_release/cs2")
    parser.add_argument("--output", type=Path, default=ROOT / "tools/check_recurrent_models.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    cfg = config()
    x, y, n = fixture()
    report = dict(status="PASS", device="cpu", torch_version=torch.__version__,
                  training_performed=False, config=cfg, models={})
    for name in ["lstm_count", "gru_count"]:
        report["models"][name] = check_new(name, cfg, x, y, n, not args.skip_long)
    report["retained_model_compatibility"] = check_retained(args.reference, cfg, x, y, n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
