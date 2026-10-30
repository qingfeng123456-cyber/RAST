from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import torch


def sample_scores(samples: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Proper sample-based scores on integer count vectors; smaller is better."""
    if samples.ndim != 3 or samples.shape[1] < 2 or samples.shape[1] % 2:
        raise ValueError("Need an even number >=2 of predictive samples")
    if not np.array_equal(samples.sum(-1), np.broadcast_to(truth.sum(-1)[:, None], samples.shape[:2])):
        raise ValueError("Predictive samples violate population conservation")
    if not np.all(samples >= 0) or not np.all(samples == samples.astype(int)):
        raise ValueError("Occupancy samples must be nonnegative integers")
    half = samples.shape[1] // 2
    energy = np.linalg.norm(samples - truth[:, None], axis=-1).mean(-1)
    energy -= .5 * np.linalg.norm(samples[:, :half] - samples[:, half:], axis=-1).mean(-1)
    probability = np.stack([(samples == count).mean(1) for count in range(6)], -1)
    target = np.eye(6)[truth.astype(int)]
    count_brier = np.square(probability - target).sum(-1).mean(-1)
    presence = samples > 0
    presence_prob = presence.mean(1)
    presence_brier = np.square(presence_prob - (truth > 0)).mean(-1)
    pairs = list(itertools.combinations(range(truth.shape[-1]), 2))
    joint_prob = np.stack([(presence[:, :, a] & presence[:, :, b]).mean(1) for a, b in pairs], -1)
    joint_truth = np.stack([(truth[:, a] > 0) & (truth[:, b] > 0) for a, b in pairs], -1)
    return {"energy_score": energy, "count_brier": count_brier,
            "presence_brier": presence_brier,
            "pair_presence_brier": np.square(joint_prob - joint_truth).mean(-1),
            "count_mae": np.abs(samples.mean(1) - truth).mean(-1)}


@torch.inference_mode()
def evaluate(model, data, indices, device, samples=128, batch_size=8, cfg=None, desc="Test predictions"):
    from torch.utils.data import DataLoader
    from .prefix import PrefixDataset, collate_prefixes
    from .training import amp_context
    from functools import partial
    from tqdm import tqdm
    import sys
    model.eval()
    predictions, nlls, score_parts = [], [], []
    pad_multiple = (cfg or {}).get("pad_multiple",32)
    longest = int((data["end"][indices]-data["start"][indices]).max())
    padded = int(np.ceil(longest/pad_multiple)*pad_multiple)
    if hasattr(model,"prefix_capacity"):
        batch_size = min(batch_size,max(1,model.prefix_capacity//padded))
    loader = DataLoader(PrefixDataset(data, indices), batch_size=batch_size, collate_fn=partial(collate_prefixes,pad_multiple=pad_multiple))
    for xb, yb, nb in tqdm(loader,desc=desc,dynamic_ncols=True,ascii=True,file=sys.stdout):
        xb, yb, nb = xb.to(device), yb.to(device), nb.to(device)
        with amp_context(model, cfg or {}, device):
            if model.exact_nll and hasattr(model,"infer"):
                generated, likelihood = model.infer(xb,yb,nb,samples)
            else:
                generated = model.sample(xb,nb,samples)
                likelihood = model.nll(xb,yb,nb) if model.exact_nll else None
            draws = generated.float().cpu().numpy()
        score_parts.append(sample_scores(draws, yb.cpu().numpy()))
        predictions.append(draws)
        if likelihood is not None:
            nlls.append(likelihood.float().cpu().numpy())
    scores = {key: np.concatenate([s[key] for s in score_parts]) for key in score_parts[0]}
    if nlls:
        scores["exact_count_nll"] = np.concatenate(nlls)
    return scores, np.concatenate(predictions)


def bootstrap_summary(frame: pd.DataFrame, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match-cluster intervals, conditional on the selected seed runs."""
    metrics = ["energy_score", "count_brier", "presence_brier", "pair_presence_brier", "count_mae", "exact_count_nll"]
    available = [m for m in metrics if m in frame]
    keys = ["model", "group"] + (["seed"] if "seed" in frame else [])
    group = frame.groupby(keys, as_index=False)[available].mean()
    group = group.groupby(["model", "group"], as_index=False)[available].mean()
    rng, rows = np.random.default_rng(1729), []
    for model, sub in group.groupby("model"):
        for metric in metrics:
            if metric not in sub or sub[metric].isna().all():
                continue
            values = sub[metric].dropna().to_numpy()
            means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
            lo, hi = np.quantile(means, [.025, .975])
            rows.append({"model": model, "metric": metric, "match_macro_mean": values.mean(),
                         "ci_low": lo, "ci_high": hi, "test_matches": len(values)})
    paired = []
    for candidate in ("population_transformer", "improved_transformer", "tactical_transformer", "diffusion", "spatial_diffusion"):
        for baseline in group.model.unique():
            if candidate == baseline:
                continue
            for metric in ("energy_score", "count_brier", "pair_presence_brier", "exact_count_nll"):
                if metric not in group:
                    continue
                wide = group.pivot(index="group", columns="model", values=metric)
                if candidate not in wide or baseline not in wide:
                    continue
                delta = (wide[candidate] - wide[baseline]).dropna().to_numpy()
                if not len(delta):
                    continue  # e.g. KNN has no evaluable exact likelihood.
                means = delta[rng.integers(len(delta), size=(repeats, len(delta)))].mean(1)
                lo, hi = np.quantile(means, [.025, .975])
                paired.append({"comparison": candidate + " - " + baseline, "metric": metric,
                               "delta": delta.mean(), "ci_low": lo, "ci_high": hi,
                               "negative_favors_candidate": True})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def seed_summaries(frame: pd.DataFrame, expected_seeds=None):
    """Equal-match scores per run, then mean and sample SD across seed runs.

    Seed SD includes predictive Monte Carlo variation; non-neural baselines have
    no training randomness. It is distinct from a match-bootstrap interval.
    """
    metrics = [m for m in ("energy_score", "count_brier", "presence_brier",
               "pair_presence_brier", "count_mae", "exact_count_nll") if m in frame]
    per_match = frame.groupby(["model", "seed", "group"], as_index=False)[metrics].mean()
    rows, aggregate = [], []
    for model, subset in per_match.groupby("model"):
        actual = sorted(int(s) for s in subset.seed.unique())
        if expected_seeds is not None and actual != sorted(expected_seeds):
            raise ValueError(f"Incomplete seed runs for {model}: {actual}, expected {expected_seeds}")
        for metric in metrics:
            valid = subset.dropna(subset=[metric])
            if valid.empty:
                continue  # KNN has no exact joint likelihood.
            cohorts = [set(part.group) for _, part in valid.groupby("seed")]
            if valid.seed.nunique() != len(actual) or any(c != cohorts[0] for c in cohorts[1:]):
                raise ValueError(f"Different seed match coverage for {model}/{metric}")
            values = valid.groupby("seed")[metric].mean()
            for seed, value in values.items():
                rows.append(dict(model=model, seed=int(seed), metric=metric,
                                 match_macro_mean=float(value), test_matches=len(cohorts[0])))
            aggregate.append(dict(model=model, metric=metric, seed_mean=float(values.mean()),
                                  seed_std=float(values.std(ddof=1)), seed_count=len(values),
                                  seeds=",".join(str(int(s)) for s in values.index)))
    return pd.DataFrame(rows), pd.DataFrame(aggregate)


def calibration_table(samples, truth):
    probabilities, targets = (samples > 0).mean(1).ravel(), (truth > 0).ravel()
    bins = np.minimum((probabilities * 10).astype(int), 9)
    return [{"bin": i, "count": int((bins == i).sum()),
             "predicted": float(probabilities[bins == i].mean()),
             "observed": float(targets[bins == i].mean())}
            for i in range(10) if np.any(bins == i)]
