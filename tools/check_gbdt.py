"""Fit only a 20-prefix synthetic CPU fixture and check HistGBDT count laws."""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations_with_replacement
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from cs2.gbdt import (PrefixSummary, _observation_hash, evaluate_gbdt, fit_gbdt,
                      legal_probabilities)


def fixture():
    rng = np.random.default_rng(2026)
    frames = rng.normal(0, .1, (32, 311)).astype(np.float32)
    frames[:, 29] = 1.
    starts, ends = [], []
    for start in [0, 8, 16, 24]:
        frames[start:start+8, 246] = np.arange(8) / 480.
        for end in range(start+3, start+8):
            starts.append(start)
            ends.append(end)
    n = np.arange(20, dtype=np.int64) % 5 + 1
    y = np.zeros((20, 12), dtype=np.int64)
    for row in range(20):
        for person in range(int(n[row])):
            region = (row + 3*person) % 12
            y[row, 11 if region == 10 else region] += 1
    # An unseen validation count tests the positive floor for absent classes.
    y[14] = 0
    y[14, 1] = 5
    return dict(frames=frames, start=np.asarray(starts), end=np.asarray(ends), n=n, y=y,
                split=np.asarray(["train"]*10+["val"]*5+["test"]*5),
                group=np.asarray([f"synthetic_match_{i//5}" for i in range(20)]))


def check(work):
    data = fixture()
    cfg = dict(gbdt_max_iter=2, gbdt_min_samples_leaf=1, gbdt_threads=2,
               gbdt_progress=False, gbdt_sampling_seed=2026)
    model, info = fit_gbdt(data, cfg, "cpu", work / "fit", 2026)
    assert info["train_prefixes"] == 10 and info["validation_prefixes"] == 5
    assert all(0 <= it <= 2 for it in info["selected_iterations"])
    assert any(row["constant_fallback"] for row in info["regions"])
    summary = PrefixSummary(data, _observation_hash(data))
    descriptors = summary.matrix(np.arange(20))
    for row in range(20):
        start, end = data["start"][row], data["end"][row]
        np.testing.assert_array_equal(descriptors[row, :311], data["frames"][end-1])
        np.testing.assert_allclose(descriptors[row, 311:], data["frames"][start:end].mean(0, dtype=np.float64),
                                   atol=1e-7, rtol=1e-6)
    # No frame occurring after a cutoff may alter that cutoff's descriptor.
    modified = dict(data, frames=data["frames"].copy())
    modified["frames"][data["end"][0]:] += 2.
    causal = PrefixSummary(modified, _observation_hash(modified)).matrix(np.array([0]))
    np.testing.assert_array_equal(causal, descriptors[:1])
    n = np.arange(6, dtype=np.int64)
    lookup = model.conditional_lookup(np.repeat(descriptors[:1], 6, axis=0), n)
    for row in range(6):
        for remaining in range(int(n[row])+1):
            probability = lookup[row, :, remaining]
            np.testing.assert_allclose(probability.sum(-1), 1., atol=1e-12)
            assert np.all(probability[:, remaining+1:] == 0.)
            assert np.all(probability[:, :remaining+1] > 0.)
    draws = model.sample_lookup(lookup, n, 32, np.random.default_rng(2026))
    assert np.all(draws >= 0) and np.array_equal(draws, draws.astype(int))
    np.testing.assert_array_equal(draws.sum(-1), np.repeat(n[:, None], 32, axis=1))
    # Enumerate the complete two-opponent support: its exact likelihood sums to one.
    truth = np.zeros((78, 12), dtype=np.int64)
    for row, pair in enumerate(combinations_with_replacement(range(12), 2)):
        np.add.at(truth[row], np.asarray(pair), 1)
    two_n = np.full(78, 2, dtype=np.int64)
    two_lookup = model.conditional_lookup(np.repeat(descriptors[:1], 78, axis=0), two_n)
    exact_nll = model.nll_lookup(two_lookup, truth, two_n)
    assert np.isfinite(exact_nll).all()
    np.testing.assert_allclose(np.exp(-exact_nll).sum(), 1., atol=1e-10)
    floor_case = legal_probabilities(np.ones((2, 1)), np.array([2]), np.array([0, 1]), 1e-6)
    np.testing.assert_array_equal(floor_case[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(floor_case[1], [.5, .5, 0, 0, 0, 0])
    test = np.flatnonzero(data["split"] == "test")
    scores, predictions = evaluate_gbdt(model, data, test, samples=8, batch_size=2, cfg=cfg)
    assert predictions.shape == (5, 8, 12)
    assert all(values.shape == (5,) and np.isfinite(values).all() for values in scores.values())
    alternate = dict(data, y=data["y"].copy())
    alternate["y"][test] = alternate["y"][test][:, ::-1]
    _, other_predictions = evaluate_gbdt(model, alternate, test, samples=8, batch_size=2, cfg=cfg)
    np.testing.assert_array_equal(predictions, other_predictions)
    reloaded = joblib.load(work / "fit/best_gbdt.joblib")
    _, loaded_predictions = evaluate_gbdt(reloaded, data, test, samples=8, batch_size=2, cfg=cfg)
    np.testing.assert_array_equal(predictions, loaded_predictions)
    _, resumed = fit_gbdt(alternate, cfg, "cpu", work / "fit", 2026)
    assert resumed["resumed_complete"] is True  # Changed test labels do not enter the fit contract.
    recovery = work / "checkpoint_only"
    recovery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "fit/best_gbdt.joblib", recovery / "best_gbdt.joblib")
    _, recovered = fit_gbdt(data, cfg, "cpu", recovery, 2026)
    assert recovered["resumed_complete"] is True
    for file in ["fit.json", "history.json", "feature_manifest.json", "training_state.json", "fit_progress.json"]:
        assert (recovery/file).is_file()
    partial = work / "partial"
    (partial / "regions").mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "fit/fit_contract.json", partial / "fit_contract.json")
    shutil.copy2(work / "fit/regions/region_00.joblib", partial / "regions/region_00.joblib")
    partial_model, partial_info = fit_gbdt(data, cfg, "cpu", partial, 2026)
    assert partial_info["regions"][0] == info["regions"][0]
    _, partial_predictions = evaluate_gbdt(partial_model, data, test, samples=8, batch_size=2, cfg=cfg)
    np.testing.assert_array_equal(predictions, partial_predictions)
    try:
        fit_gbdt(data, dict(cfg, gbdt_max_iter=3), "cpu", work / "fit", 2026)
    except RuntimeError as exc:
        assert "contract" in str(exc)
    else:
        raise AssertionError("Changed fit contract was accepted")
    return dict(status="PASS", fixture="20 synthetic prefixes / 32 shared frames", device="cpu",
        formal_training_performed=False, tiny_fixture_fit=True, max_iter=2,
        train_prefixes=10, validation_prefixes=5, test_prefixes=5,
        descriptor_features=624, regional_classifiers=11,
        selected_iterations=info["selected_iterations"], fitted_trees=info["fitted_trees"],
        sklearn_version=info["sklearn_version"], checkpoint=str(work / "fit/best_gbdt.joblib"),
        checks=["complete_prefix_mean_and_last", "no_future_observations", "legal_known_populations_0_to_5",
                "positive_absent_class_floor", "zero_remaining_determinism", "constant_class_fallback",
                "normalized_exact_joint_likelihood_over_78_states", "finite_sample_scores_and_nll",
                "no_test_target_leakage", "portable_joblib_checkpoint", "completed_fit_resume",
                "checkpoint_only_metadata_recovery",
                "partial_region_resume", "changed_contract_rejected"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tools/check_gbdt.json")
    parser.add_argument("--workdir", type=Path,
                        default=ROOT / "outputs" / ("_gbdt_smoke_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report = check(args.workdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
