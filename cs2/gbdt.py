"""Histogram gradient-boosted trees for legal anonymous occupancy counts.

The baseline uses full-prefix mean and last-frame features. Eleven conditional
classifiers allocate regional counts, with the last region receiving the exact
remaining population. All tree operations use public scikit-learn APIs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .evaluation import sample_scores
from .io import save_json

FEATURE_VERSION = "last_plus_complete_prefix_mean_known_and_remaining_v1"
OFFICIAL_API = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"
_PREFIX_CACHE = None  # At most one float64 cumulative frame array is retained.


def _array_hash(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    if array.dtype.hasobject:
        digest.update(json.dumps(array.tolist(), sort_keys=True, default=str).encode())
        return digest.hexdigest()
    values = array.reshape(1) if array.ndim == 0 else array
    for start in range(0, len(values), 4096):
        digest.update(np.ascontiguousarray(values[start:start+4096]).view(np.uint8))
    return digest.hexdigest()


def _observation_hash(data):
    return {name: _array_hash(data[name]) for name in ("frames", "start", "end")}


class PrefixSummary:
    """One shared float64 cumulative sum; descriptors are materialized in chunks."""

    def __init__(self, data, fingerprints):
        self.frames = np.asarray(data["frames"])
        if self.frames.ndim != 2 or self.frames.shape[1] != 311:
            raise ValueError("HistGBDT expects the frozen 311-feature frame schema")
        self.start = np.asarray(data["start"], dtype=np.int64)
        self.end = np.asarray(data["end"], dtype=np.int64)
        if np.any(self.start < 0) or np.any(self.end <= self.start) or np.any(self.end > len(self.frames)):
            raise ValueError("Invalid complete-prefix start/end offsets")
        self.fingerprints = fingerprints
        self.cumulative = np.empty((len(self.frames)+1, self.frames.shape[1]), dtype=np.float64)
        self.cumulative[0] = 0.
        np.cumsum(self.frames, axis=0, dtype=np.float64, out=self.cumulative[1:])

    def matrix(self, indices, known_n=None, chunk_size=4096):
        indices = np.asarray(indices, dtype=np.int64)
        width = 2 * self.frames.shape[1]
        output = np.empty((len(indices), width + (2 if known_n is not None else 0)), dtype=np.float32)
        for offset in range(0, len(indices), chunk_size):
            rows = indices[offset:offset+chunk_size]
            start, end = self.start[rows], self.end[rows]
            mean = self.cumulative[end] - self.cumulative[start]
            mean /= (end-start)[:, None]
            output[offset:offset+len(rows), :self.frames.shape[1]] = self.frames[end-1]
            output[offset:offset+len(rows), self.frames.shape[1]:width] = mean
        if known_n is not None:
            output[:, width] = np.asarray(known_n, dtype=np.float32) / 5.
            output[:, width+1] = 0.
        return output


def _summary(data, fingerprints=None):
    global _PREFIX_CACHE
    fingerprints = fingerprints or _observation_hash(data)
    if _PREFIX_CACHE is None or _PREFIX_CACHE.fingerprints != fingerprints:
        _PREFIX_CACHE = None  # Release an old cumulative array before replacing it.
        _PREFIX_CACHE = PrefixSummary(data, fingerprints)
    else:
        # Bind last-frame/offset reads to this fingerprint-verified input, even
        # when an earlier caller supplied an equal-valued, later-mutated array.
        _PREFIX_CACHE.frames = np.asarray(data["frames"])
        _PREFIX_CACHE.start = np.asarray(data["start"], dtype=np.int64)
        _PREFIX_CACHE.end = np.asarray(data["end"], dtype=np.int64)
    return _PREFIX_CACHE


def _settings(cfg):
    settings = dict(max_iter=int(cfg.get("gbdt_max_iter", 200)),
                    learning_rate=float(cfg.get("gbdt_learning_rate", .05)),
                    max_leaf_nodes=int(cfg.get("gbdt_max_leaf_nodes", 31)),
                    min_samples_leaf=int(cfg.get("gbdt_min_samples_leaf", 32)),
                    max_bins=int(cfg.get("gbdt_max_bins", 63)),
                    l2_regularization=float(cfg.get("gbdt_l2_regularization", 1.)),
                    threads=int(cfg.get("gbdt_threads", 4)),
                    probability_floor=float(cfg.get("gbdt_probability_floor", 1e-6)))
    if settings["max_iter"] < 1 or settings["threads"] < 1:
        raise ValueError("HistGBDT iterations and CPU threads must be positive")
    if not 0 < settings["probability_floor"] < 1/6:
        raise ValueError("HistGBDT probability floor must be between zero and 1/6")
    return settings


def _classifier(settings, seed, iterations):
    return HistGradientBoostingClassifier(loss="log_loss", learning_rate=settings["learning_rate"],
        max_iter=iterations, max_leaf_nodes=settings["max_leaf_nodes"],
        min_samples_leaf=settings["min_samples_leaf"], max_bins=settings["max_bins"],
        l2_regularization=settings["l2_regularization"], categorical_features=None,
        early_stopping=False, random_state=seed)


class ConstantCountClassifier:
    """Portable fallback when a conditional training target has one class."""

    def __init__(self, value):
        self.classes_ = np.asarray([value], dtype=np.int64)
        self.n_iter_ = 0
        self.n_trees_per_iteration_ = 0

    def predict_proba(self, x):
        return np.ones((len(x), 1), dtype=np.float64)


def legal_probabilities(raw, classes, remaining, floor):
    """Expand absent classes, floor legal mass, mask and renormalize over 0..R."""
    raw = np.asarray(raw, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)
    remaining = np.asarray(remaining, dtype=np.int64)
    if raw.shape != (len(remaining), len(classes)) or not np.isfinite(raw).all():
        raise ValueError("Invalid classifier probability output")
    if np.any(classes < 0) or np.any(classes > 5) or np.any(remaining < 0) or np.any(remaining > 5):
        raise ValueError("Occupancy count classes and remaining population must lie in 0..5")
    probability = np.full((len(remaining), 6), floor, dtype=np.float64)
    probability[:, classes] = np.maximum(raw, floor)
    probability *= np.arange(6)[None] <= remaining[:, None]
    probability /= probability.sum(-1, keepdims=True)
    return probability


def _conditional_nll(probability, target, groups=None):
    values = -np.log(probability[np.arange(len(target)), target])
    if groups is None:
        return float(values.mean())
    _, inverse = np.unique(groups, return_inverse=True)
    return float((np.bincount(inverse, weights=values) / np.bincount(inverse)).mean())


class HistGBDTCountModel:
    exact_nll = True
    supports_amp = False

    def __init__(self, estimators, k, settings, seed, contract, fit_info=None):
        self.estimators, self.k = estimators, int(k)
        self.settings, self.seed, self.contract = settings, int(seed), contract
        self.fit_info = fit_info or {}
        self.feature_version = FEATURE_VERSION

    def eval(self):
        return self

    def conditional_lookup(self, descriptors, n):
        """One tree call per region for all feasible remaining-count states."""
        descriptors = np.asarray(descriptors, dtype=np.float32)
        n = np.asarray(n, dtype=np.int64)
        if descriptors.ndim != 2 or descriptors.shape[1] != 622 or len(descriptors) != len(n):
            raise ValueError("Expected 622-dimensional complete-prefix descriptors")
        if np.any(n < 0) or np.any(n > 5):
            raise ValueError("Known hidden population must lie in 0..5")
        lookup = np.zeros((len(n), self.k-1, 6, 6), dtype=np.float64)
        lookup[:, :, 0, 0] = 1.
        rows, choices = np.where(np.arange(1, 6)[None] <= n[:, None])
        remaining = choices + 1
        if not len(rows):
            return lookup
        design = np.empty((len(rows), 624), dtype=np.float32)
        design[:, :622] = descriptors[rows]
        design[:, 622] = n[rows] / 5.
        design[:, 623] = remaining / 5.
        with threadpool_limits(limits=self.settings["threads"]):
            for region, estimator in enumerate(self.estimators):
                probability = legal_probabilities(estimator.predict_proba(design), estimator.classes_,
                                                   remaining, self.settings["probability_floor"])
                lookup[rows, region, remaining] = probability
        return lookup

    def sample_lookup(self, lookup, n, samples, rng):
        n = np.asarray(n, dtype=np.int64)
        remaining = np.broadcast_to(n[:, None], (len(n), samples)).copy()
        output = np.zeros((len(n), samples, self.k), dtype=np.float32)
        rows = np.arange(len(n))[:, None]
        for region in range(self.k-1):
            probability = lookup[rows, region, remaining]
            value = (rng.random(remaining.shape)[..., None] > probability.cumsum(-1)).sum(-1)
            value = np.minimum(value, remaining)  # Guard only final-CDF floating rounding.
            output[:, :, region] = value
            remaining -= value
        output[:, :, -1] = remaining
        return output

    def nll_lookup(self, lookup, truth, n):
        truth = np.asarray(truth)
        n = np.asarray(n, dtype=np.int64)
        if truth.shape != (len(n), self.k) or np.any(truth < 0) or np.any(truth != truth.astype(int)):
            raise ValueError("Invalid integer occupancy target")
        truth = truth.astype(np.int64)
        if not np.array_equal(truth.sum(-1), n):
            raise ValueError("Target occupancy does not conserve known population")
        remaining, nll = n.copy(), np.zeros(len(n), dtype=np.float64)
        rows = np.arange(len(n))
        for region in range(self.k-1):
            nll -= np.log(lookup[rows, region, remaining, truth[:, region]])
            remaining -= truth[:, region]
        return nll


def _dump_joblib(path, value):
    temporary = path.with_name(path.name+".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def fit_gbdt(data, cfg, device, output, seed):
    """Fit train-only trees; select each iteration count on the fixed val split."""
    del device  # HistGradientBoostingClassifier is always a CPU estimator.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings(cfg)
    train, val = [np.flatnonzero(np.asarray(data["split"]) == split) for split in ("train", "val")]
    if not len(train) or not len(val):
        raise ValueError("HistGBDT requires nonempty fixed training and validation splits")
    fingerprints = _observation_hash(data)
    k = int(np.asarray(data["y"]).shape[-1])
    contract = dict(feature_version=FEATURE_VERSION, settings=settings, seed=int(seed), k=k,
        sklearn_version=sklearn.__version__, observations=fingerprints,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_indices=_array_hash(train), val_indices=_array_hash(val),
        train_targets=_array_hash(data["y"][train]), val_targets=_array_hash(data["y"][val]),
        train_population=_array_hash(data["n"][train]), val_population=_array_hash(data["n"][val]),
        train_groups=_array_hash(data["group"][train]), val_groups=_array_hash(data["group"][val]),
        selection_aggregation="cutoff_to_match_then_equal_match_mean")
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract_path = output / "fit_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8"))["sha256"] != contract_hash:
        raise RuntimeError("HistGBDT checkpoint contract differs; use a fresh output directory")
    save_json(contract_path, dict(sha256=contract_hash, contract=contract))
    save_json(output / "feature_manifest.json", dict(version=FEATURE_VERSION, features=624,
        order=[dict(name="last_observed_frame", columns=[0, 310]),
               dict(name="complete_prefix_mean", columns=[311, 621]),
               dict(name="known_hidden_population_divided_by_5", columns=[622, 622]),
               dict(name="remaining_population_divided_by_5", columns=[623, 623])],
        prefix_storage="Shared float64 cumulative sum; complete start:end offsets; no cropping",
        observations=fingerprints, training_prefixes=len(train), validation_prefixes=len(val),
        training_matches=len(np.unique(np.asarray(data["group"])[train])),
        validation_matches=len(np.unique(np.asarray(data["group"])[val])),
        selection_aggregation="cutoff_to_match_then_equal_match_mean",
        test_targets_used=False, algorithm="sklearn.ensemble.HistGradientBoostingClassifier",
        official_api=OFFICIAL_API))
    checkpoint = output / "best_gbdt.joblib"
    if checkpoint.exists():
        model = joblib.load(checkpoint)
        if model.contract != contract:
            raise RuntimeError("Saved HistGBDT model has a different fit contract")
        # Recover a run interrupted between final checkpoint and JSON writes.
        save_json(output / "fit.json", model.fit_info)
        save_json(output / "history.json", [dict(region=row["region"], **stage)
            for row in model.fit_info["regions"] for stage in row["stages"]])
        save_json(output / "training_state.json", dict(contract_hash=contract_hash, completed_regions=k-1, total_regions=k-1))
        save_json(output / "fit_progress.json", dict(model="hist_gbdt", seed=int(seed),
            phase="complete", completed_regions=k-1, total_regions=k-1,
            validation_exact_count_nll=model.fit_info["validation_exact_count_nll"]))
        return model, dict(model.fit_info, resumed_complete=True)
    summary = _summary(data, fingerprints)
    xtrain = summary.matrix(train, np.asarray(data["n"])[train])
    xval = summary.matrix(val, np.asarray(data["n"])[val])
    ytrain, yval = np.asarray(data["y"])[train], np.asarray(data["y"])[val]
    ntrain, nval = np.asarray(data["n"])[train], np.asarray(data["n"])[val]
    train_groups, val_groups = np.asarray(data["group"])[train], np.asarray(data["group"])[val]
    if np.any(ytrain < 0) or np.any(yval < 0) or not np.array_equal(ytrain.sum(-1), ntrain) or not np.array_equal(yval.sum(-1), nval):
        raise ValueError("Invalid legal-count targets in training or validation split")
    region_dir = output / "regions"
    region_dir.mkdir(exist_ok=True)
    estimators, records, history = [], [], []
    remaining_train, remaining_val = ntrain.copy(), nval.copy()
    started = time.perf_counter()
    regions = tqdm(range(k-1), desc=f"HistGBDT seed={seed} regions", dynamic_ncols=True,
                   ascii=True, file=sys.stdout, disable=not cfg.get("gbdt_progress", True))
    with threadpool_limits(limits=settings["threads"]):
        for region in regions:
            save_json(output / "fit_progress.json", dict(model="hist_gbdt", seed=int(seed),
                phase="fit_conditional_classifier", region=region, completed_regions=region,
                total_regions=k-1, candidate_max_iter=settings["max_iter"]))
            path = region_dir / f"region_{region:02d}.joblib"
            if path.exists():
                saved = joblib.load(path)
                if saved["contract_hash"] != contract_hash or saved["region"] != region:
                    raise RuntimeError("Partial HistGBDT region checkpoint contract differs")
                estimator, record = saved["estimator"], saved["record"]
            else:
                step_start = time.perf_counter()
                xtrain[:, -1], xval[:, -1] = remaining_train/5., remaining_val/5.
                active = remaining_train > 0
                classes = np.unique(ytrain[active, region])
                stage_history = []
                if len(classes) <= 1:
                    estimator = ConstantCountClassifier(int(classes[0]) if len(classes) else 0)
                    selected = 0
                else:
                    candidate = _classifier(settings, int(seed)+region, settings["max_iter"])
                    candidate.fit(xtrain[active], ytrain[active, region])
                    stages = tqdm(candidate.staged_predict_proba(xval), total=settings["max_iter"],
                        desc=f"Validate region R{region}", leave=False, dynamic_ncols=True,
                        ascii=True, file=sys.stdout, disable=not cfg.get("gbdt_progress", True))
                    for iteration, raw in enumerate(stages, 1):
                        probability = legal_probabilities(raw, candidate.classes_, remaining_val, settings["probability_floor"])
                        loss = _conditional_nll(probability, yval[:, region], val_groups)
                        stage_history.append(dict(iteration=iteration, validation_conditional_nll=loss))
                        if iteration == 1 or iteration % 10 == 0 or iteration == settings["max_iter"]:
                            save_json(output / "fit_progress.json", dict(model="hist_gbdt", seed=int(seed),
                                phase="select_validation_iteration", region=region, completed_regions=region,
                                total_regions=k-1, iteration=iteration, candidate_max_iter=settings["max_iter"],
                                validation_conditional_nll=loss))
                    selected = min(stage_history, key=lambda row: (row["validation_conditional_nll"], row["iteration"]))["iteration"]
                    # Refit the selected public iteration count, never truncate private tree arrays.
                    save_json(output / "fit_progress.json", dict(model="hist_gbdt", seed=int(seed),
                        phase="refit_selected_iteration", region=region, completed_regions=region,
                        total_regions=k-1, selected_iterations=selected))
                    estimator = _classifier(settings, int(seed)+region, selected)
                    estimator.fit(xtrain[active], ytrain[active, region])
                train_probability = legal_probabilities(estimator.predict_proba(xtrain), estimator.classes_, remaining_train, settings["probability_floor"])
                val_probability = legal_probabilities(estimator.predict_proba(xval), estimator.classes_, remaining_val, settings["probability_floor"])
                record = dict(region=region, selected_iterations=selected, observed_classes=estimator.classes_.tolist(),
                    active_training_rows=int(active.sum()), training_rows=len(train), validation_rows=len(val),
                    train_conditional_nll=_conditional_nll(train_probability, ytrain[:, region], train_groups),
                    validation_conditional_nll=_conditional_nll(val_probability, yval[:, region], val_groups),
                    constant_fallback=isinstance(estimator, ConstantCountClassifier),
                    trees=int(estimator.n_iter_*estimator.n_trees_per_iteration_),
                    fit_seconds=time.perf_counter()-step_start, stages=stage_history)
                _dump_joblib(path, dict(contract_hash=contract_hash, region=region, estimator=estimator, record=record))
            estimators.append(estimator)
            records.append(record)
            history.extend(dict(region=region, **stage) for stage in record["stages"])
            remaining_train -= ytrain[:, region]
            remaining_val -= yval[:, region]
            save_json(output / "history.json", history)
            save_json(output / "training_state.json", dict(contract_hash=contract_hash, completed_regions=len(records), total_regions=k-1))
            regions.set_postfix(region=f"R{region}", selected=record["selected_iterations"], val_nll=f"{record['validation_conditional_nll']:.4f}")
    info = dict(algorithm="HistGradientBoostingClassifier", model="hist_gbdt", device="cpu",
        parameters=None, fitted_trees=sum(row["trees"] for row in records), epochs=0,
        selected_iterations=[row["selected_iterations"] for row in records], fit_seconds=time.perf_counter()-started,
        completed_region_fit_seconds=sum(row["fit_seconds"] for row in records),
        training_objective="conditional_multiclass_log_loss", conditional_rps_weight=0,
        validation_draws=None, selection_objective="validation_exact_conditional_nll",
        selection_aggregation="cutoff_to_match_then_equal_match_mean",
        validation_exact_count_nll=sum(row["validation_conditional_nll"] for row in records),
        train_prefixes=len(train), validation_prefixes=len(val), feature_count=624,
        prefix_summary=FEATURE_VERSION, zero_remaining_rule="deterministic_zero_not_fitted",
        probability_floor=settings["probability_floor"], settings=settings,
        sklearn_version=sklearn.__version__, contract_hash=contract_hash, regions=records)
    model = HistGBDTCountModel(estimators, k, settings, seed, contract, info)
    _dump_joblib(checkpoint, model)
    save_json(output / "fit.json", info)
    save_json(output / "fit_progress.json", dict(model="hist_gbdt", seed=int(seed),
        phase="complete", completed_regions=k-1, total_regions=k-1,
        validation_exact_count_nll=info["validation_exact_count_nll"]))
    return model, info


def evaluate_gbdt(model, data, indices, samples=256, batch_size=128, cfg=None):
    """Evaluate complete-history descriptors; tree prediction is independent of draws."""
    cfg = cfg or {}
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices) or samples < 2 or samples % 2 or batch_size < 1:
        raise ValueError("Need nonempty indices, positive batch size, and an even sample count >=2")
    summary = _summary(data)
    predictions = np.empty((len(indices), samples, model.k), dtype=np.float32)
    score_parts = []
    rng = np.random.default_rng(int(cfg.get("gbdt_sampling_seed", model.seed)))
    batches = tqdm(range(0, len(indices), batch_size), desc="HistGBDT predictions",
        dynamic_ncols=True, ascii=True, file=sys.stdout, disable=not cfg.get("gbdt_progress", True))
    for offset in batches:
        rows = indices[offset:offset+batch_size]
        n = np.asarray(data["n"])[rows]
        # No target enters descriptor construction or the conditional lookup.
        lookup = model.conditional_lookup(summary.matrix(rows), n)
        draws = model.sample_lookup(lookup, n, samples, rng)
        truth = np.asarray(data["y"])[rows]
        scores = sample_scores(draws, truth)
        scores["exact_count_nll"] = model.nll_lookup(lookup, truth, n)
        score_parts.append(scores)
        predictions[offset:offset+len(rows)] = draws
    scores = {key: np.concatenate([part[key] for part in score_parts]) for key in score_parts[0]}
    return scores, predictions
