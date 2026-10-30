"""Frozen-data recurrent/tree controls, safe resume, and complete return bundles.

Original paper-model artifacts are immutable inputs. Standard-library-only
imports here keep failure packaging available even if a dependency is missing.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import uuid
import zipfile

DATA_SHA256 = "a624d414d323c096fd5d5ff195f8fa8f7beb9e7ed22494987406b4d6c41b129a"
OLD_MODELS = ("knn", "vanilla_transformer", "improved_transformer")
NEW_MODELS = ("lstm_count", "gru_count", "hist_gbdt")
SEEDS = (2026, 2027, 2028)
METRICS = ("energy_score", "count_brier", "presence_brier", "pair_presence_brier", "count_mae", "exact_count_nll")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def csv_write(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sample_ids(data, indices):
    return [f"{data['group'][i]}:{data['demo'][i]}:{int(data['round'][i])}:{int(data['tick'][i])}" for i in indices]


def test_fingerprint(data, indices):
    import numpy as np
    digest = hashlib.sha256("\n".join(sample_ids(data, indices)).encode("utf-8"))
    digest.update(np.asarray(data["y"][indices], dtype="<i8").tobytes())
    digest.update(np.asarray(data["n"][indices], dtype="<i8").tobytes())
    return digest.hexdigest()


def validate_dataset(root):
    import numpy as np
    path = root / "data/processed/dataset.npz"
    actual_sha = sha(path)
    if actual_sha != DATA_SHA256:
        raise RuntimeError(f"Frozen dataset SHA mismatch: {actual_sha}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data["frames"].shape != (275577, 311) or data["y"].shape != (98203, 12):
        raise RuntimeError("Unexpected frozen dataset shape")
    if not np.isfinite(data["frames"]).all() or not np.array_equal(data["y"].sum(-1), data["n"]):
        raise RuntimeError("Nonfinite features or illegal ground-truth counts")
    if not np.isin(data["n"], [1, 2, 3, 4, 5]).all() or (data["y"] < 0).any():
        raise RuntimeError("Invalid hidden population")
    split_stats, groups = {}, []
    for split, expected in [("train", (34, 707, 60338)), ("val", (5, 127, 10831)), ("test", (14, 293, 27034))]:
        indices = np.flatnonzero(data["split"] == split)
        current_groups = set(data["group"][indices].tolist())
        rounds = set(zip(data["group"][indices].tolist(), data["round"][indices].tolist()))
        actual = (len(current_groups), len(rounds), len(indices))
        if actual != expected:
            raise RuntimeError(f"Split {split}: {actual} != {expected}")
        groups.append(current_groups)
        split_stats[split] = dict(matches=actual[0], rounds=actual[1], prefixes=actual[2])
    if any(groups[i] & groups[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise RuntimeError("Match leakage between data splits")
    test = np.flatnonzero(data["split"] == "test")
    if len(set(sample_ids(data, test))) != len(test):
        raise RuntimeError("Repeated test cutoff identifiers")
    return data, dict(dataset_sha256=DATA_SHA256, split_stats=split_stats,
                      test_alignment_sha256=test_fingerprint(data, test), history_mode="full")


def check_cohort(location, data, indices, model, seed, samples=256, expected_hashes=None):
    import numpy as np
    import pandas as pd
    if expected_hashes:
        for filename, expected in expected_hashes.items():
            if not (location / filename).is_file() or sha(location / filename) != expected:
                raise RuntimeError(f"Cached artifact changed: {location / filename}")
    scores_file = location / "test_scores.csv"
    predictions_file = location / "test_predictions.npz"
    frame = pd.read_csv(scores_file, dtype={"group": str, "demo": str, "sample_id": str})
    ids = sample_ids(data, indices)
    if len(frame) != len(indices) or frame["sample_id"].tolist() != ids:
        raise RuntimeError(f"Cutoff/order mismatch: {scores_file}")
    if not (frame["model"] == model).all() or not (frame["seed"] == seed).all():
        raise RuntimeError(f"Wrong method/seed labels: {scores_file}")
    for key in ("group", "demo", "round", "tick", "n"):
        if not np.array_equal(frame[key].astype(str).to_numpy(), data[key][indices].astype(str)):
            raise RuntimeError(f"Test context mismatch {key}: {scores_file}")
    required = METRICS[:-1] if model == "knn" else METRICS
    if any(metric not in frame or not np.isfinite(frame[metric].to_numpy()).all() for metric in required):
        raise RuntimeError(f"Missing/nonfinite scores: {scores_file}")
    with np.load(predictions_file, allow_pickle=False) as predictions:
        draws = predictions["samples"]
        if draws.shape != (len(indices), samples, 12):
            raise RuntimeError(f"Incorrect prediction shape: {predictions_file}")
        if not np.array_equal(predictions["truth"], data["y"][indices]):
            raise RuntimeError(f"Test-label mismatch: {predictions_file}")
        for key in ("group", "tick"):
            if not np.array_equal(predictions[key].astype(str), data[key][indices].astype(str)):
                raise RuntimeError(f"Prediction {key} mismatch: {predictions_file}")
        if not np.issubdtype(draws.dtype, np.integer) or (draws < 0).any() or not np.all(draws.sum(-1) == data["n"][indices, None]):
            raise RuntimeError(f"Illegal sampled count vectors: {predictions_file}")
        if "sample_id" in predictions and predictions["sample_id"].tolist() != ids:
            raise RuntimeError(f"Prediction cutoff IDs mismatch: {predictions_file}")
    return frame


def validate_reused(root, data, output):
    import numpy as np
    old = root / "results/paper_model"
    manifest = read_json(root / "FROZEN_REFERENCE_MANIFEST.json")
    if manifest["dataset_sha256"] != DATA_SHA256:
        raise RuntimeError("Reference manifest specifies different data")
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen reference changed/missing: {item['path']}")
    test = np.flatnonzero(data["split"] == "test")
    frames, records = [], []
    for seed in SEEDS:
        for model in OLD_MODELS:
            location = old / f"seed_{seed}" / model
            print(f"Checking frozen cohort {model} / {seed} ...", flush=True)
            frame = check_cohort(location, data, test, model, seed)
            frame["provenance"] = "reused_saved"
            frames.append(frame)
            records.append(dict(model=model, seed=seed, provenance="reused_saved",
                                source=location.relative_to(root).as_posix(),
                                test_scores_sha256=sha(location / "test_scores.csv"),
                                test_predictions_sha256=sha(location / "test_predictions.npz"), test_cutoffs=len(test)))
    write_json(output / "log/reused_cohorts.json", records)
    return frames, records


def code_fingerprints(root):
    paths = [root / name for name in ["main.py", "supplemental_entry.py", "supplemental_runner.py", "oneclick_environment.py"]]
    paths += sorted((root / "cs2").glob("*.py"))
    return {path.relative_to(root).as_posix(): sha(path) for path in paths}


def snapshot_code(root, output, fingerprints):
    for relative, expected in fingerprints.items():
        target=output/"code_at_start"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=expected:
            raise RuntimeError("Code snapshot differs: "+relative)
        if not target.exists():shutil.copy2(root/relative,target)


def bind_contract(path, contract):
    if path.exists() and read_json(path) != contract:
        raise RuntimeError(f"Resume contract differs: {path}. Preserve that run and choose a new run_name.")
    write_json(path, contract)


def smoke_data(data):
    import numpy as np
    selected = []
    lengths = data["end"] - data["start"]
    for split, number in [("train", 4), ("val", 2), ("test", 2)]:
        available = np.flatnonzero(data["split"] == split)
        selected.extend(available[np.argsort(lengths[available], kind="stable")[:number]])
    selected = np.asarray(sorted(selected), dtype=np.int64)
    tiny = {key: value[selected].copy() for key, value in data.items() if key not in {"frames", "round_starts"}}
    prefixes = [data["frames"][data["start"][i]:data["end"][i]].copy() for i in selected]
    sizes = np.asarray([len(prefix) for prefix in prefixes])
    tiny["frames"] = np.concatenate(prefixes)
    tiny["start"] = np.r_[0, sizes.cumsum()[:-1]]
    tiny["end"] = sizes.cumsum()
    tiny["round_starts"] = tiny["start"].copy()
    tiny["source_sample_index"] = selected
    return tiny


def score_context(data, indices):
    import numpy as np
    import pandas as pd
    keys = ["group", "demo", "round", "tick", "n", "elapsed", "contact_age", "had_contact", "ct_team", "t_team", "smoke_active", "ct_elo", "t_elo", "skill_band"]
    frame = pd.DataFrame({key: data[key][indices] for key in keys})
    current = data["frames"][data["end"][indices]-1]
    k = data["y"].shape[-1]
    frame["ct_alive"] = np.rint(current[:, 2*k]*5).astype(int)
    frame["t_alive"] = np.rint(current[:, 2*k+1]*5).astype(int)
    frame["audio_available"] = current[:, -1] > .5
    frame["ct_elo_known"] = current[:, 20*k+13] > .5
    frame["t_elo_known"] = current[:, 20*k+20] > .5
    frame["sample_id"] = sample_ids(data, indices)
    frame["prefix_frames"] = data["end"][indices]-data["start"][indices]
    frame["sample_index"] = data.get("source_sample_index", np.arange(len(data["n"])))[indices]
    return frame


def train_cohort(root, output, data, cfg, device, model_name, seed, contract_hash, smoke=False):
    import numpy as np
    import pandas as pd
    import torch
    from cs2.models import MODEL_CLASSES
    from cs2.training import fit, seed_everything
    from cs2.evaluation import evaluate, calibration_table
    is_tree = model_name == "hist_gbdt"
    checkpoint_name = "best_gbdt.joblib" if is_tree else "best.pt"
    test = np.flatnonzero(data["split"] == "test")
    location = output / f"seed_{seed}" / model_name
    location.mkdir(parents=True, exist_ok=True)
    cohort_contract = dict(contract_sha256=contract_hash, model=model_name, seed=seed,
                           test_alignment_sha256=test_fingerprint(data, test), smoke_only=smoke)
    contract_path = location / "cohort_contract.json"
    if not contract_path.exists() and (any(location.glob("*.pt")) or any(location.glob("*.joblib"))):
        raise RuntimeError(f"Checkpoint without a verifiable contract: {location}")
    bind_contract(contract_path, cohort_contract)
    completed = location / "cohort_complete.json"
    if completed.exists():
        meta = read_json(completed)
        if meta.get("status") != "complete" or meta["contract"] != cohort_contract:
            raise RuntimeError(f"Completed cohort has different settings/status: {location}")
        print(f"Reusing validated supplemental cohort {model_name} / {seed}", flush=True)
        return check_cohort(location, data, test, model_name, seed, cfg["eval_samples"], meta["artifact_sha256"]), meta
    write_json(output / "log/current_progress.json", dict(stage="initializing", model=model_name, seed=seed))
    trained = location / "training_complete.json"
    if trained.exists():
        meta = read_json(trained)
        if meta["contract"] != cohort_contract or sha(location/checkpoint_name) != meta["best_sha256"] or sha(location/"fit.json") != meta["fit_sha256"]:
            raise RuntimeError(f"Training completion record changed: {location}")
        seed_everything(seed)
        info = read_json(location/"fit.json")
        if is_tree:
            # The tree loader also verifies its sklearn/version/feature contract.
            # The already checked checkpoint makes this a reuse, not another fit.
            from cs2.gbdt import fit_gbdt
            model, info = fit_gbdt(data,cfg,device,location,seed)
            # Regenerated metadata may have platform-specific line endings.
            write_json(trained, dict(contract=cohort_contract,checkpoint=checkpoint_name,
                                    best_sha256=sha(location/checkpoint_name),fit_sha256=sha(location/"fit.json")))
        else:
            model = MODEL_CLASSES[model_name](data["frames"].shape[-1], 12, cfg).to(device)
            checkpoint = torch.load(location/checkpoint_name, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model.prefix_capacity = info["prefix_frame_capacity"]
        print(f"Training complete; resuming evaluation for {model_name} / {seed}", flush=True)
    else:
        if is_tree:
            from cs2.gbdt import fit_gbdt
            model, info = fit_gbdt(data, cfg, device, location, seed)
        else:
            model, info = fit(model_name, data, cfg, device, location, seed)
        write_json(trained, dict(contract=cohort_contract, checkpoint=checkpoint_name,
                                best_sha256=sha(location/checkpoint_name), fit_sha256=sha(location/"fit.json")))
    seed_everything(seed+314159)
    write_json(output/"log/current_progress.json", dict(stage="test_predictions", model=model_name, seed=seed, test_samples=len(test)))
    before = time.perf_counter()
    if is_tree:
        from cs2.gbdt import evaluate_gbdt
        scores, samples = evaluate_gbdt(model, data, test, samples=cfg["eval_samples"],
                                        batch_size=cfg.get("gbdt_eval_batch_size",128),
                                        cfg=dict(cfg,gbdt_sampling_seed=seed+314159))
    else:
        scores, samples = evaluate(model, data, test, device, cfg["eval_samples"], cfg["eval_batch_size"], cfg,
                                    desc=f"{model_name} seed={seed} held-out test")
    if device.type=="cuda":torch.cuda.synchronize(device)
    evaluation_seconds=time.perf_counter()-before
    frame = score_context(data, test)
    for metric, values in scores.items():frame[metric] = values
    frame["model"], frame["seed"], frame["provenance"] = model_name, seed, "smoke_only" if smoke else "newly_trained"
    csv_write(frame, location/"test_scores.csv")
    csv_write(pd.DataFrame(calibration_table(samples, data["y"][test])), location/"calibration.csv")
    prediction_path = location/"test_predictions.npz"
    temporary = prediction_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, samples=samples.astype(np.uint8), truth=data["y"][test].astype(np.uint8),
                            group=data["group"][test], tick=data["tick"][test], n=data["n"][test],
                            sample_id=np.asarray(sample_ids(data, test)), sample_index=frame["sample_index"].to_numpy())
    temporary.replace(prediction_path)
    del samples, model
    if device.type=="cuda":torch.cuda.empty_cache()
    check_cohort(location, data, test, model_name, seed, cfg["eval_samples"])
    meta = dict(contract=cohort_contract, status="complete", provenance="smoke_only" if smoke else "newly_trained",
                evaluation_seconds=evaluation_seconds, evaluation_and_export_seconds=time.perf_counter()-before,
                evaluation_random_seed=seed+314159,
                training_objective=info["training_objective"],
                conditional_rps_weight=info["conditional_rps_weight"],
                artifact_sha256={name:sha(location/name) for name in ["test_scores.csv", "test_predictions.npz", checkpoint_name, "fit.json", "history.json"]})
    write_json(completed, meta)
    return frame, meta


def summarize(frames, output, repeats=10000, smoke=False):
    import numpy as np
    import pandas as pd
    combined = pd.concat(frames, ignore_index=True)
    csv_write(combined, output/"log/all_test_scores.csv")
    metrics = [metric for metric in METRICS if metric in combined]
    per_match = combined.groupby(["model", "seed", "group"], as_index=False)[metrics].mean()
    per_seed = per_match.groupby(["model", "seed"], as_index=False)[metrics].mean()
    csv_write(per_match, output/"log/per_match_metrics.csv")
    csv_write(per_seed, output/"log/per_seed_metrics.csv")
    rows = []
    for model, cohort in per_seed.groupby("model"):
        for metric in metrics:
            valid = cohort[metric].dropna()
            if valid.empty:continue
            rows.append(dict(model=model, metric=metric, seed_mean=float(valid.mean()),
                seed_std=float(valid.std(ddof=1)) if len(valid)>1 else None, seed_count=len(valid),
                provenance="smoke_only" if smoke else ("reused_saved" if model in OLD_MODELS else "newly_trained")))
    summary = pd.DataFrame(rows)
    summary["rank"] = summary.groupby("metric")["seed_mean"].rank(method="min", ascending=True).astype(int)
    csv_write(summary, output/"log/combined_summary.csv")
    comparisons = []
    match_means = per_match.groupby(["model", "group"])[metrics].mean().reset_index()
    rng = np.random.default_rng(2026)
    if "improved_transformer" in match_means.model.unique():
        for baseline in sorted(set(match_means.model)-{"improved_transformer"}):
            for metric in metrics:
                wide = match_means.pivot(index="group", columns="model", values=metric)
                valid = wide[["improved_transformer", baseline]].dropna()
                if valid.empty:continue
                delta = (valid["improved_transformer"]-valid[baseline]).to_numpy()
                boot = delta[rng.integers(len(delta), size=(repeats, len(delta)))].mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                reference = float(valid[baseline].mean())
                comparisons.append(dict(candidate="improved_transformer", baseline=baseline, metric=metric,
                    delta=float(delta.mean()), ci_low=float(lo), ci_high=float(hi), matches=len(delta),
                    relative_reduction_percent=float(-delta.mean()/reference*100) if reference else None,
                    sign_convention="negative delta favors RAST; negative reduction favors baseline"))
    csv_write(pd.DataFrame(comparisons), output/"log/paired_comparisons.csv")
    write_json(output/"log/combined_summary.json", dict(status="smoke_only" if smoke else "complete",
        aggregation="cutoff mean within match -> equal match mean per seed -> mean/sample SD across seeds",
        scores_lower_is_better=True, rows=json.loads(summary.to_json(orient="records")), comparisons=comparisons,
        all_methods_retained_regardless_of_ranking=True,
        bootstrap=dict(repeats=repeats, random_seed=2026, unit="match", conditional_on="saved trained seed runs")))
    plot_summary(summary, output, smoke)
    return combined, summary


def plot_summary(summary, output, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"pdf.fonttype":42, "svg.fonttype":"none"})
    order = [model for model in (*OLD_MODELS, *NEW_MODELS) if model in set(summary.model)]
    labels = {"knn":"KNN", "vanilla_transformer":"Vanilla", "improved_transformer":"RAST", "lstm_count":"LSTM", "gru_count":"GRU", "hist_gbdt":"HistGBDT"}
    colors = ["#D55E00" if model=="improved_transformer" else "#0072B2" for model in order]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), layout="constrained")
    for ax, metric in zip(axes.flat, METRICS):
        source = summary[summary.metric==metric].set_index("model").reindex(order)
        values = source.seed_mean.to_numpy(float)
        error = np.nan_to_num(source.seed_std.to_numpy(dtype=float), nan=0.0)
        ax.barh(np.arange(len(order)), values, xerr=error, color=colors, alpha=.85, capsize=2)
        ax.set(yticks=range(len(order)), yticklabels=[labels[model] for model in order], title=metric.replace("_", " "))
        ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
        for j, value in enumerate(values):
            if not np.isfinite(value):ax.text(.02, j, "not available", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    fig.suptitle("SOFTWARE SMOKE CHECK - NOT A PAPER RESULT" if smoke else "Frozen-data additional controls: all completed methods", fontsize=13)
    folder = output/"log/figures"; folder.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:fig.savefig(folder/f"combined_comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    csv_write(summary, folder/"combined_comparison_source.csv")
    write_json(folder/"combined_comparison_metadata.json", dict(error_bars="sample SD across training seeds; zero for one-seed smoke",
        lower_is_better=True, sources=["../combined_summary.csv", "../per_seed_metrics.csv"], retained_models=order,
        smoke_only=smoke, redraw="python main.py --replot-only --offline"))


def return_bundle(root, output, status, smoke=False):
    """Explicit roots avoid recursive inclusion of runtime/cache/return files."""
    allowed = {".py", ".json", ".jsonl", ".csv", ".npz", ".pt", ".joblib", ".pdf", ".svg", ".png", ".md", ".txt", ".log"}
    selected = set()
    for path in root.iterdir():
        if path.is_file() and (path.suffix in {".py", ".json", ".md", ".txt"} or path.name in {".gitignore", "LICENSE", "LICENSE_CODE"}):
            if not path.name.startswith("LATEST_") and path.name!="RETURN_MANIFEST.json":selected.add(path)
    folders = [root/"cs2", root/"tools", output]
    if not smoke:folders += [root/"data/processed", root/"data/provenance", root/"results/paper_model"]
    for folder in folders:
        if not folder.exists():continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix not in allowed:continue
            if any(part in {"__pycache__", ".venv_run", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", "raw"} for part in path.relative_to(root).parts):continue
            if path.name.startswith("RETURN_") or path.name.endswith(".tmp"):continue
            selected.add(path)
    manifest = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(selected)]
    target = root/f"RETURN_{'SMOKE_' if smoke else ''}{stamp()}_{status}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(selected):archive.write(path, "baseline_expansion/"+path.relative_to(root).as_posix())
        archive.writestr("baseline_expansion/RETURN_MANIFEST.json", json.dumps(dict(status=status, smoke_only=smoke,
            paper_experiment_complete=status=="completed" and not smoke, files=manifest,
            exclusions=["raw demos", "virtual environments", "Python caches", "previous return ZIPs"]), indent=2))
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:raise RuntimeError(f"Return archive CRC failure: {bad}")
    record = dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target), status=status,
                  smoke_only=smoke, crc_check="PASS", included_files=len(selected))
    write_json(output/"log/return_archive.json", record)
    write_json(root/("LATEST_SMOKE_RETURN.json" if smoke else "LATEST_RETURN.json"), record)
    print("\nReturn archive: "+str(target), flush=True)
    return record


def package_early_failure(root, cfg, error, smoke=False):
    run_name = cfg.get("run_name", "baseline_supplement")
    if not isinstance(run_name,str) or Path(run_name).name!=run_name or run_name in {".",".."}:run_name="configuration_failure"
    output = root/("smoke_checks/early_failure_"+stamp() if smoke else "outputs/"+run_name)
    status = "interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
    write_json(output/"log/run_status.json", dict(status=status, paper_experiment_complete=False,
        stage="environment_or_configuration", error=repr(error), traceback=traceback.format_exc()))
    try:return_bundle(root,output,status,smoke)
    except Exception as packing_error:print("Return packing failed: "+repr(packing_error),file=sys.stderr,flush=True)


def execute(root, cfg, smoke=False, prepare_only=False, replot_only=False):
    root = Path(root).resolve()
    run_name = cfg.get("run_name","baseline_supplement")
    if not isinstance(run_name,str) or Path(run_name).name!=run_name or run_name in {".",".."}:
        raise ValueError("run_name must be a simple folder name")
    output = root/("smoke_checks/"+stamp() if smoke else "outputs/"+run_name)
    output.mkdir(parents=True,exist_ok=True)
    complete = output/"COMPLETE.json"
    if complete.exists() and not prepare_only and not replot_only:
        history=output/"completion_history"/(stamp()+".json")
        history.parent.mkdir(exist_ok=True); complete.replace(history)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(output/"execution.log",encoding="utf-8")],force=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    status="running"
    write_json(output/"log/run_status.json",dict(status=status,smoke_only=smoke,paper_experiment_complete=False))
    try:
        import numpy as np
        data,data_info=validate_dataset(root)
        write_json(output/"log/data_validation.json",data_info)
        if tuple(cfg["models"])!=NEW_MODELS or tuple(cfg["seeds"])!=SEEDS:
            raise RuntimeError("Supplemental profile requires LSTM/GRU/HistGBDT and seeds 2026/2027/2028")
        if cfg.get("history_mode")!="full":raise RuntimeError("Complete observation histories are required")
        cfg=dict(cfg)
        cfg["region_centers"]=read_json(root/"data/processed/regions.json")["centers_scaled"]
        if smoke:
            data=smoke_data(data)
            cfg.update(epochs=1,patience=1,seeds=[2026],batch_size=4,micro_batch_size=4,max_micro_batch=4,
                auto_batch=False,transformer_dim=24,transformer_layers=1,transformer_heads=4,hidden_dim=24,
                rnn_layers=1,lstm_hidden_dim=16,gru_hidden_dim=16,validation_samples=8,eval_samples=8,
                eval_batch_size=2,pad_multiple=1,cpu_threads=2,device="cpu",amp=False,require_cuda=False,smoke_only=True,
                gbdt_max_iter=2,gbdt_min_samples_leaf=1,gbdt_threads=2,gbdt_eval_batch_size=2)
            reused,records=[],[]
            write_json(output/"NOT_FOR_PAPER.json",dict(reason="Eight real short prefixes; one epoch/seed; each selected complete history preserved",prefixes=8,train=4,val=2,test=2))
        else:reused,records=validate_reused(root,data,output)
        write_json(output/"resolved_config.json",cfg)
        fingerprints=code_fingerprints(root)
        write_json(output/"log/code_fingerprints.json",fingerprints)
        settings={key:value for key,value in cfg.items() if key not in {"offline","device","run_name","auto_project_environment"}}
        contract=dict(schema=1,data=data_info,config=settings,code=fingerprints,
            frozen_reference_manifest_sha256=sha(root/"FROZEN_REFERENCE_MANIFEST.json"),smoke_only=smoke,
            new_test_alignment_sha256=test_fingerprint(data,np.flatnonzero(data["split"]=="test")))
        contract_hash=canonical_hash(contract)
        if not prepare_only:bind_contract(output/"experiment_contract.json",contract)
        if not prepare_only:snapshot_code(root,output,fingerprints)
        if prepare_only:
            status="prepared_verified"
            write_json(output/"log/preflight.json",dict(status=status,cohorts_verified=len(records),**data_info))
            print("Prepared data and nine frozen cohorts verified. No training performed.",flush=True)
            return
        frames=list(reused)
        if replot_only:
            test=np.flatnonzero(data["split"]=="test")
            for seed in cfg["seeds"]:
                for name in cfg["models"]:
                    location=output/f"seed_{seed}"/name
                    marker=read_json(location/"cohort_complete.json")
                    if marker.get("status")!="complete" or marker["contract"]["contract_sha256"]!=contract_hash:
                        raise RuntimeError("Cached supplemental contract/status changed")
                    frames.append(check_cohort(location,data,test,name,seed,cfg["eval_samples"],marker["artifact_sha256"]))
            summarize(frames,output,cfg.get("bootstrap_repeats",10000))
            status="completed"
            write_json(complete,dict(status="completed",paper_experiment_complete=True,action="redraw_only",
                newly_trained_cohorts=len(NEW_MODELS)*len(SEEDS),reused_saved_cohorts=len(OLD_MODELS)*len(SEEDS),
                methods=len(OLD_MODELS)+len(NEW_MODELS),seeds=list(SEEDS),contract_sha256=contract_hash,
                ranking_is_not_a_completion_criterion=True))
            return
        import torch
        torch.set_num_threads(cfg.get("cpu_threads",4))
        selected=cfg.get("device","auto")
        if smoke:selected="cpu"
        if selected=="auto":selected="cuda" if torch.cuda.is_available() else "cpu"
        device=torch.device(selected)
        if not smoke and device.type!="cuda":raise RuntimeError("Formal training requires CUDA; --smoke is the isolated tiny CPU check")
        if device.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
            torch.ones(1,device=device).add_(1);torch.cuda.synchronize(device)
            torch.set_float32_matmul_precision("high")
        versions={name:importlib.metadata.version(name) for name in ["torch","numpy","pandas","matplotlib"]}
        write_json(output/"environment.json",dict(python=sys.version,platform=platform.platform(),packages=versions,
            device=str(device),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None))
        for seed in cfg["seeds"]:
            for name in cfg["models"]:
                print(f"\nSupplemental cohort: {name} / seed {seed}",flush=True)
                frame,meta=train_cohort(root,output,data,cfg,device,name,seed,contract_hash,smoke)
                frames.append(frame)
                records.append(dict(model=name,seed=seed,provenance=meta["provenance"],
                    location=(output/f"seed_{seed}"/name).relative_to(root).as_posix(),test_cutoffs=len(frame),
                    training_objective=meta["training_objective"],conditional_rps_weight=meta["conditional_rps_weight"]))
                write_json(output/"log/cohort_inventory.json",records)
        expected=len(NEW_MODELS) if smoke else (len(OLD_MODELS)+len(NEW_MODELS))*len(SEEDS)
        if len(records)!=expected:raise RuntimeError(f"Incomplete cohorts: {len(records)}/{expected}")
        summarize(frames,output,min(cfg.get("bootstrap_repeats",10000),50) if smoke else cfg.get("bootstrap_repeats",10000),smoke)
        if not smoke:
            for item in read_json(root/"FROZEN_REFERENCE_MANIFEST.json")["files"]:
                if sha(root/item["path"])!=item["sha256"]:raise RuntimeError("Frozen reference changed: "+item["path"])
        if code_fingerprints(root)!=fingerprints:raise RuntimeError("Implementation changed during this experiment")
        status="smoke_passed" if smoke else "completed"
        if not smoke:write_json(complete,dict(status="completed",paper_experiment_complete=True,
            newly_trained_cohorts=len(NEW_MODELS)*len(SEEDS),reused_saved_cohorts=len(OLD_MODELS)*len(SEEDS),
            methods=len(OLD_MODELS)+len(NEW_MODELS),seeds=list(SEEDS),contract_sha256=contract_hash,
            ranking_is_not_a_completion_criterion=True))
    except (Exception,KeyboardInterrupt) as error:
        status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
        setattr(error,"supplemental_bundle_handled",True)
        write_json(output/"log/FAILURE.json",dict(status=status,error=repr(error),traceback=traceback.format_exc(),paper_experiment_complete=False))
        raise
    finally:
        write_json(output/"log/run_status.json",dict(status=status,smoke_only=smoke,
            paper_experiment_complete=status=="completed" and not smoke,output=output.relative_to(root).as_posix(),
            finished_utc=datetime.now(timezone.utc).isoformat()))
        try:return_bundle(root,output,status,smoke)
        except Exception as error:
            write_json(output/"log/RETURN_PACKING_FAILED.json",dict(error=repr(error),traceback=traceback.format_exc()))
            print("Return ZIP could not be written: "+repr(error),file=sys.stderr,flush=True)
            if status in {"completed","prepared_verified","smoke_passed"}:
                setattr(error,"supplemental_bundle_handled",True)
                raise
