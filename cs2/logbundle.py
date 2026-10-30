"""A portable evidence folder that can be analyzed/redrawn without retraining."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from .io import save_json


def enrich_test_context(output, root):
    import numpy as np
    import pandas as pd
    path = root / "data/processed/dataset.npz"
    if not path.exists(): return
    with np.load(path, allow_pickle=False) as z:
        test = z["split"] == "test"
        k = z["y"].shape[-1]
        current = z["frames"][z["end"][test]-1]
        base = pd.DataFrame({key: z[key][test] for key in ("group", "demo", "round", "tick")})
        base["ct_alive"] = np.rint(current[:, 2*k]*5).astype(int)
        base["t_alive"] = np.rint(current[:, 2*k+1]*5).astype(int)
        base["audio_available"] = current[:, -1] > .5
        base["ct_elo_known"], base["t_elo_known"] = current[:, 20*k+13] > .5, current[:, 20*k+20] > .5
        base["sample_id"] = base[["group", "demo", "round", "tick"]].astype(str).agg(":".join, axis=1)
    keys = ["group", "demo", "round", "tick"]
    base["group"] = base.group.astype(str)
    base["demo"] = base.demo.astype(str)
    for csv in [output / "all_test_scores.csv", *output.glob("seed_*/*/test_scores.csv")]:
        if not csv.exists(): continue
        scores = pd.read_csv(csv, dtype={"group": str, "demo": str})
        extras = [c for c in base if c not in keys and c not in scores]
        if extras:
            before = len(scores)
            scores = scores.merge(base[keys+extras], on=keys, how="left", validate="many_to_one", sort=False)
            if len(scores) != before or scores["sample_id"].isna().any():
                raise ValueError("Cannot align test context to saved predictions")
            scores.to_csv(csv, index=False)
        # Each row of the draw archive has a complete stable identity.  This
        # permits analysis after moving log/ without relying on implicit CSV
        # row order or assuming that match+tick uniquely identifies a map.
        predictions = csv.parent / "test_predictions.npz"
        if csv.name == "test_scores.csv" and predictions.exists():
            with np.load(predictions, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            if len(arrays["truth"]) != len(scores):
                raise ValueError(f"Prediction/score row count differs: {predictions}")
            for key in ("group", "tick"):
                if key in arrays and not np.array_equal(arrays[key].astype(str), scores[key].astype(str).to_numpy()):
                    raise ValueError(f"Prediction/score {key} order differs: {predictions}")
            changed = False
            for key in ("sample_id", "demo", "round", "n", "seed", "model"):
                if key not in scores:
                    continue
                values = scores[key].to_numpy()
                if key in arrays:
                    if not np.array_equal(arrays[key].astype(str), values.astype(str)):
                        raise ValueError(f"Prediction/score {key} differs: {predictions}")
                    continue
                arrays[key] = values.astype(str) if values.dtype.kind in "OU" else values
                changed = True
            if changed:
                np.savez_compressed(predictions, **arrays)


def export_log(output, root, cfg, summary, status="completed", error=None):
    enrich_test_context(output, root)
    destination = output / "log"
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "current_progress.json", {"stage": status, "training_seeds": cfg["seeds"], "run_name": cfg.get("run_name")})
    for source in list(output.rglob("*")):
        relative = source.relative_to(output)
        if not source.is_file() or relative.parts[0] == "log" or source.suffix == ".tmp" or (source.suffix == ".pt" and source.name != "best.pt"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    provenance = destination / "provenance"
    provenance.mkdir(exist_ok=True)
    for name in ("summary.json", "dataset.npz", "demo_audit.csv", "regions.json", "split_matches.json", "skill_context.json", "DATA_CARD.md", "map_support.json", "map_support.npz", "raw_fingerprints.json", "split_rebalance.json", "skill_band_migration.json"):
        path = root / "data/processed" / name
        if path.exists(): shutil.copy2(path, provenance / name)
    if (root / "data/audit").exists():
        shutil.copytree(root / "data/audit",provenance / "data_preparation_audit",dirs_exist_ok=True)
    manifest = root / "data/manifest.json"
    split_path = root / "data/processed/split_matches.json"
    if manifest.exists() and split_path.exists():
        all_records = json.loads(manifest.read_text(encoding="utf-8"))
        groups = set(json.loads(split_path.read_text(encoding="utf-8")))
        save_json(provenance / "source_manifest.json", [r for r in all_records if str(r["match_id"]) in groups])
    scripts = destination / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("plot_paper.py", "PAPER_PLAN.md", "PAPER_STORY.md", "RESEARCH_REVIEW.md", "THIRD_PARTY.md"):
        if (root / name).exists(): shutil.copy2(root / name, scripts / name)
    (scripts / "cs2").mkdir(exist_ok=True)
    for source in (root / "cs2").glob("*.py"):
        shutil.copy2(source, scripts / "cs2" / source.name)
    # Preserve the code that actually trained the checkpoint, even if plotting
    # or runtime improvements were implemented while that run was in progress.
    snapshot = output / "code_at_start/cs2"
    if snapshot.exists():
        for source in snapshot.glob("*.py"):
            shutil.copy2(source, scripts / "cs2" / source.name)
    if (root / "figures").exists():
        shutil.copytree(root / "figures", scripts / "figures", dirs_exist_ok=True)
    if (root / "tools/verify_log.py").exists():
        shutil.copy2(root / "tools/verify_log.py", scripts / "verify_log.py")
    (scripts / "requirements_redraw.txt").write_text(
        "numpy>=1.26,<3\npandas>=2.2,<4\nmatplotlib>=3.8,<4\n"
        "scikit-learn>=1.4,<2\nscipy>=1.10,<2\nthreadpoolctl>=3.1,<4\n", encoding="utf-8")
    checkpoints = []
    for path in output.glob("seed_*/*/best.pt"):
        checkpoints.append({"path_in_training_output": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / "README.txt").write_text(
        "Download this entire log directory. It contains complete configs, environment/data provenance, training histories, per-cutoff predictions and scores, raw mechanism traces, interventions, projection parameters, figures and plot-only scripts.\n"
        "Replot from inside this directory: python scripts/plot_paper.py --run .\n"
        "If needed: python -m pip install -r scripts/requirements_redraw.txt\n"
        "Requires no torch, GPU, raw DEMO or retraining for redraw. Saved projection coordinates are reused.\n"
        "Best model checkpoints are included for optional new forward-pass diagnostics; last/optimizer checkpoints remain outside log. Weights and GPU are not required to redraw saved figures.\n"
        "A SOFTWARE_VERIFICATION_ONLY marker means a short software check, never evidence of publishable superiority.\n", encoding="utf-8")
    files = []
    for p in destination.rglob("*"):
        if not p.is_file() or p.name == "log_manifest.json": continue
        relative = p.relative_to(destination)
        role = "live_status" if relative.as_posix() == "current_progress.json" else "derived_plot" if relative.parts[0] == "figures" or p.name.startswith("projection_") else "code" if relative.parts[0] == "scripts" else "evidence"
        files.append({"path": relative.as_posix(), "role": role, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    save_json(destination / "log_manifest.json", {"status": status, "error": error, "utc": datetime.now(timezone.utc).isoformat(),
        "verification_only": bool(cfg.get("verification_samples_per_split")), "training_seeds": cfg["seeds"],
        "data_kind": summary["kind"], "samples": summary["samples"], "checkpoint_fingerprints": checkpoints,
        "files": files, "limitations": summary.get("limitations", [])})
    return destination
