from __future__ import annotations

import json
import logging
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch

from .data import DATA_KEYS, SCHEMA, build_dataset, synthetic_dataset, data_config
from .evaluation import evaluate, calibration_table
from .io import acquire, save_json
from .training import fit, seed_everything

LOG = logging.getLogger("cs2")


class Retrieval:
    exact_nll = False

    def __init__(self, x, y, n):
        from sklearn.neighbors import NearestNeighbors
        self.groups = {}
        self.scale = x.std(0).clip(.05)
        for number in np.unique(n):
            mask = n == number
            values = x[mask] / self.scale
            self.groups[int(number)] = (NearestNeighbors(n_neighbors=min(32, len(values))).fit(values), y[mask])

    def eval(self):
        return self

    def sample(self, x, n, samples):
        from .prefix import retrieval_descriptor
        descriptor = retrieval_descriptor(x).cpu().numpy()
        result = np.zeros((len(x), samples, next(iter(self.groups.values()))[1].shape[-1]), np.float32)
        for number in n.unique().tolist():
            mask = (n == number).cpu().numpy()
            if number not in self.groups:
                # Backoff is needed if this hidden population was absent from training.
                pooled = np.concatenate([v[1] for v in self.groups.values()]).sum(0) + 1
                p = pooled / pooled.sum()
                for index in np.where(mask)[0]:
                    result[index] = np.random.multinomial(number, p, size=samples)
                continue
            estimator, labels = self.groups[number]
            distances, neighbors = estimator.kneighbors(descriptor[mask] / self.scale)
            for index, candidates in zip(np.where(mask)[0], neighbors):
                result[index] = labels[np.random.choice(candidates, samples)]
        return torch.as_tensor(result, device=x.device)


def run(root: Path, cfg: dict, prepare_only=False, rebuild=False):
    if cfg["source"] == "synthetic":
        root = root / "synthetic_smoke"
    if not cfg.get("run_name", "run01").replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_name must contain letters, digits, underscores or hyphens")
    if cfg["regions"] < 2 or cfg["history_steps"] < 1 or cfg["eval_samples"] < 2 or cfg["eval_samples"] % 2:
        raise ValueError("Invalid region/history/evaluation sample configuration")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(root / "pipeline.log", encoding="utf-8")], force=True)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    from threadpoolctl import threadpool_limits
    thread_controller = threadpool_limits(limits=cfg.get("cpu_threads", 4))
    summary_path = root / "data/processed/summary.json"
    if summary_path.exists() and not rebuild:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = data_config(cfg)
        if summary.get("schema") != SCHEMA or summary.get("data_config") != expected:
            raise RuntimeError("Processed cache configuration differs. Restore the delivered config, or --rebuild with raw demos available.")
        if not (root / "data/processed/dataset.npz").is_file():
            raise FileNotFoundError("summary.json exists but dataset.npz is missing")
        LOG.info("Using prepared offline dataset: %s", summary_path)
    else:
        if cfg.get("prepared_data_only",False) and cfg["source"]!="synthetic":
            raise FileNotFoundError("Delivered processed data is missing. Extract the complete package again; this profile never downloads or rebuilds the dataset automatically.")
        if cfg["source"] == "synthetic":
            summary = synthetic_dataset(root, cfg)
        else:
            records = acquire(root, cfg)
            summary = build_dataset(root, records, cfg)
    LOG.info("Data: %s | %s matches, %s rounds, %s full-prefix samples, %s features/frame; split details in data/processed/summary.json",
             summary["kind"],summary.get("match_groups"),summary.get("rounds"),summary["samples"],summary.get("features_per_frame"))
    if prepare_only:
        return
    if cfg.get("cloud_training_only",False) and platform.system() == "Windows":
        raise RuntimeError("Full experiment is configured for Linux/AutoDL. Locally use --prepare-only or config_verify_improved.json for the small software check.")
    selected = cfg["device"]
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected)
    if cfg.get("require_cuda",False) and device.type != "cuda":
        raise RuntimeError("The full cloud experiment requires CUDA. Select a GPU PyTorch image; no large CPU training was started.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; select a CUDA-enabled PyTorch AutoDL image")
    if device.type == "cuda":
        # Trigger a real device operation before the long experiment.
        torch.ones(1, device=device).add_(1)
        LOG.info("GPU: %s, PyTorch=%s, CUDA=%s", torch.cuda.get_device_name(device), torch.__version__, torch.version.cuda)
        torch.set_float32_matmul_precision("high")
    else:
        LOG.info("CPU execution; AutoDL will select its GPU automatically when CUDA is available")
    data_file = root / "data/processed/dataset.npz"
    with np.load(data_file, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    limit = cfg.get("verification_samples_per_split")
    if limit:
        selected_indices = []
        original_size = len(data["n"])
        for split in ("train", "val", "test"):
            available = np.flatnonzero(data["split"] == split)
            selected_indices.extend(available[np.linspace(0, len(available)-1, min(int(limit), len(available))).round().astype(int)])
        selected_indices = np.asarray(sorted(selected_indices))
        data = {key: value if key in {"frames", "round_starts"} else value[selected_indices] for key, value in data.items()}
        LOG.warning("SOFTWARE VERIFICATION SUBSET: %d/%d samples (%s). Not a paper result.", len(data["n"]), original_size, summary["kind"])
    regions = json.loads((root / "data/processed/regions.json").read_text(encoding="utf-8"))
    cfg["region_centers"] = regions["centers_scaled"]
    history_mode = cfg.get("history_mode", "full")
    if history_mode != "full":
        raise ValueError("This experiment requires complete round prefixes: history_mode must be full")
    if not np.isfinite(data["frames"]).all() or not np.array_equal(data["y"].sum(-1), data["n"]):
        raise ValueError("Dataset invariant violated")
    groups = [set(data["group"][data["split"] == split]) for split in ("train", "val", "test")]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train/validation/test match leakage")
    output = root / "outputs" / cfg.get("run_name", "run01")
    output.mkdir(parents=True, exist_ok=True)
    def progress(stage,**details):
        save_json(output / "log/current_progress.json",dict(stage=stage,run_name=cfg.get("run_name"),training_seeds=cfg["seeds"],**details))
    live_manifest = output / "log/log_manifest.json"
    if live_manifest.exists():
        state = json.loads(live_manifest.read_text(encoding="utf-8"))
        state["status"] = "running"
        for item in state.get("files",[]):
            if item["path"] == "current_progress.json": item["role"] = "live_status"
        save_json(live_manifest,state)
    contract = {k: v for k, v in cfg.items() if k not in {"offline", "device", "epochs", "download_workers", "max_download_gb"}}
    import hashlib
    contract["implementation_sha256"] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                         for name in ("models.py", "improved.py", "training.py", "data.py", "prefix.py", "audio.py", "skill.py")}
    contract_path = output / "experiment_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Existing run has different settings. Use --run-name another_name for a separate experiment")
    save_json(contract_path, contract)
    snapshot = output / "code_at_start/cs2"
    snapshot.mkdir(parents=True, exist_ok=True)
    import shutil
    for filename in contract["implementation_sha256"]:
        target = snapshot / filename
        if not target.exists():
            shutil.copy2(Path(__file__).parent / filename, target)
    save_json(output / "resolved_config.json", cfg)
    context = output / "plot_context"
    save_json(context / "regions.json", regions)
    save_json(context / "data_summary.json", summary)
    import shutil
    for name in ("map_support.npz", "map_support.json"):
        if (root / "data/processed" / name).exists():
            shutil.copy2(root / "data/processed" / name, context / name)
    save_json(context / "split_matches.json", json.loads((root / "data/processed/split_matches.json").read_text(encoding="utf-8"))) if (root / "data/processed/split_matches.json").exists() else None
    save_json(context / "data_fingerprint.json", {"dataset_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(), "schema": SCHEMA})
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "demoparser2")}
    save_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": versions,
                                          "device": str(device), "cuda": torch.version.cuda,
                                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    test, train = data["split"] == "test", data["split"] == "train"
    all_scores, fits = [], []
    for seed in cfg["seeds"]:
        for name in cfg["models"]:
            progress("initializing_model",model=name,seed=seed)
            location = output / f"seed_{seed}" / name
            location.mkdir(parents=True, exist_ok=True)
            if name == "knn":
                cumulative = np.concatenate([np.zeros((1, data["frames"].shape[-1]), np.float64), data["frames"].cumsum(0, dtype=np.float64)])
                descriptors = np.concatenate([data["frames"][data["end"]-1],
                     (cumulative[data["end"]]-cumulative[data["start"]])/(data["end"]-data["start"])[:, None]], -1).astype(np.float32)
                model = Retrieval(descriptors[train], data["y"][train], data["n"][train])
                del cumulative, descriptors
                info = {"parameters": 0, "epochs": 0}
            else:
                model, info = fit(name, data, cfg, device, location, seed)
            seed_everything(seed + 314159)
            progress("test_predictions",model=name,seed=seed,test_samples=int(test.sum()))
            before = time.perf_counter()
            scores, samples = evaluate(model, data, np.flatnonzero(test), device, cfg["eval_samples"], cfg.get("eval_batch_size", 8), cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            info.update(model=name, seed=seed, evaluation_seconds=time.perf_counter() - before)
            fits.append(info)
            frame = pd.DataFrame({k: data[k][test] for k in ("group", "demo", "round", "tick", "n", "elapsed", "contact_age", "had_contact", "ct_team", "t_team", "smoke_active", "ct_elo", "t_elo", "skill_band")})
            current = data["frames"][data["end"][test]-1]
            k = data["y"].shape[-1]
            frame["ct_alive"],frame["t_alive"] = np.rint(current[:,2*k]*5).astype(int),np.rint(current[:,2*k+1]*5).astype(int)
            frame["audio_available"] = current[:,-1] > .5
            frame["ct_elo_known"],frame["t_elo_known"] = current[:,20*k+13] > .5,current[:,20*k+20] > .5
            for metric, values in scores.items():
                frame[metric] = values
            frame["model"], frame["seed"] = name, seed
            frame.to_csv(location / "test_scores.csv", index=False)
            pd.DataFrame(calibration_table(samples, data["y"][test])).to_csv(location / "calibration.csv", index=False)
            np.savez_compressed(location / "test_predictions.npz", samples=samples.astype(np.uint8),
                                truth=data["y"][test].astype(np.uint8), group=data["group"][test], tick=data["tick"][test])
            if cfg.get("diagnostics", True):
                progress("mechanism_diagnostics",model=name,seed=seed)
                from .diagnostics import collect
                collect(model, name, data, cfg, device, location, output)
            all_scores.append(frame)
            LOG.info("Evaluated %s seed=%d, energy=%.4f", name, seed, frame.energy_score.mean())
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(output / "all_test_scores.csv", index=False)
    save_json(output / "fit_costs.json", fits)
    # Finalize score columns before any figure fingerprints these files.
    from .logbundle import enrich_test_context, export_log
    enrich_test_context(output, root)
    combined = pd.read_csv(output / "all_test_scores.csv", dtype={"group": str,"demo":str})
    from .reporting import report
    progress("reports")
    report(output, combined, cfg, summary)
    if cfg.get("diagnostics", True) and (output / "diagnostic_selection.json").exists():
        from .paperplots import render
        progress("paper_figures")
        render(output)
    if limit:
        (output / "SOFTWARE_VERIFICATION_ONLY.txt").write_text("This run uses an explicitly limited subset for software verification. It is not a full-cohort experiment or evidence of model superiority.\n", encoding="utf-8")
    progress("exporting_portable_log")
    export_log(output, root, cfg, summary)
    LOG.info("Finished. Open %s", output / "comparison.csv")
