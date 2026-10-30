"""AutoDL entry point. Only Python's standard library is needed to start."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def bootstrap(offline: bool) -> None:
    deps = {"numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<4",
            "pyarrow": "pyarrow>=15,<27", "sklearn": "scikit-learn>=1.4,<2",
            "matplotlib": "matplotlib>=3.8,<4", "demoparser2": "demoparser2==0.42.0",
            "torch": "torch>=2.2", "tqdm": "tqdm>=4.66",
            "scipy": "scipy>=1.11,<2", "threadpoolctl": "threadpoolctl>=3,<4"}
    missing = [requirement for module, requirement in deps.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        if offline:
            raise RuntimeError("Missing dependencies: " + ", ".join(missing) +
                             ". Install requirements.txt before --offline.")
        print("Installing missing dependencies:", ", ".join(missing), flush=True)
        if importlib.util.find_spec("torch") is None:
            subprocess.check_call([sys.executable,"-m","pip","install","torch==2.11.0",
                                   "--index-url","https://download.pytorch.org/whl/cu128"])
            missing = [requirement for requirement in missing if not requirement.startswith("torch")]
        if missing: subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def cli() -> None:
    p = argparse.ArgumentParser(description="CS2 anonymous hidden-enemy occupancy research pipeline")
    p.add_argument("--config", default="config.json")
    p.add_argument("--prepare-only", action="store_true", help="Download, parse and build the offline dataset")
    p.add_argument("--offline", action="store_true", help="Never install dependencies or download data")
    p.add_argument("--source", choices=["hf", "local", "synthetic"])
    p.add_argument("--max-demos", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default=None, help="auto, cpu, cuda or cuda:0")
    p.add_argument("--run-name", default=None)
    p.add_argument("--models", nargs="+", choices=["knn", "vanilla_transformer", "improved_transformer"])
    p.add_argument("--history-mode", choices=["full"])
    p.add_argument("--pack", action="store_true", help="Create upload ZIP after the requested pipeline finishes")
    p.add_argument("--include-raw", action="store_true", help="Include original .dem files in upload ZIP")
    p.add_argument("--rebuild", action="store_true", help="Rebuild processed data; never delete raw demos")
    args = p.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("source", "max_demos", "epochs", "seeds", "device", "run_name", "models", "history_mode"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["offline"] = args.offline
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.get("cpu_threads", 4)))
    if not args.prepare_only:
        # Keep prior failures as history when the same command resumes a run.
        from datetime import datetime, timezone
        import uuid
        prior_root = ROOT / "synthetic_smoke" if cfg["source"] == "synthetic" else ROOT
        prior_log = prior_root / "outputs" / cfg.get("run_name", "run01") / "log"
        prior_failure = prior_log / "FAILURE.json"
        if prior_failure.exists():
            history = prior_log / "failure_history"
            history.mkdir(parents=True,exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
            prior_failure.rename(history / (stamp+".json"))
    try:
        from oneclick_environment import ensure_environment
        ensure_environment(ROOT,cfg,offline=args.offline,prepare_only=args.prepare_only)
        bootstrap(args.offline)
        from cs2.pipeline import run
        run(ROOT, cfg, prepare_only=args.prepare_only, rebuild=args.rebuild)
    except (Exception,KeyboardInterrupt) as exc:
        import traceback
        status = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        failure = {"status": status, "traceback": traceback.format_exc()}
        run_root = ROOT / "synthetic_smoke" if cfg["source"] == "synthetic" else ROOT
        output = run_root / "outputs" / cfg.get("run_name", "run01")
        folder = output / "log"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (folder / "current_progress.json").write_text(json.dumps({"stage":status,"training_seeds":cfg["seeds"],
            "run_name":cfg.get("run_name"),"details":"FAILURE.json"},indent=2),encoding="utf-8")
        manifest_path = folder / "log_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(status=status,error=failure["traceback"])
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        context = output / "plot_context/data_summary.json"
        if context.exists():
            try:
                from cs2.logbundle import export_log
                export_log(output,run_root,cfg,json.loads(context.read_text(encoding="utf-8")),status=status,error=failure["traceback"])
            except Exception as export_error:
                failure["log_export_error"] = repr(export_error)
                (folder / "FAILURE.json").write_text(json.dumps(failure,indent=2),encoding="utf-8")
        raise
    if args.pack:
        from cs2.packaging import pack
        print("Upload archive:", pack(ROOT, args.include_raw, cfg), flush=True)


if __name__ == "__main__":
    cli()
