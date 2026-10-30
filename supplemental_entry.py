"""One command: train missing controls and return the complete experiment."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def bootstrap(offline=False, training=True):
    requirements = {"numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<4",
                    "matplotlib": "matplotlib>=3.8,<4", "scipy": "scipy>=1.11,<2",
                    "sklearn": "scikit-learn>=1.4,<2", "threadpoolctl": "threadpoolctl>=3,<4",
                    "joblib": "joblib>=1.3,<2"}
    if training:
        requirements.update(torch="torch>=2.2", tqdm="tqdm>=4.66")
    missing = [value for key, value in requirements.items() if importlib.util.find_spec(key) is None]
    if missing and offline:
        raise RuntimeError("Offline mode: missing dependencies " + ", ".join(missing))
    if missing:
        print("Installing missing dependencies: " + ", ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="supplemental_config.json")
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Verify prepared data and old saved cohorts; do not train")
    mode.add_argument("--smoke", action="store_true", help="Tiny isolated CPU software check; never a paper result")
    mode.add_argument("--replot-only", action="store_true", help="Rebuild comparisons from saved scores, without training/GPU")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default=None)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received termination signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    cfg = {"run_name": "baseline_supplement"}
    try:
        config = Path(args.config)
        if not config.is_absolute():
            config = ROOT / config
        cfg = json.loads(config.read_text(encoding="utf-8"))
        cfg["offline"] = args.offline
        if args.device is not None:
            cfg["device"] = args.device
        if args.smoke:
            cfg["auto_project_environment"] = False
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.get("cpu_threads", 4)))
        from oneclick_environment import ensure_environment
        ensure_environment(ROOT, cfg, offline=args.offline,
                           prepare_only=args.prepare_only or args.smoke or args.replot_only)
        bootstrap(args.offline, training=not (args.prepare_only or args.replot_only))
        from supplemental_runner import execute
        execute(ROOT, cfg, smoke=args.smoke, prepare_only=args.prepare_only, replot_only=args.replot_only)
    except (Exception, KeyboardInterrupt) as exc:
        if not getattr(exc, "supplemental_bundle_handled", False):
            from supplemental_runner import package_early_failure
            package_early_failure(ROOT, cfg, exc, smoke=args.smoke)
        traceback.print_exc()
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    cli()
