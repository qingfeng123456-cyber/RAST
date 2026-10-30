"""Check portable experiment evidence hashes without importing ML libraries."""
import argparse
import hashlib
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    root = args.folder.resolve()
    manifest = json.loads((root / "log_manifest.json").read_text(encoding="utf-8"))
    missing, changed, regenerated, code = [], [], [], []
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root): raise ValueError("Manifest path escapes the log folder")
        if not path.is_file():
            missing.append(row["path"]); continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            target = regenerated if row.get("role") in {"derived_plot","live_status"} else code if row.get("role") == "code" else changed
            target.append(row["path"])
    figure_sources = []
    figure_manifest = root / "figures/figure_manifest.json"
    if figure_manifest.exists():
        for row in json.loads(figure_manifest.read_text(encoding="utf-8")).get("source_fingerprints",[]):
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root): raise ValueError("Figure source escapes the log folder")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                figure_sources.append(row["path"])
    result = {"status": ("PASS" if manifest["status"] == "completed" else manifest["status"].upper()) if not missing and not changed and not figure_sources else "EVIDENCE_MISMATCH", "run_status": manifest["status"],
              "verification_only": manifest.get("verification_only"), "training_seeds": manifest["training_seeds"],
              "missing": missing, "changed_evidence": changed, "figure_source_mismatches":figure_sources, "regenerated_figures": regenerated, "modified_code": code}
    print(json.dumps(result, indent=2))
    if missing or changed or figure_sources: raise SystemExit(1)
