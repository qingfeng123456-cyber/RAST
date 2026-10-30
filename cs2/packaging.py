from __future__ import annotations

import json
from pathlib import Path
import zipfile


def pack(root: Path, include_raw=False, config=None, archive_path=None) -> Path:
    archive = Path(archive_path) if archive_path else root.parent / ("CS2_Belief_AutoDL_with_raw.zip" if include_raw else "CS2_Belief_AutoDL.zip")
    selected = []
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    allowed_raw = {Path("data/raw") / row["file_name"] for row in manifest} if all("file_name" in row for row in manifest) else None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in {"__pycache__", ".pytest_cache", "synthetic_smoke", ".git", ".venv_5070", "return_packages"} for part in rel.parts):
            continue
        if any(part.startswith("pytest_temp") for part in rel.parts):
            continue
        if rel.parts[0] == "outputs" or path.suffix == ".log":
            continue
        if rel.as_posix() in {"data/reserve_manifest_48.json", "config_expanded.json", "config_optimized.json", "config_module_controls.json", "config_reference_baselines.json"}:
            continue
        if rel.parts[0] == "validation" and path.suffix == ".ps1":
            continue
        if rel.parts[:2] == ("data", "raw") and not include_raw:
            continue
        if rel.parts[:2] == ("data", "raw") and allowed_raw is not None and rel not in allowed_raw:
            continue
        if rel.parts[:2] == ("data", "raw") and allowed_raw is None:
            continue
        if rel.parts[:2] == ("data", "frames"):
            # Audit JSON stays; raw per-player ground truth frame caches are unnecessary for training.
            if path.suffix == ".npz":
                continue
            if not any(path.name.startswith(str(r["match_id"])+"_") for r in manifest):
                continue
        if rel.parts[:2] == ("data", "sounds"):
            continue
        if path.suffix in {".part", ".tmp", ".zip"}:
            continue
        selected.append(path)
    if not (root / "data/processed/dataset.npz").is_file():
        raise RuntimeError("Prepare real offline data before packaging")
    summary = json.loads((root / "data/processed/summary.json").read_text(encoding="utf-8"))
    if summary["kind"] != "real_demo_radar_proxy":
        raise RuntimeError("Refusing to label a synthetic diagnostic as an upload dataset")
    upload_config = dict(config or json.loads((root / "config.json").read_text(encoding="utf-8")))
    upload_config.update(summary["data_config"])
    upload_config.pop("offline", None)
    config_text = json.dumps(upload_config, ensure_ascii=False, indent=2)
    original_config_text = (root / "config.json").read_bytes().decode("utf-8")
    if json.loads(original_config_text) == upload_config:
        # Preserve the outgoing source fingerprint when only JSON formatting differs.
        config_text = original_config_text
    upload_manifest = []
    imported_raw = []
    for row in manifest:
        row = dict(row)
        if "file_name" in row:
            relative = Path("data/raw") / row["file_name"]
            original = root / relative
        else:
            original = Path(row["local_path"])
            relative = Path("data/raw/local") / str(row["match_id"]) / original.name
            if ".." in relative.parts:
                raise ValueError("Invalid local match_id")
            imported_raw.append((original, relative))
        if include_raw and (not original.is_file() or (row.get("size_bytes") and original.stat().st_size != row["size_bytes"])):
            raise FileNotFoundError(f"Cannot create a complete raw archive: {original}")
        row["local_path"] = relative.as_posix()
        row["path"] = relative.as_posix()
        upload_manifest.append(row)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for path in selected:
            if path.relative_to(root).as_posix() in {"config.json", "data/manifest.json", "data/local_manifest.csv"}:
                continue
            # Linux must receive POSIX archive paths even when created on Windows.
            z.write(path, (Path("cs2_belief") / path.relative_to(root)).as_posix())
        z.writestr("cs2_belief/config.json", config_text)
        z.writestr("cs2_belief/data/manifest.json", json.dumps(upload_manifest, ensure_ascii=False, indent=2))
        if upload_config["source"] == "local":
            import csv
            import io
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=["path", "match_id", "map_name", "match_date", "tick_rate", "team1", "team2", "event", "team_skill_bands", "team_ranks"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{**r, "team_skill_bands": json.dumps(r.get("team_skill_bands", {})), "team_ranks": json.dumps(r.get("team_ranks", {}))} for r in upload_manifest])
            z.writestr("cs2_belief/data/local_manifest.csv", buffer.getvalue())
            if include_raw:
                for original, relative in imported_raw:
                    z.write(original, (Path("cs2_belief") / relative).as_posix())
    return archive
