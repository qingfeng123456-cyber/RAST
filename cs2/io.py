from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

LOG = logging.getLogger("cs2")


def save_json(path: Path, value) -> None:
    def encode(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, Path):
            return obj.as_posix()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(f"Unsupported JSON value: {type(obj)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode), encoding="utf-8")
    tmp.replace(path)


def get_json(url: str):
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={"User-Agent": "CS2BeliefResearch/0.1"}), timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download(url: str, path: Path, expected: int | None = None) -> Path:
    """Resume interrupted public downloads. HTTP 200 after Range restarts safely."""
    if path.exists() and (expected is None or path.stat().st_size == expected):
        return path
    if expected is not None and expected > 64 * 1024 * 1024:
        return download_chunks(url, path, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(6):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "CS2BeliefResearch/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        retry_url = url + (("&" if "?" in url else "?") + f"retry={time.time_ns()}" if attempt else "")
        try:
            with urlopen(Request(retry_url, headers=headers), timeout=120) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Unexpected Content-Range")
                base = offset if resumed else 0
                length = response.headers.get("Content-Length")
                total = base + int(length) if length else expected
                with tmp.open("ab" if resumed else "wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                size = tmp.stat().st_size
                if (total is not None and size != total) or (expected is not None and size != expected):
                    raise OSError(f"Incomplete file: {size}, expected {expected or total}")
                tmp.replace(path)
                LOG.info("Downloaded %s (%.1f MB)", path.name, size / 1e6)
                return path
        except HTTPError as exc:
            if exc.code == 416 and expected is not None and offset == expected:
                tmp.replace(path)
                return path
            if attempt == 5:
                raise
        except Exception:
            if attempt == 5:
                raise
        time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Download failed: {url}")


def download_chunks(url: str, path: Path, expected: int, chunk_size: int = 8 * 1024 * 1024) -> Path:
    """Bounded HTTP ranges avoid large-object stalls; each completed range is reusable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = path.with_suffix(path.suffix + ".chunks")
    parts.mkdir(exist_ok=True)
    def fetch(start):
        end = min(start + chunk_size, expected) - 1
        piece = parts / f"{start:012d}.bin"
        length = end - start + 1
        if piece.is_file() and piece.stat().st_size == length:
            return piece
        partial = piece.with_suffix(".part")
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == length:
                partial.replace(piece)
                return piece
            headers = {"User-Agent": "CS2BeliefResearch/0.1", "Accept-Encoding": "identity",
                       "Range": f"bytes={start + offset}-{end}"}
            chunk_url = url + ("&" if "?" in url else "?") + f"range_start={start + offset}&range_end={end}"
            if attempt:
                chunk_url += f"&retry={time.time_ns()}"
            try:
                with urlopen(Request(chunk_url, headers=headers), timeout=90) as response:
                    expected_header = f"bytes {start + offset}-{end}/{expected}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_header:
                        raise ValueError("Server did not honor the requested bounded byte range")
                    remaining = length - offset
                    with partial.open("ab") as output:
                        while remaining:
                            block = response.read(min(512 * 1024, remaining))
                            if not block:
                                raise OSError("Short ranged response")
                            output.write(block)
                            remaining -= len(block)
                if partial.stat().st_size != length:
                    raise OSError("Wrong chunk length")
                partial.replace(piece)
                return piece
            except Exception as exc:
                LOG.warning("Range retry %d/6 %s @ %.0f MB (%s)", attempt + 1, path.name, start / 1e6, type(exc).__name__)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("Range retries exhausted")
    starts = list(range(0, expected, chunk_size))
    ready = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, piece in enumerate(pool.map(fetch, starts)):
            ready.append(piece)
            if (index + 1) % 8 == 0 or index + 1 == len(starts):
                LOG.info("Download progress %s: %d/%d chunks", path.name, index + 1, len(starts))
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("wb") as output:
        for piece in ready:
            with piece.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if temp.stat().st_size != expected:
        raise OSError("Merged demo length mismatch")
    temp.replace(path)
    for piece in ready:
        piece.unlink()
    if not any(parts.iterdir()):
        parts.rmdir()
    LOG.info("Downloaded %s (%.1f MB)", path.name, expected / 1e6)
    return path


def hf_url(cfg: dict, path: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{cfg['repo']}/resolve/{cfg['revision']}/{quote(path, safe='/')}?download=true"


def acquire(root: Path, cfg: dict) -> list[dict]:
    import pandas as pd
    manifest_path = root / "data/manifest.json"
    if cfg["source"] == "local":
        csv = root / "data/local_manifest.csv"
        if not csv.exists():
            raise RuntimeError("Local source requires data/local_manifest.csv: path,match_id,map_name,match_date,tick_rate")
        records = pd.read_csv(csv, dtype={"match_id": str}).to_dict("records")
        for row in records:
            for key in ("team_skill_bands", "team_ranks"):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
                elif not isinstance(row.get(key), dict):
                    row[key] = {}
            path = Path(row["path"])
            row["local_path"] = str(path if path.is_absolute() else root / path)
            if not Path(row["local_path"]).is_file():
                raise FileNotFoundError(row["local_path"])
        records = [r for r in records if r["map_name"] in cfg["maps"]][:cfg["max_demos"]]
        save_json(manifest_path, records)
        return records
    if cfg.get("offline"):
        if not manifest_path.exists():
            raise RuntimeError("Offline data not prepared. Use the delivered data/processed cache or prepare online first.")
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [row for row in records if row["map_name"] in cfg["maps"]]
        if len(records) < cfg["max_demos"]:
            raise RuntimeError("Offline manifest has fewer demos than requested; expand the dataset online first")
        records = records[:cfg["max_demos"]]
        for row in records:
            row["local_path"] = str(root / "data/raw" / row["file_name"])
        return records
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    tree_url = f"{endpoint}/api/datasets/{cfg['repo']}/tree/{cfg['revision']}/data?limit=1000"
    tree = get_json(tree_url)
    paths = [row["path"] for row in tree if row["path"].endswith(".parquet")]
    if not paths:
        raise RuntimeError("No metadata shards found")
    def fetch(path):
        return download(hf_url(cfg, path), root / "data/metadata" / Path(path).name)
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        shards = list(pool.map(fetch, paths))
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.map_name.isin(cfg["maps"])].drop_duplicates("file_name")
    df = df.sort_values(["match_date", "match_id", "file_name"]).reset_index(drop=True)
    if len(df) < cfg["max_demos"]:
        raise RuntimeError(f"Only {len(df)} matching demos; requested {cfg['max_demos']}")
    # Time-spread sampling, not taking only one early tournament.
    import numpy as np
    retained = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(previous) <= cfg["max_demos"]:
            selected_files = {r.get("file_name") for r in previous}
            retained = np.flatnonzero(df.file_name.isin(selected_files)).tolist()
    remaining = [i for i in range(len(df)) if i not in set(retained)]
    needed = cfg["max_demos"] - len(retained)
    added = [remaining[i] for i in np.linspace(0, len(remaining)-1, needed).round().astype(int)] if needed else []
    indices = sorted(retained + added)
    records = []
    for _, row in df.iloc[indices].iterrows():
        r = {k: str(row[k]) for k in ("file_name", "match_id", "map_name", "match_date", "team1", "team2", "event", "meta_file_name", "analysis_file_name")}
        match = row["match"]
        r["tick_rate"] = float(match["tick_rate"])
        records.append(r)
    def enrich(row):
        meta_path = download(hf_url(cfg, row["meta_file_name"]), root / "data/metadata" / (row["match_id"] + "_meta.json"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        row["size_bytes"] = int(meta.get("demo_bytes", {}).get(Path(row["file_name"]).name, 0))
        row["match_url"] = meta.get("match_url", "")
        row["original_demo_url"] = meta.get("demo_url", "")
        row["local_path"] = str(root / "data/raw" / row["file_name"])
        return row
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        records = list(pool.map(enrich, records))
    total = sum(r["size_bytes"] for r in records)
    LOG.info("Selected %d map demos, %d matches, %.2f GB", len(records), len({r['match_id'] for r in records}), total / 1e9)
    if total > cfg["max_download_gb"] * 1e9:
        raise RuntimeError("Download exceeds max_download_gb; increase the explicit budget or reduce max_demos")
    save_json(manifest_path, records)
    def fetch_demo(row):
        download(hf_url(cfg, row["file_name"]), Path(row["local_path"]), row["size_bytes"] or None)
        # Sidecar is provenance only. Round state and labels come from the raw demo.
        download(hf_url(cfg, row["analysis_file_name"]), root / "data/metadata" / (Path(row["file_name"]).stem + ".analysis.json"))
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        list(pool.map(fetch_demo, records))
    return records
