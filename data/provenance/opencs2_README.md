---
license: cc-by-4.0
task_categories:
  - other
language:
  - en
tags:
  - counter-strike
  - csgo
  - cs2
  - esports
  - hltv
  - demos
pretty_name: "HLTV CS2 Demos Dataset"
size_categories:
  - 100K<n<1M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/metadata-*.parquet
---

# HLTV CS2 Demos Dataset

Counter-Strike 2 match demos scraped from [HLTV.org](https://www.hltv.org/results)
plus a compact per-map analysis JSON. Each row of the metadata Parquet is
**one `.dem` file (one played CS2 map)**; a best-of-3 match contributes 2
or 3 rows depending on whether it went 2-0 or 2-1.

- Parquet holds everything you typically filter on: `map_name`,
  `patch_version`, `rounds_played`, per-player `kast` / `adr` / `rating`,
  every kill tick with weapon + headshot, every round's winner + end reason.
- `.dem` binaries live alongside as loose files; each match subfolder also
  carries a `meta.json` (HLTV sidecar) and a `<demo>.analysis.json` (the
  exact JSON that was embedded into the Parquet row, so the shard is
  self-describing if you grab it via the filesystem).

## Repository layout

```
data/                                  # metadata Parquet (self-describing)
  metadata-<machine>-<uuid>.parquet
demos/
  shard-<machine>-<uuid>/
    <match_id>/
      meta.json                        # HLTV sidecar (teams, event, demo id …)
      <demo1>.dem                      # bo3 map 1
      <demo1>.analysis.json            # per-map analysis (full)
      <demo2>.dem
      <demo2>.analysis.json
      <demo3>.dem                      # absent if bo3 ended 2-0
      <demo3>.analysis.json
```

Shard names carry the producing machine id + a UUID so any number of
machines can upload to the dataset in parallel without colliding.

## Schema (one row per `.dem`)

Promoted top-level columns let you filter fast:

| column | type | source |
|---|---|---|
| `file_name` | string | `.dem` path inside the repo |
| `analysis_file_name` | string | `<demo>.analysis.json` path |
| `meta_file_name` | string | per-match `meta.json` path |
| `match_id` | string | HLTV numeric id (same across the bo3 maps) |
| `event`, `team1`, `team2` | string | HLTV `/results` row |
| `score1`, `score2` | int32 | map score of the full bo3 |
| `format` | string | `bo3` / `bo5` / map name for bo1s |
| `match_date` | timestamp | UTC, from HLTV |
| `map_index` | int32 | 1, 2, 3 position of this map in the bo3 |
| `map_name` | string | e.g. `de_overpass` |
| `patch_version` | string | CS2 build, e.g. `"14141"` = `1.41.4.1` |
| `rounds_played` | int32 | actual rounds this map went |
| `winner_side` | string | `ct` / `t` |

Native nested columns for the full analysis:

| column | type |
|---|---|
| `header` | `struct<map_name, patch_version, server_name, network_protocol, demo_version_name>` |
| `match` | `struct<map_name, patch_version, tick_rate, first_tick, last_tick, total_ticks, rounds_played, score_ct, score_t, winner_side>` |
| `teams` | `list<struct<side_start, players: list<struct<name, steamid>>>>` |
| `players` | `list<struct<name, steamid, side, n_rounds, kills, headshots, deaths, assists, hs_rate, adr, dmg, kast, kast_rounds, rating, impact>>` |
| `rounds` | `list<struct<round_num, start, freeze_end, end, official_end, winner, reason, bomb_plant, bomb_site, duration_ticks>>` |
| `kills` | `list<struct<tick, round_num, attacker_name, attacker_steamid, attacker_side, victim_name, victim_steamid, victim_side, assister_name, assister_steamid, weapon, headshot, distance, noscope, thrusmoke, penetrated, attackerblind, dmg_health, hitgroup>>` |
| `meta` | `struct` mirror of the on-disk `meta.json` |

## Query with DuckDB (no download)

The Hub supports `hf://` paths in DuckDB `≥ 0.10.3`, so you can run SQL
directly against the parquet without fetching the `.dem`s. See the [HF
DuckDB docs](https://huggingface.co/docs/hub/datasets-duckdb).

```sql
-- Quick peek
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet' LIMIT 3;

-- All de_overpass maps on patch 14141
SELECT match_id, team1, team2, map_name, score_ct, score_t, rounds_played
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet'
WHERE patch_version = '14141' AND map_name = 'de_overpass';

-- Dot-access into the nested `match` struct
SELECT match_id, match.tick_rate, match.winner_side
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet';

-- UNNEST kills to one row per kill
SELECT match_id, k.weapon, k.headshot, k.distance
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet', UNNEST(kills) AS t(k)
WHERE k.headshot;

-- Inline list predicates (avoid UNNEST when you want aggregates)
SELECT match_id, map_name,
       len(kills) AS n_kills,
       len(filter(kills, x -> x.headshot)) AS n_hs,
       list_avg(list_transform(players, p -> p.adr)) AS avg_adr
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet';
```

Also works from DuckDB's Python client:

```python
import duckdb
duckdb.sql("SELECT count(*) FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet'")
```

Or via the Hub's auto-converted parquet branch (useful when the dataset
gets huge and the Viewer shards it for you):

```sql
SELECT count(*) FROM 'hf://datasets/blanchon/cs2_dataset_demo@~parquet/**/*.parquet';
```

## Stream with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("blanchon/cs2_dataset_demo", split="train", streaming=True)

# Take the first 10 rows — metadata only, no .dem transferred
for row in ds.take(10):
    print(row["match_id"], row["map_name"], row["patch_version"],
          row["match"]["score_ct"], row["match"]["score_t"])

# Filter in the stream: patch 14141 + tier-1 events
ds = ds.filter(
    lambda r: r["patch_version"] == "14141" and "IEM" in (r["event"] or "")
)
```

## Stream single files with `HfFileSystem`

For random access (read one Parquet row group, one `.analysis.json`
without downloading neighbours, etc.) use
[`HfFileSystem`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_file_system):

```python
from huggingface_hub import hffs
import json, pyarrow.parquet as pq

# Read one .analysis.json
with hffs.open("datasets/blanchon/cs2_dataset_demo/demos/shard-<id>/<match_id>/<demo>.analysis.json") as f:
    analysis = json.load(f)
print(analysis["match"]["rounds_played"], len(analysis["kills"]))

# List every .analysis.json in a shard
for p in hffs.glob("datasets/blanchon/cs2_dataset_demo/demos/shard-*/*/*.analysis.json"):
    print(p)

# Row-group-level streaming on the metadata Parquet
with pq.ParquetFile("hf://datasets/blanchon/cs2_dataset_demo/data/metadata-xyz.parquet") as pf:
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i)
        # process tbl …
```

## Partial download with the `hf` CLI

The [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) ships
with `huggingface_hub`. `hf_transfer` is the Rust-backed uploader/downloader
that saturates a gigabit link per file.

```bash
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Metadata only (fast, ~tens of MB for the whole dataset)
hf download blanchon/cs2_dataset_demo --repo-type dataset --include "data/*.parquet"

# One match: all files under a given <match_id>/
hf download blanchon/cs2_dataset_demo --repo-type dataset     --include "demos/shard-*/2393304/*"

# One shard (one machine's batch) worth of data
hf download blanchon/cs2_dataset_demo --repo-type dataset     --include "demos/shard-machineA-*/**"

# Only the .analysis.json sidecars (tens of KB each, no .dem)
hf download blanchon/cs2_dataset_demo --repo-type dataset     --include "demos/shard-*/**/*.analysis.json"
```

The equivalent in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="blanchon/cs2_dataset_demo", repo_type="dataset",
    allow_patterns=["data/*", "demos/shard-*/2393304/*"],
)
```

## Example workflow: metadata-first, fetch on demand

```python
from huggingface_hub import HfFileSystem, hf_hub_download
import duckdb

# 1. Decide which matches you care about via SQL — stays on the Hub
paths = duckdb.sql(f'''
    SELECT file_name FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet'
    WHERE patch_version = '14141'
      AND map_name = 'de_overpass'
      AND score_ct + score_t >= 24
''').fetchall()

# 2. Fetch only the matching .dem files
for (rel,) in paths:
    local = hf_hub_download(repo_id="blanchon/cs2_dataset_demo", repo_type="dataset", filename=rel)
    print(local)
```

## Collection & licensing

Demos are mirrored from HLTV's public `/download/demo/<id>` endpoint, which
hosts VODs provided by tournament organizers. Downstream users are
responsible for respecting the original terms from those tournaments. This
dataset exists for research / analytics / ML use on competitive CS2 play.
