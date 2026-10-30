from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np

from .io import save_json

LOG = logging.getLogger("cs2")
SCHEMA = 9
PARSE_SCHEMA = 7
PUBLIC_CLOCK_PROTOCOL = "round-start grid + freeze + public deaths + friendly smoke lifecycle events"
CT_AUX_NAMES = ["health", "armor", "equipment", "velocity_x", "velocity_y", "yaw_sin", "yaw_cos",
                "flash_duration", "smoke", "flashbang", "he_grenade", "molotov", "defuser", "scoped"]
DATA_KEYS = ["source", "repo", "revision", "maps", "max_demos", "tick_rate", "sample_seconds",
             "history_steps", "target_stride", "max_round_seconds", "regions", "region_z_weight",
             "train_fraction", "validation_fraction", "minimum_match_groups", "skill_lag_days", "default_skill_band"]


def data_config(cfg):
    """Cache meaning includes an explicit new band scale, without changing old contracts."""
    result = {key: cfg[key] for key in DATA_KEYS}
    if "skill_band_scale_max" in cfg:
        from .skill import skill_band_protocol
        skill_band_protocol(cfg["skill_band_scale_max"])
        result["skill_band_scale_max"] = cfg["skill_band_scale_max"]
    return result


def spotted_ids(value) -> set[str] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Radar field must decode to a list, got {type(value)}")
    return {str(int(v)) for v in value if v is not None and int(v) != 0}


def measure_tick_rate(clock) -> float | None:
    if not hasattr(clock, "columns") or not {"tick", "game_time"}.issubset(clock.columns):
        return None
    clock = clock[["tick", "game_time"]].dropna().drop_duplicates("tick").sort_values("tick")
    if len(clock) < 3:
        return None
    dt, dk = np.diff(clock.game_time.to_numpy(float)), np.diff(clock.tick.to_numpy(float))
    valid = (dt > 0) & (dk > 0)
    ratios = dk[valid] / dt[valid]
    if len(ratios) < 2:
        return None
    rate = float(np.median(ratios))
    if not 16 <= rate <= 256 or np.mean(abs(ratios - rate) / rate < .03) < .8:
        return None
    return rate


def friendly_smoke_lifetimes(detonations, expirations):
    """Pair expiry with one CT detonation, never with a reused entity's later life.

    Enemy detonations are used only as pairing boundaries, not as observations.
    An unmatched friendly event remains open until the round ends; an enemy
    event cannot terminate it or create an observation tick.
    """
    required = {"user_team_num", "tick", "entityid", "x", "y", "z"}
    if len(detonations) == 0:
        return []
    if not required.issubset(detonations.columns):
        raise ValueError("Smoke events require team and entity metadata for the public observation clock")
    events = detonations.sort_values("tick", kind="stable")
    has_expiry = {"entityid", "tick"}.issubset(expirations.columns)
    result = []
    for entity, lifecycle in events.groupby("entityid", sort=False):
        rows = list(lifecycle.itertuples(index=False))
        expiry_ticks = (np.sort(expirations.loc[expirations.entityid == entity, "tick"].to_numpy(np.int64))
                        if has_expiry else np.empty(0, np.int64))
        for i, row in enumerate(rows):
            if row.user_team_num != 3:
                continue
            next_detonation = int(rows[i + 1].tick) if i + 1 < len(rows) else np.iinfo(np.int64).max
            matching = expiry_ticks[(expiry_ticks > int(row.tick)) & (expiry_ticks <= next_detonation)]
            expiry = int(matching[0]) if len(matching) else np.iinfo(np.int64).max
            result.append((int(row.tick), expiry, [row.x, row.y, row.z]))
    return sorted(result, key=lambda event: event[0])


def public_event_ticks(deaths, smoke_lifetimes):
    """Only kill feed events and friendly smoke changes may refine the grid."""
    ticks = deaths["tick"].to_numpy(np.int64).tolist() if "tick" in deaths else []
    for start, expiry, _ in smoke_lifetimes:
        ticks.append(start)
        if expiry != np.iinfo(np.int64).max:
            ticks.append(expiry)
    return np.unique(np.asarray(ticks, np.int64))


def parse_demo(root: Path, record: dict, cfg: dict) -> Path:
    """Store truth and observation flags separately; do not preserve enemy IDs."""
    from demoparser2 import DemoParser
    path = Path(record["local_path"])
    cache = root / "data/frames" / (str(record["match_id"]) + "_" + path.stem + ".npz")
    audit_path = cache.with_suffix(".audit.json")
    parse_contract = {k: cfg[k] for k in ("sample_seconds", "max_round_seconds", "tick_rate")}
    parse_contract["metadata_tick_rate"] = record.get("tick_rate")
    parse_contract["schema"] = PARSE_SCHEMA
    parse_contract["observation_clock"] = PUBLIC_CLOCK_PROTOCOL
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("parse_contract") == parse_contract and audit.get("raw_bytes") == path.stat().st_size:
            return cache
    LOG.info("Parsing raw demo: %s", path.name)
    parser = DemoParser(str(path))
    header = parser.parse_header()
    if header.get("map_name") != record["map_name"]:
        raise ValueError(f"Map mismatch: {header.get('map_name')} vs {record['map_name']}")
    freezes = parser.parse_event("round_freeze_end", other=["game_time"])
    starts = parser.parse_event("round_start")
    ends = parser.parse_event("round_end")
    detonations = parser.parse_event("smokegrenade_detonate", player=["team_num"])
    expirations = parser.parse_event("smokegrenade_expired")
    deaths = parser.parse_event("player_death")
    if not hasattr(freezes, "columns") or "tick" not in freezes or len(freezes) < 2:
        raise ValueError("No usable round_freeze_end events")
    reported_rate = record.get("tick_rate")
    measured_rate = measure_tick_rate(freezes)
    rate = cfg.get("tick_rate") or measured_rate or reported_rate
    if rate is None or not np.isfinite(float(rate)) or not 16 <= float(rate) <= 256:
        raise ValueError("A verified tick_rate is required in the manifest or config; no silent 128/64 guess")
    rate = float(rate)
    if measured_rate is not None and abs(measured_rate - rate) / rate > .03:
        raise ValueError(f"Explicit tick_rate {rate} contradicts demo clock {measured_rate}")
    if reported_rate is not None and abs(float(reported_rate) - rate) / rate > .03:
        LOG.warning("Correcting metadata clock %.1f to measured %.1f for %s", float(reported_rate), rate, path.name)
    step = int(round(rate * cfg["sample_seconds"]))
    freeze_ticks = np.unique(freezes["tick"].to_numpy(dtype=np.int64))
    end_ticks = np.unique(ends["tick"].to_numpy(dtype=np.int64))
    start_ticks = np.unique(starts["tick"].to_numpy(dtype=np.int64))
    smoke_lifetimes = friendly_smoke_lifetimes(detonations, expirations)
    event_ticks = public_event_ticks(deaths, smoke_lifetimes)
    boundaries = []
    for i, start in enumerate(freeze_ticks):
        next_start = freeze_ticks[i + 1] if i + 1 < len(freeze_ticks) else np.iinfo(np.int32).max
        candidates = end_ticks[(end_ticks > start) & (end_ticks < next_start)]
        if not len(candidates):
            continue
        end = min(int(candidates[0]), int(start + rate * cfg["max_round_seconds"]))
        possible = start_ticks[(start_ticks <= start) & (start_ticks > (freeze_ticks[i-1] if i else -1))]
        round_start = int(possible[-1]) if len(possible) else int(start)
        ticks = np.unique(np.concatenate([np.arange(round_start, end, step, dtype=np.int64),
                                         [int(start)], event_ticks[(event_ticks >= round_start) & (event_ticks < end)] + 1]))
        ticks = ticks[ticks < end]
        if len(ticks) >= cfg["history_steps"]:
            boundaries.append((i, int(start), round_start, ticks))
    if not boundaries:
        raise ValueError("No complete live-round windows")
    wanted = np.unique(np.concatenate([b[3] for b in boundaries])).tolist()
    fields = ["X", "Y", "Z", "team_num", "is_alive", "is_coach_team", "approximate_spotted_by", "is_warmup_period", "game_time",
              "health", "armor_value", "current_equip_value", "yaw", "flash_duration", "has_defuser", "is_scoped", "inventory", "team_clan_name"]
    df = parser.parse_ticks(fields, ticks=wanted)
    required = {"tick", "steamid", *fields}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing mandatory columns: {required - set(df.columns)}")
    # Verify the metadata clock where the demo exposes a usable game_time clock.
    sampled_rate = measure_tick_rate(df)
    if sampled_rate is not None and abs(sampled_rate - rate) / rate > .03:
        raise ValueError(f"Sampled clock {sampled_rate} contradicts chosen tick_rate {rate}")
    measured_rate = sampled_rate or measured_rate
    groups = {int(t): frame for t, frame in df.groupby("tick", sort=False)}
    lists = {k: [] for k in ("ct_xyz", "ct_alive", "t_xyz", "t_alive", "t_seen", "tick", "round", "elapsed", "ct_aux", "since_start", "ct_team", "t_team", "smoke_xyz", "smoke_age", "smoke_mask")}
    skipped, radar_null, radar_total, spotted_total = 0, 0, 0, 0
    round_counts = {}
    discarded_rounds = 0
    for round_id, freeze_tick, round_start, ticks in boundaries:
        valid_round = []
        previous_ct = None
        incomplete = False
        smoke_events = [event for event in smoke_lifetimes if round_start <= event[0] <= int(ticks[-1])]
        for tick in ticks:
            frame = groups.get(int(tick))
            if frame is None or frame.is_warmup_period.isna().any() or frame.is_warmup_period.any():
                skipped += 1
                incomplete = True
                continue
            frame = frame[frame.team_num.isin([2, 3]) & frame.steamid.notna() & (frame.is_coach_team == 0)].drop_duplicates("steamid")
            ct, ts = frame[frame.team_num == 3].sort_values("steamid"), frame[frame.team_num == 2]
            if len(ct) != 5 or len(ts) != 5 or frame.is_alive.isna().any():
                skipped += 1
                incomplete = True
                continue
            alive_ct = ct[ct.is_alive.astype(bool)]
            if not len(alive_ct):
                break  # No remaining CT decision maker; no subsequent targets.
            observers = {str(int(x)) for x in alive_ct.steamid}
            seen, missing = [], False
            for row in ts.itertuples(index=False):
                ids = spotted_ids(row.approximate_spotted_by)
                radar_total += 1
                if ids is None:
                    radar_null += 1
                    missing = True
                exposed = bool(ids and ids.intersection(observers) and row.is_alive)
                seen.append(exposed)
                spotted_total += int(exposed)
            if missing:
                skipped += 1
                incomplete = True
                continue
            ct_pos, t_pos = ct[["X", "Y", "Z"]].to_numpy(np.float32), ts[["X", "Y", "Z"]].to_numpy(np.float32)
            ct_alive, t_alive = ct.is_alive.to_numpy(bool), ts.is_alive.to_numpy(bool)
            if not np.isfinite(ct_pos[ct_alive]).all() or not np.isfinite(t_pos[t_alive]).all():
                skipped += 1
                incomplete = True
                continue
            auxiliary = []
            ids = ct.steamid.to_numpy()
            velocity = np.zeros((5, 2), np.float32)
            if previous_ct is not None:
                old_ids, old_pos, old_alive, old_tick = previous_ct
                consistent = (ids == old_ids) & ct_alive & old_alive
                velocity[consistent] = (ct_pos[consistent, :2] - old_pos[consistent, :2]) / ((tick - old_tick) / rate)
            previous_ct = (ids, ct_pos, ct_alive, tick)
            for player_index, row in enumerate(ct.itertuples(index=False)):
                inventory = [str(item).lower() for item in row.inventory] if isinstance(row.inventory, (list, tuple, np.ndarray)) else []
                count = lambda text: sum(text in item for item in inventory)
                yaw = np.deg2rad(float(row.yaw or 0))
                auxiliary.append([row.health / 100, row.armor_value / 100, row.current_equip_value / 10000,
                                  float(np.clip(velocity[player_index, 0] / 300, -3, 3)),
                                  float(np.clip(velocity[player_index, 1] / 300, -3, 3)),
                                  np.sin(yaw), np.cos(yaw), min(float(row.flash_duration), 25) / 5,
                                  count("smoke"), count("flashbang") / 2, count("he grenade"),
                                  count("molotov") + count("incendiary"), float(row.has_defuser), float(row.is_scoped)])
            auxiliary = np.nan_to_num(np.asarray(auxiliary, np.float32))
            auxiliary[~ct_alive] = 0
            active_smokes = [smoke for smoke in smoke_events if smoke[0] <= tick < smoke[1]]
            if len(active_smokes) > 10:
                raise ValueError("Unexpected >10 friendly active smoke events; audit instead of truncating")
            smoke_xyz, smoke_age, smoke_mask = np.zeros((10, 3), np.float32), np.zeros(10, np.float32), np.zeros(10, bool)
            for si, (detonate, _, xyz) in enumerate(active_smokes):
                smoke_xyz[si], smoke_age[si], smoke_mask[si] = xyz, (tick - detonate) / rate, True
            ct_name = str(ct.team_clan_name.mode().iloc[0]) if len(ct.team_clan_name.mode()) else "unknown"
            t_name = str(ts.team_clan_name.mode().iloc[0]) if len(ts.team_clan_name.mode()) else "unknown"
            valid_round.append((ct_pos, ct_alive, t_pos, t_alive, seen, int(tick), round_id, (tick - freeze_tick) / rate, auxiliary,
                                (tick - round_start) / rate, ct_name, t_name, smoke_xyz, smoke_age, smoke_mask))
        if incomplete:
            discarded_rounds += 1
            continue  # Never silently reset or truncate an incomplete round prefix.
        round_counts[round_id] = len(valid_round)
        for record_values in valid_round:
            for key, value in zip(lists, record_values):
                lists[key].append(value)
    if not lists["tick"] or spotted_total == 0:
        raise ValueError("Radar proxy has no positive CT sightings; audit manually before proceeding")
    arrays = {k: np.asarray(v) for k, v in lists.items()}
    arrays["tick_rate"] = np.asarray(rate)
    arrays["ct_xyz"] = np.nan_to_num(arrays["ct_xyz"])
    arrays["t_xyz"] = np.nan_to_num(arrays["t_xyz"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    audit = {"parse_contract": parse_contract, "raw_bytes": path.stat().st_size,
             "header": header, "metadata_tick_rate": reported_rate, "measured_tick_rate": measured_rate,
             "effective_tick_rate": rate,
             "clock_note": "demo game_time measured first; metadata fallback only if demo clock unavailable",
             "observation_clock": PUBLIC_CLOCK_PROTOCOL,
             "frames": len(arrays["tick"]), "rounds": sum(v > 0 for v in round_counts.values()),
             "skipped_frames": skipped, "radar_null_fraction": radar_null / max(1, radar_total),
             "incomplete_rounds_excluded": discarded_rounds,
             "radar_positive_fraction": spotted_total / max(1, radar_total),
             "observation_protocol": "intersection of approximate_spotted_by with currently alive CT steamids; identifiers discarded after per-frame fusion",
             "round_frames": round_counts}
    save_json(audit_path, audit)
    LOG.info("Parsed %d frames; radar-positive fraction %.3f", audit["frames"], audit["radar_positive_fraction"])
    return cache


def chronological_split(records: list[dict], cfg: dict) -> dict[str, str]:
    import pandas as pd
    dates = {}
    for row in records:
        group = str(row["match_id"])
        date = pd.to_datetime(row["match_date"], utc=True)
        if pd.isna(date):
            raise ValueError(f"Missing date for match {group}")
        dates[group] = min(dates.get(group, date), date)
    ordered = sorted(dates, key=lambda g: (dates[g], g))
    if len(ordered) < cfg["minimum_match_groups"]:
        raise ValueError(f"Need at least {cfg['minimum_match_groups']} independent match groups, got {len(ordered)}")
    ntrain = max(1, int(len(ordered) * cfg["train_fraction"]))
    nval = max(1, int(len(ordered) * cfg["validation_fraction"]))
    if ntrain + nval >= len(ordered):
        raise ValueError("Split leaves no test matches")
    return {g: "train" if i < ntrain else "val" if i < ntrain + nval else "test" for i, g in enumerate(ordered)}


def fit_regions(frame_records: list[tuple[dict, Path]], splits: dict, cfg: dict) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    positions = []
    for record, path in frame_records:
        if splits[str(record["match_id"])] != "train":
            continue
        with np.load(path, allow_pickle=False) as z:
            positions.extend([z["ct_xyz"][z["ct_alive"]], z["t_xyz"][z["t_alive"]]])
    xyz = np.concatenate(positions)
    scale = np.array([1., 1., cfg["region_z_weight"]], dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=cfg["regions"], random_state=1729, n_init=10, batch_size=4096)
    km.fit(xyz[::max(1, len(xyz) // 150000)] * scale)
    # Lexicographic spatial ordering gives interpretable, reproducible region indices.
    centers = km.cluster_centers_[np.lexsort((km.cluster_centers_[:, 1], km.cluster_centers_[:, 0]))]
    return {"centers_scaled": centers.tolist(), "scale": scale.tolist(), "fit_split": "train",
            "method": "training-only 3D coordinate k-means, not navmesh or named tactical zones",
            "map": cfg["maps"][0]}


def assign_regions(xyz: np.ndarray, regions: dict) -> np.ndarray:
    centers, scale = np.asarray(regions["centers_scaled"]), np.asarray(regions["scale"])
    return np.square(xyz[..., None, :] * scale - centers).sum(-1).argmin(-1)


def feature_frame(ct_xyz, ct_alive, t_xyz, t_alive, t_seen, elapsed, regions: dict, ct_aux=None):
    """Only the exposed subset of t_xyz is ever passed to coordinate quantization."""
    k = len(regions["centers_scaled"])
    exposed = np.asarray(t_seen, bool) & np.asarray(t_alive, bool)
    ct_counts = np.bincount(assign_regions(ct_xyz[ct_alive], regions), minlength=k)
    visible_counts = np.bincount(assign_regions(t_xyz[exposed], regions), minlength=k)
    hidden = int(np.sum(t_alive) - exposed.sum())
    x = np.concatenate([ct_counts / 5, visible_counts / 5,
                        [np.sum(ct_alive) / 5, np.sum(t_alive) / 5, exposed.sum() / 5,
                         hidden / 5, elapsed / 120, 1.0]]).astype(np.float32)
    if ct_aux is not None:
        extra = np.zeros((k, len(CT_AUX_NAMES)), np.float32)
        np.add.at(extra, assign_regions(ct_xyz[ct_alive], regions), ct_aux[ct_alive] / 5)
        x = np.concatenate([x, extra.ravel()])
    return x, hidden


def build_dataset(root: Path, records: list[dict], cfg: dict, *, frame_records=None, cached_sounds=None) -> dict:
    if len(cfg["maps"]) != 1:
        raise ValueError("Run one map per experiment; use a separate package/config for the second map")
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    from .skill import build_skill_context, skill_features, skill_band_note, skill_band_protocol
    skill = build_skill_context(root, records, cfg)
    frame_records = frame_records if frame_records is not None else [(r, parse_demo(root, r, cfg)) for r in records]
    splits = chronological_split(records, cfg)
    regions = fit_regions(frame_records, splits, cfg)
    save_json(out / "regions.json", regions)
    save_json(out / "split_matches.json", splits)
    cols = {key: [] for key in ("start", "end", "y", "n", "group", "demo", "round", "tick", "split", "elapsed", "contact_age", "had_contact", "ct_team", "t_team", "smoke_active", "ct_elo", "t_elo", "skill_band")}
    h, k = cfg["history_steps"], cfg["regions"]
    zero_hidden, all_frames, round_starts = 0, [], []
    # Stable regional schema: own count, exposed count, own attributes, smoke,
    # contact memory. Global channels include population, time and skill masks.
    regional_dim = 2 + len(CT_AUX_NAMES) + 8
    global_dim = 6 + 2 + 14 + 1
    layout = {"base_global_start": 2*k, "availability_index": 2*k+5,
              "aux_start": 2*k+6, "aux_dim": len(CT_AUX_NAMES),
              "smoke_start": 2*k+6+k*len(CT_AUX_NAMES), "memory_start": 2*k+6+k*len(CT_AUX_NAMES)+2*k,
              "extra_global_start": 2*k+6+k*len(CT_AUX_NAMES)+4*k,
              "skill_start": 2*k+6+k*len(CT_AUX_NAMES)+4*k+2,
              "acoustic_start": 20*k+22,
              "regional_dim": regional_dim, "global_dim": global_dim}
    audio_audits = []
    for record, path in frame_records:
        from .audio import parse_footsteps, footstep_features
        sound = cached_sounds[path.stem] if cached_sounds is not None else parse_footsteps(root, record)
        audio_available = int(sound.get("available", 1))
        audio_audits.append({"match_id": str(record["match_id"]), "available": bool(audio_available), "footstep_events": len(sound["tick"])})
        sound_regions = assign_regions(sound["xyz"], regions)
        with np.load(path, allow_pickle=False) as cache:
            # Decompress once. NpzFile re-decompresses an array on every lookup.
            z = {key: cache[key] for key in cache.files}
            previous_round, last_contact, last_region_contact = None, None, np.full(k, np.nan)
            for i in range(len(z["tick"])):
                rid, t = int(z["round"][i]), float(z["elapsed"][i])
                if rid != previous_round:
                    prefix_start, last_contact, last_region_contact = len(all_frames), None, np.full(k, np.nan)
                    round_starts.append(prefix_start)
                    previous_sound_tick = int(z["tick"][i]) - 1
                    last_heard = np.full(k, np.nan)
                previous_round = rid
                x, hidden = feature_frame(*(z[key][i] for key in ("ct_xyz", "ct_alive", "t_xyz", "t_alive", "t_seen")), t, regions, z["ct_aux"][i])
                if z["t_seen"][i].any():
                    last_contact = t
                    seen_regions = assign_regions(z["t_xyz"][i][z["t_seen"][i] & z["t_alive"][i]], regions)
                    last_region_contact[seen_regions] = t
                smoke = np.zeros((k, 2), np.float32)
                active = z["smoke_mask"][i]
                if active.any():
                    indices = assign_regions(z["smoke_xyz"][i][active], regions)
                    np.add.at(smoke[:, 0], indices, 1/5)
                    for si, age in zip(indices, z["smoke_age"][i][active]):
                        smoke[si, 1] = max(smoke[si, 1], min(age / 20, 3))
                ever = np.isfinite(last_region_contact)
                memory = np.stack([np.where(ever, np.minimum((t - np.nan_to_num(last_region_contact)) / 120, 3), 0), ever], -1).astype(np.float32)
                covariates, labels, elos = skill_features(record, z["ct_team"][i], z["t_team"][i], skill,
                                                        cfg["default_skill_band"], cfg.get("skill_band_scale_max", 3))
                extra_globals = np.array([z["since_start"][i]/240, float(t >= 0)], np.float32)
                acoustic = footstep_features(sound["tick"], sound["side"], sound_regions,
                                             previous_sound_tick, int(z["tick"][i]), last_heard, float(z["tick_rate"]), k)
                previous_sound_tick = int(z["tick"][i])
                all_frames.append(np.concatenate([x, smoke.ravel(), memory.ravel(), extra_globals, covariates, acoustic.ravel(), [audio_available]]))
                prefix_length = len(all_frames) - prefix_start
                # Dense endpoint augmentation always keeps the ENTIRE earlier prefix.
                if prefix_length < h or t < cfg.get("minimum_live_seconds", 2) or i % cfg["target_stride"]:
                    continue
                if hidden == 0:
                    zero_hidden += 1
                    continue
                truth_mask = z["t_alive"][i] & ~z["t_seen"][i]
                y = np.bincount(assign_regions(z["t_xyz"][i][truth_mask], regions), minlength=k)
                assert y.sum() == hidden and 1 <= hidden <= 5
                values = [prefix_start, len(all_frames), y, hidden, str(record["match_id"]), path.stem,
                          rid, int(z["tick"][i]), splits[str(record["match_id"])], t,
                          t - last_contact if last_contact is not None else -1.0, last_contact is not None,
                          *labels, int(active.sum()), *elos, cfg["default_skill_band"]]
                for key, value in zip(cols, values):
                    cols[key].append(value)
    arrays = {key: np.asarray(v) for key, v in cols.items()}
    arrays["frames"] = np.asarray(all_frames, np.float32)
    arrays["round_starts"] = np.asarray(round_starts, np.int64)
    for split in ("train", "val", "test"):
        if np.sum(arrays["split"] == split) < 20:
            raise ValueError(f"Insufficient {split} samples; do not train on a degenerate split")
    np.savez_compressed(out / "dataset.npz", **arrays)
    summary = {"schema": SCHEMA, "kind": "real_demo_radar_proxy", "data_config": data_config(cfg),
               "maps": cfg["maps"], "samples": len(arrays["n"]), "demos": len(records),
               "match_groups": len(splits), "zero_hidden_windows_excluded": zero_hidden,
               "splits": {s: {"samples": int(np.sum(arrays["split"] == s)),
                               "matches": sum(v == s for v in splits.values())} for s in ("train", "val", "test")},
               "feature_layout": layout, "features_per_frame": arrays["frames"].shape[-1],
               "history_protocol": "full sampled round_start-to-cutoff prefix, buy phase included; no sliding window or history dropout",
               "observation_clock": PUBLIC_CLOCK_PROTOCOL,
               "ct_aux_names": CT_AUX_NAMES,
               "limitations": ["radar proxy is not full human perception", "idealized team-level per-frame observation fusion", "no persistent enemy identities", "regions are spatial clusters, not tactical labels", "not a causal or win-rate study"]}
    audits = [json.loads(path.with_suffix(".audit.json").read_text(encoding="utf-8")) for _, path in frame_records]
    summary["rounds"] = sum(a["rounds"] for a in audits)
    summary["frames"] = sum(a["frames"] for a in audits)
    summary["clock_corrections"] = sum(a["metadata_tick_rate"] is not None and abs(float(a["metadata_tick_rate"]) - a["effective_tick_rate"]) > 1 for a in audits)
    summary["history_with_radar_sighting_fraction"] = float(arrays["had_contact"].mean())
    summary["prefix_length"] = {"min": int((arrays["end"] - arrays["start"]).min()), "max": int((arrays["end"] - arrays["start"]).max()), "median": float(np.median(arrays["end"] - arrays["start"]))}
    summary["feature_frames"] = len(all_frames)
    summary["skill_band_counts"] = {str(b): int((arrays["skill_band"] == b).sum()) for b in np.unique(arrays["skill_band"])}
    summary["skill_band_protocol"] = skill_band_protocol(cfg.get("skill_band_scale_max", 3))
    summary["skill_note"] = skill_band_note(summary["skill_band_counts"], cfg.get("skill_band_scale_max", 3))
    summary["smoke_note"] = "Friendly smoke detonation/expiry history only; no hidden enemy smoke positions, volumetric vision reconstruction or movement blockade."
    summary["audio_note"] = "All logged player_sound step=true events assumed heard and localized to training-fitted regions. Idealized team audio: no walls, distance attenuation, masking, HRTF or hearing uncertainty. No enemy identifiers or exact sound coordinates reach the model."
    summary["audio_audits"] = audio_audits
    summary["audio_demos_available"] = sum(a["available"] for a in audio_audits)
    summary["hidden_population_counts"] = {str(i): int(np.sum(arrays["n"] == i)) for i in range(1, 6)}
    summary["date_range"] = [min(str(r["match_date"]) for r in records), max(str(r["match_date"]) for r in records)]
    import pandas as pd
    pd.DataFrame([{**{key: record.get(key) for key in ("match_id", "file_name", "match_date", "team1", "team2", "event")},
                   **{key: audit.get(key) for key in ("metadata_tick_rate", "measured_tick_rate", "effective_tick_rate", "frames", "rounds", "radar_positive_fraction", "radar_null_fraction", "skipped_frames", "incomplete_rounds_excluded")}}
                  for (record, _), audit in zip(frame_records, audits)]).to_csv(out / "demo_audit.csv", index=False)
    save_json(out / "summary.json", summary)
    card = ("# Prepared dataset\n\n"
            f"Source: {cfg['repo']} at revision {cfg['revision']}.\n\n"
            f"Map: {cfg['maps'][0]}; demos: {len(records)}; match groups: {len(splits)}; "
            f"rounds: {summary['rounds']}; windows: {len(arrays['n'])}.\n\n"
            "Targets: current anonymous regional counts of alive T actors not exposed by the specified radar proxy. "
            "Only training matches fit the spatial partition. Inputs do not contain hidden positions or enemy IDs.\n\n"
            "Full sampled round prefixes include buy time. The 2 Hz grid is refined only with public deaths and friendly smoke lifecycle changes; hidden enemy smoke timing never creates observations. This is not every game tick or full audiovisual information.\n\n"
            + summary["skill_note"] + "\n\n" + summary["smoke_note"] + "\n\n" + summary["audio_note"] + "\n\n"
            f"Input histories with at least one sampled exposure: {summary['history_with_radar_sighting_fraction']:.2%}. "
            f"Metadata clock corrections: {summary['clock_corrections']}.\n\n"
            "Splits and per-demo audits: split_matches.json, demo_audit.csv, summary.json. "
            "No expert tactical labels or human-visibility ground truth are provided. "
            "This subset is not the full upstream dataset and not exclusively top-tier play.\n")
    (out / "DATA_CARD.md").write_text(card, encoding="utf-8")
    return summary


def synthetic_dataset(root: Path, cfg: dict) -> dict:
    """Explicit software-only diagnostic. Never substituted for real demos."""
    rng = np.random.default_rng(1729)
    k, h, m = cfg["regions"], cfg["history_steps"], 1200
    f = 24*k+23
    x = rng.normal(0, .05, (m, h, f)).astype(np.float32)
    x[:, :, 2*k+5] = 1
    n = rng.integers(1, 6, m)
    y = np.zeros((m, k), dtype=np.int64)
    for i in range(m):
        mode = rng.integers(0, 2)
        p = np.ones(k) * .01
        p[int(mode * (k - 1))] += 1
        y[i] = rng.multinomial(n[i], p / p.sum())
        x[i, :, 0] += mode * .3
    group = np.asarray([str(i // 50) for i in range(m)])
    split = np.where(np.arange(m) < 800, "train", np.where(np.arange(m) < 1000, "val", "test"))
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", frames=x.reshape(-1, f), start=np.arange(m)*h, end=(np.arange(m)+1)*h, round_starts=np.arange(m)*h, y=y, n=n, group=group, demo=group,
                        round=np.zeros(m, int), tick=np.arange(m), split=split,
                        elapsed=np.ones(m) * 30, contact_age=np.ones(m) * 5, had_contact=np.ones(m, bool),
                        ct_team=np.repeat("synthetic_CT", m), t_team=np.repeat("synthetic_T", m), smoke_active=np.zeros(m, int), ct_elo=np.ones(m)*1500, t_elo=np.ones(m)*1500, skill_band=np.full(m, cfg["default_skill_band"]))
    summary = {"schema": SCHEMA, "kind": "SYNTHETIC_SOFTWARE_TEST_ONLY", "samples": m,
               "data_config": data_config(cfg)}
    save_json(out / "summary.json", summary)
    save_json(out / "regions.json", {"synthetic": True, "centers_scaled": rng.normal(size=(k, 3)).tolist()})
    return summary
