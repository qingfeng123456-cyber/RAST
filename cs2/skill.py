"""Pre-match covariates. Event-level skill bands are not Valve ranks."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .io import save_json


def normalize_team(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_team(clan, teams):
    value = normalize_team(clan)
    exact = [name for name in teams if normalize_team(name) == value]
    if len(exact) == 1:
        return exact[0]
    candidates = [name for name in teams if len(normalize_team(name)) >= 2 and
                  (normalize_team(name) in value or value in normalize_team(name)) and value]
    return candidates[0] if len(candidates) == 1 else None


def skill_band_protocol(band_scale_max=3):
    """Keep legacy cache encoding explicit; new cohorts reserve band 4 for pros."""
    if band_scale_max not in (3, 4):
        raise ValueError("skill_band_scale_max must be 3 (legacy) or 4")
    return {
        "scale_max": int(band_scale_max),
        "ordinary_highest_band": 3,
        "professional_band": int(band_scale_max),
        "value_encoding": "band / scale_max, with a separate known-value mask",
        "semantics": ("0..3 ordinary player strata; 4 professional event cohort"
                      if band_scale_max == 4 else "legacy 0..3 event-level strata; 3 highest"),
        "rank_source": "user-defined event-level band; not a Valve rank or official ranking",
    }


def skill_band_note(band_counts, band_scale_max=3):
    protocol = skill_band_protocol(band_scale_max)
    observed = sorted(int(band) for band in band_counts)
    if observed == [4] and band_scale_max == 4:
        assignment = ("All current professional event demos assigned band=4 by user convention, "
                      "one level above the highest ordinary-player band=3. ")
    elif observed == [3] and band_scale_max == 3:
        assignment = "All current event demos assigned highest band=3 by user convention. "
    else:
        assignment = (f"Observed event-level skill bands: {observed}; "
                      f"normalization divisor={protocol['scale_max']}. ")
    coverage = ("This constant-band subset cannot validate cross-rank generalization. "
                if len(observed) <= 1 else
                "Event-level strata are covariates; cross-rank generalization needs a dedicated evaluation. ")
    return assignment + coverage + "Chronological Elo is a separate proxy, never an official rank."


def prematch_ratings(matches, queries, lag_days=1):
    """A result is usable only after its UTC calendar day plus a full lag day.

    Date is generally a scheduled start, not a reliable match completion time.
    The embargo avoids using overlapping or same-day match outcomes.
    """
    matches = matches.drop_duplicates("match_id").copy()
    matches["date"] = pd.to_datetime(matches.match_date, utc=True)
    matches = matches.dropna(subset=["date", "score1", "score2"])
    matches["available"] = matches.date.dt.floor("D") + pd.Timedelta(days=1 + lag_days)
    matches = matches.sort_values(["available", "match_id"])
    pending = list(matches.itertuples(index=False))
    ordered = sorted(queries, key=lambda row: (pd.Timestamp(row["match_date"]), str(row["match_id"])))
    ratings, counts, result, cursor = {}, {}, {}, 0
    for query in ordered:
        when = pd.to_datetime(query["match_date"], utc=True)
        while cursor < len(pending) and pending[cursor].available < when:
            row = pending[cursor]
            cursor += 1
            if float(row.score1) == float(row.score2):
                continue
            a, b = normalize_team(row.team1), normalize_team(row.team2)
            ra, rb = ratings.get(a, 1500.), ratings.get(b, 1500.)
            change = 24 * (float(row.score1 > row.score2) - 1 / (1 + 10 ** ((rb - ra) / 400)))
            ratings[a], ratings[b] = ra + change, rb - change
            counts[a], counts[b] = counts.get(a, 0) + 1, counts.get(b, 0) + 1
        result[str(query["match_id"])] = {
            str(query[side]): {"elo": ratings.get(normalize_team(query[side]), 1500.),
                               "past_matches": counts.get(normalize_team(query[side]), 0)}
            for side in ("team1", "team2")}
    return result


def build_skill_context(root: Path, records, cfg):
    columns = ["match_id", "match_date", "team1", "team2", "score1", "score2"]
    files = list((root / "data/metadata").rglob("*.parquet"))
    matches = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True) if files else pd.DataFrame(columns=columns)
    ratings = prematch_ratings(matches, records, cfg.get("skill_lag_days", 1)) if len(matches) else {}
    protocol = skill_band_protocol(cfg.get("skill_band_scale_max", 3))
    payload = {"source": "chronological Elo from archived series outcomes; not official ranking",
               "lag_days": cfg.get("skill_lag_days", 1), "initial_elo": 1500, "k_factor": 24,
               "professional_band": ("4, above ordinary highest band 3; user-defined professional event cohort, not Valve rank"
                                     if protocol["scale_max"] == 4 else
                                     "3 (highest), user-defined event-level band; not Valve rank"),
               "skill_band_protocol": protocol,
               "matches": ratings}
    save_json(root / "data/processed/skill_context.json", payload)
    return ratings


def skill_features(record, ct_clan, t_clan, context, default_band=3, band_scale_max=3):
    skill_band_protocol(band_scale_max)
    teams = [record.get("team1", ""), record.get("team2", "")]
    values, labels, elos = [], [], []
    for clan in (ct_clan, t_clan):
        resolved = resolve_team(clan, teams)
        info = context.get(str(record["match_id"]), {}).get(resolved, {})
        count = info.get("past_matches", 0)
        elo = info.get("elo", 1500.)
        # local manifest may specify team_skill_bands and team_ranks (normalized [0,1]).
        band = record.get("team_skill_bands", {}).get(resolved, default_band)
        if band is not None and band not in range(band_scale_max + 1):
            raise ValueError(f"Skill band must be 0..{band_scale_max} or null")
        rank = record.get("team_ranks", {}).get(resolved)
        if rank is not None and not 0 <= rank <= 1:
            raise ValueError("External pre-match rank must be normalized to [0,1]")
        values.extend([0 if band is None else band / band_scale_max, float(band is not None),
                       0 if rank is None else rank, float(rank is not None),
                       (elo - 1500) / 400, float(count > 0), min(count, 30) / 30])
        labels.append(resolved or str(clan) or "unknown")
        elos.append(elo)
    return np.asarray(values, np.float32), labels, elos
