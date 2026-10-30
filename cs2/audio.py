"""Idealized audible footsteps, explicitly requested by the user.

Every logged enemy step is treated as heard and coarsely localized to a region.
This is stronger than real audio; no acoustics/wall attenuation is simulated.
"""
from pathlib import Path
import logging
import numpy as np


def parse_footsteps(root, record):
    from demoparser2 import DemoParser
    raw = Path(record["local_path"])
    cache = root / "data/sounds" / (str(record["match_id"]) + "_" + raw.stem + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if int(z["raw_bytes"]) == raw.stat().st_size:
                return {k: z[k] for k in z.files}
    sound = DemoParser(str(raw)).parse_event("player_sound", player=["team_num", "X", "Y", "Z"])
    required = {"step", "tick", "user_team_num", "user_X", "user_Y", "user_Z"}
    available = hasattr(sound, "columns") and required.issubset(sound.columns) and len(sound) > 0
    if available:
        sound = sound[sound.step.fillna(False).astype(bool) & sound.user_team_num.isin([2, 3])].dropna(subset=["user_X", "user_Y", "user_Z"]).sort_values("tick")
        data = {"tick": sound.tick.to_numpy(np.int64), "xyz": sound[["user_X", "user_Y", "user_Z"]].to_numpy(np.float32), "side": sound.user_team_num.to_numpy(np.int8)}
    else:
        logging.getLogger("cs2").warning("No logged footstep stream: %s; audio availability=0, no synthesized footsteps", raw.name)
        data = {"tick": np.empty(0, np.int64), "xyz": np.empty((0, 3), np.float32), "side": np.empty(0, np.int8)}
    data.update(raw_bytes=np.asarray(raw.stat().st_size), available=np.asarray(int(available)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    logging.getLogger("cs2").info("Extracted %d logged footsteps: %s", len(data["tick"]), raw.name)
    return data


def footstep_features(ticks, sides, regions, previous_tick, now_tick, last_heard, rate, k):
    """No future event is accessed; carry anonymous regional acoustic memory."""
    relevant = (ticks > previous_tick) & (ticks <= now_tick)
    result = np.zeros((k, 4), np.float32)
    for side, column in ((3, 0), (2, 1)):
        select = relevant & (sides == side)
        result[:, column] = np.log1p(np.bincount(regions[select], minlength=k)) / np.log(21)
    for tick, region in zip(ticks[relevant & (sides == 2)], regions[relevant & (sides == 2)]):
        last_heard[region] = tick
    known = np.isfinite(last_heard)
    result[:, 2] = np.where(known, np.minimum((now_tick - np.nan_to_num(last_heard)) / rate / 120, 3), 0)
    result[:, 3] = known
    return result
