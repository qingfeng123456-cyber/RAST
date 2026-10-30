"""Read-only checks for the delivered flat, complete-prefix real dataset."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from cs2.data import SCHEMA,PUBLIC_CLOCK_PROTOCOL
processed = root / "data/processed"
summary = json.loads((processed / "summary.json").read_text(encoding="utf-8"))
splits = json.loads((processed / "split_matches.json").read_text(encoding="utf-8"))
regions = json.loads((processed / "regions.json").read_text(encoding="utf-8"))
assert summary["kind"] == "real_demo_radar_proxy" and summary["schema"] == SCHEMA
assert summary["observation_clock"] == PUBLIC_CLOCK_PROTOCOL
assert regions["fit_split"] == "train"
with np.load(processed / "dataset.npz", allow_pickle=False) as z:
    k = summary["data_config"]["regions"]
    x, y, n = z["frames"], z["y"], z["n"]
    start, end, starts = z["start"], z["end"], z["round_starts"]
    assert not {"t_xyz", "ct_xyz", "steamid", "enemy_id"}.intersection(z.files)
    assert x.shape[-1] == 24*k+23 and np.isfinite(x).all()
    assert np.all(y >= 0) and np.array_equal(y.sum(-1), n)
    assert np.all((n >= 1) & (n <= 5))
    assert np.isin(start, starts).all() and np.all(start < end)
    stop = np.concatenate([starts[1:], [len(x)]])
    assert np.all(end <= stop[np.searchsorted(starts, start)])
    np.testing.assert_allclose(x[end-1, 2*k+3]*5, n, atol=1e-5)
    np.testing.assert_allclose(x[:, :k].sum(-1), x[:, 2*k], atol=1e-5)
    np.testing.assert_allclose(x[:, k:2*k].sum(-1), x[:, 2*k+2], atol=1e-5)
    np.testing.assert_allclose(x[end-1, 2*k+4]*120, z["elapsed"], atol=1e-4)
    assert np.all(x[:, 2*k+5] == 1)
    assert np.all(z["contact_age"][~z["had_contact"]] == -1)
    assert all(splits[str(g)] == s for g, s in zip(z["group"], z["split"]))
    groups = [set(z["group"][z["split"] == split]) for split in ["train", "val", "test"]]
    assert all(not (groups[a] & groups[b]) for a,b in [(0,1),(0,2),(1,2)])
    print(json.dumps({"status": "PASS", "samples": len(n), "feature_frames": len(x), "features": x.shape[-1],
                      "groups": len(splits), "splits": summary["splits"], "full_prefix_max": int((end-start).max()),
                      "clock_corrections": summary["clock_corrections"], "audio_demos": summary["audio_demos_available"]}, indent=2))
