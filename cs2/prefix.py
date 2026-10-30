"""Shared flat round storage + complete variable-length prefix collation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class PrefixDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        start, end = int(self.data["start"][i]), int(self.data["end"][i])
        return self.data["frames"][start:end], self.data["y"][i], self.data["n"][i]


def collate_prefixes(rows, pad_multiple=1):
    longest = max(len(row[0]) for row in rows)
    longest = ((longest+pad_multiple-1)//pad_multiple)*pad_multiple
    values = np.zeros((len(rows), longest, rows[0][0].shape[-1]), np.float32)
    for i, (prefix, _, _) in enumerate(rows):
        values[i, :len(prefix)] = prefix
    return (torch.from_numpy(values), torch.from_numpy(np.stack([r[1] for r in rows])).long(),
            torch.as_tensor([r[2] for r in rows], dtype=torch.long))


def epoch_order(lengths, batch_size, pool_batches=1):
    """Visit each prefix once, with random-pool length grouping to reduce pads.

    Pools start from a random permutation, and complete optimizer batches are
    shuffled again. The sole partial batch stays last so flattening never mixes
    batch boundaries. Only sample order changes; no history is cropped.
    NumPy's seeded RNG makes the schedule reproducible within each epoch.
    """
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or batch_size < 1 or pool_batches < 1:
        raise ValueError("Expected one-dimensional lengths and positive batch/pool sizes")
    order = np.random.permutation(len(lengths))
    if pool_batches == 1 or not len(order):
        return order
    complete, tail = [], []
    pool_size = batch_size * pool_batches
    for start in range(0, len(order), pool_size):
        pool = order[start:start+pool_size]
        pool = pool[np.argsort(lengths[pool], kind="stable")]
        for offset in range(0, len(pool), batch_size):
            chunk = pool[offset:offset+batch_size]
            (complete if len(chunk) == batch_size else tail).append(chunk)
    shuffled = [complete[int(i)] for i in np.random.permutation(len(complete))]
    return np.concatenate(shuffled + tail)


def micro_batches(logical_indices, lengths, base_micro, reference_steps, cap, pad_multiple=32, dynamic=True):
    """Reorder only inside a fixed optimizer batch; preserve its exact examples.

    The capacity is measured in padded frame-tokens. Histories are neither
    cropped nor resampled. Similar lengths reduce padding and allocator churn.
    """
    logical_indices = np.asarray(logical_indices)
    ordered = logical_indices[np.argsort(lengths[logical_indices], kind="stable")]
    budget = base_micro * (((reference_steps+pad_multiple-1)//pad_multiple)*pad_multiple)
    position = 0
    while position < len(ordered):
        candidates = ordered[position:position+(cap if dynamic else base_micro)]
        if dynamic:
            sizes = np.arange(1, len(candidates)+1)
            padded = ((lengths[candidates]+pad_multiple-1)//pad_multiple)*pad_multiple
            valid = sizes*padded <= budget
            size = int(sizes[valid][-1]) if valid.any() else 1
        else:
            size = len(candidates)
        yield ordered[position:position+size]
        position += size


def retrieval_descriptor(x):
    """All-prefix summary for the optional fixed-dimensional retrieval baseline."""
    valid = x.abs().sum(-1) > 0
    lengths = valid.sum(-1).clamp_min(1)
    last = x[torch.arange(len(x), device=x.device), lengths-1]
    average = x.sum(1) / lengths[:, None]
    return torch.cat([last, average], -1)
