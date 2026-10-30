"""Check loss connectivity and parameter drift without an optimizer step."""
from __future__ import annotations
import numpy as np
import torch
from .io import save_json
from .models import MODEL_CLASSES
from .prefix import PrefixDataset, collate_prefixes


def audit(model, name, data, cfg, device, folder, seed):
    train = np.flatnonzero(data["split"] == "train")
    chosen = [int(train[0]), int(train[len(train)//2])]
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.inference_mode(False), torch.enable_grad():
        dataset = PrefixDataset(data, chosen)
        x, y, n = [v.to(device) for v in collate_prefixes([dataset[0], dataset[1]])]
        loss = model.loss(x, y, n)
        loss.backward()
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            initial = MODEL_CLASSES[name](data["frames"].shape[-1], data["y"].shape[-1], cfg)
        original = dict(initial.named_parameters())
        records = []
        for key, value in model.named_parameters():
            gradient = value.grad
            current, old = value.detach().float().cpu(), original[key].detach().float().cpu()
            records.append({"parameter": key, "elements": value.numel(), "has_gradient": gradient is not None,
                            "gradient_l2": None if gradient is None else float(gradient.detach().float().norm()),
                            "change_from_seeded_initial_rms": float((current-old).square().mean().sqrt())})
        model.zero_grad(set_to_none=True)
    save_json(folder / "gradient_audit.json", {"training_seed": seed, "probe_split": "train", "sample_indices": chosen,
        "objective": float(loss.detach()), "optimizer_steps_in_this_audit": 0,
        "interpretation": "Nonzero gradients verify computational loss connectivity on this fixed training probe. Weight drift can also include weight decay; neither quantity alone proves predictive benefit. Output-level interventions and controlled comparisons are separately saved.",
        "parameters": records})
