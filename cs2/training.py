from __future__ import annotations
import logging
import json
import sys
from pathlib import Path
import random
import time
from contextlib import nullcontext
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .io import save_json
from .models import MODEL_CLASSES, log_coefficient, multinomial_counts
from .prefix import PrefixDataset, collate_prefixes, micro_batches, epoch_order

LOG = logging.getLogger("cs2")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(device):
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state,device):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None: torch.cuda.set_rng_state(state[3],device)


class Prior(nn.Module):
    exact_nll = True
    def __init__(self, y):
        super().__init__()
        counts = torch.as_tensor(y.sum(0), dtype=torch.float32) + 1
        self.register_buffer("p", counts / counts.sum())
    def sample(self, x, n, samples):
        return multinomial_counts(self.p.expand(len(x), -1), n, samples)
    def nll(self, x, y, n):
        return -log_coefficient(y) - (y * self.p.log()).sum(-1)


def save_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def amp_context(model, cfg, device):
    enabled = (device.type == "cuda" and cfg.get("amp", True) and
               getattr(model, "supports_amp", False) and torch.cuda.is_bf16_supported())
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def choose_micro_batch(model, dataset, cfg, device):
    """Probe longest TRAIN prefix; reserve optimizer states and a VRAM margin."""
    cap = min(cfg["batch_size"], cfg.get("max_micro_batch", 256))
    if device.type != "cuda" or not cfg.get("auto_batch", True):
        return min(cap, cfg.get("micro_batch_size", 32)), []
    lengths = dataset.data["end"][dataset.indices] - dataset.data["start"][dataset.indices]
    example = dataset[int(lengths.argmax())]
    free, total = torch.cuda.mem_get_info(device)
    budget = free * cfg.get("gpu_memory_fraction", .8)
    # Enforce the budget BEFORE probing, avoiding a transient oversized
    # allocation that can trigger WDDM/system-memory paging on Windows.
    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(min(budget/total, .9), cuda_index)
    reserve = sum(p.numel()*p.element_size() for p in model.parameters()) * 2
    probes, chosen, size = [], 0, 1
    model.train()
    while size <= cap:
        xb = yb = nb = loss = None
        try:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            xb, yb, nb = [v.to(device) for v in collate_prefixes([example]*size, cfg.get("pad_multiple", 32))]
            with amp_context(model, cfg, device):
                loss = model.loss(xb, yb, nb)
            loss.backward()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            # Different B/T shapes can dispatch different attention kernels.
            # Reserve headroom beyond Adam states, even inside the hard cap.
            safety = float(cfg.get("auto_batch_safety_fraction", .85))
            fits = peak + reserve < budget*safety
            probes.append({"micro_batch": size, "longest_train_prefix": len(example[0]), "padded_steps": xb.shape[1], "peak_bytes": peak, "within_budget": fits, "safety_fraction":safety})
            if not fits:
                break
            chosen = size
            if size == cap:
                break
            size = min(size*2, cap)
        except torch.cuda.OutOfMemoryError:
            probes.append({"micro_batch": size, "oom": True})
            break
        finally:
            del xb, yb, nb, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if not chosen:
        raise RuntimeError("Even micro_batch=1 exceeds the VRAM budget. Reduce transformer_dim/layers; histories will never be truncated.")
    LOG.info("Auto batch: micro=%d, effective=%d; complete prefixes retained", chosen, cfg["batch_size"])
    return chosen, probes


def fit(name, data, cfg, device, output: Path, seed):
    seed_everything(seed)
    output.mkdir(parents=True, exist_ok=True)
    train, val = np.flatnonzero(data["split"] == "train"), np.flatnonzero(data["split"] == "val")
    if name == "prior":
        model = Prior(data["y"][train]).to(device)
        save_checkpoint(output / "prior.pt", {"state": model.state_dict()})
        return model, {"parameters": 0, "epochs": 0, "fit_seconds": 0.0}
    model = MODEL_CLASSES[name](data["frames"].shape[-1], data["y"].shape[-1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    dataset, validation_set = PrefixDataset(data, train), PrefixDataset(data, val)
    lengths = data["end"][train]-data["start"][train]
    pad_multiple = cfg.get("pad_multiple", 32)
    collate = partial(collate_prefixes, pad_multiple=pad_multiple)
    micro, probes = choose_micro_batch(model, dataset, cfg, device)
    save_json(output / "batch_probe.json", probes)
    reference_steps = int(np.ceil(lengths.max()/pad_multiple)*pad_multiple)
    model.prefix_capacity = micro * reference_steps
    seed_everything(seed)
    start_epoch, best, stale, history, seconds = 0, float("inf"), 0, [], 0.0
    latest = output / "last.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best, stale = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"]
        history, seconds = checkpoint["history"], checkpoint["seconds"]
        LOG.info("Resuming %s seed %s at epoch %s", name, seed, start_epoch)
    peak_bytes = 0
    objective = "exact_count_nll" if model.exact_nll else "denoising_ce"
    training_objective = "exact_count_nll_plus_conditional_rps" if getattr(model,"rps_weight",0) > 0 else objective
    selection_metric = cfg.get("selection_metric","objective")
    if selection_metric not in {"objective","energy_score"}: raise ValueError("Unknown selection_metric")
    for epoch in range(start_epoch, cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        seed_everything(seed + epoch*1009)
        order = epoch_order(lengths, cfg["batch_size"],
                            cfg.get("length_bucket_pool_batches", 1) if cfg.get("length_bucketing", True) else 1)
        begin, total = time.perf_counter(), 0.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        bar = tqdm(total=len(dataset), desc=f"{name} seed={seed} | epoch {epoch+1}/{cfg['epochs']}", unit="sample",
                   dynamic_ncols=True, ascii=True, file=sys.stdout, mininterval=.5)
        for offset in range(0, len(order), cfg["batch_size"]):
            logical = order[offset:offset+cfg["batch_size"]]
            rng = capture_rng(device)
            prior_total = total
            while True:
                optimizer.zero_grad(set_to_none=True)
                batches = micro_batches(logical, lengths, micro, int(lengths.max()), min(cfg.get("max_micro_batch",256),cfg["batch_size"]),
                                        pad_multiple, dynamic=device.type == "cuda") if cfg.get("length_bucketing",True) else [logical[p:p+micro] for p in range(0,len(logical),micro)]
                processed, physical_sizes = 0, []
                xb = yb = nb = loss = None
                try:
                    for indices in batches:
                        xb, yb, nb = [v.to(device) for v in collate([dataset[int(i)] for i in indices])]
                        with amp_context(model, cfg, device):
                            loss = model.loss(xb, yb, nb)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite training objective in {name}")
                        (loss * (len(indices)/len(logical))).backward()
                        total += float(loss.detach())*len(indices)
                        processed += len(indices)
                        physical_sizes.append(len(indices))
                        bar.update(len(indices))
                        bar.set_postfix(loss=f"{total/max(1,offset+processed):.4f}", micro=len(indices),
                                        gpu_peak_GB=f"{torch.cuda.max_memory_allocated(device)/1e9:.2f}" if device.type == "cuda" else "CPU", refresh=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda" or micro <= 1: raise
                    del xb, yb, nb, loss
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    bar.update(-processed)
                    total = prior_total
                    micro = max(1,micro//2)
                    model.prefix_capacity = micro*reference_steps
                    restore_rng(rng,device)
                    retry = {"epoch":epoch+1,"logical_batch_offset":offset,"new_base_micro":micro,
                             "policy":"discard partial gradients and replay whole optimizer batch; complete prefixes retained"}
                    with (output / "memory_retries.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(retry)+"\n")
                    LOG.warning("GPU memory retry: %s",retry)
            nn.utils.clip_grad_norm_(model.parameters(), 5.,error_if_nonfinite=True)
            optimizer.step()
            record = {"epoch": epoch+1, "samples_seen": min(offset+len(logical), len(dataset)), "samples_total": len(dataset),
                      "running_objective": total/(offset+len(logical)), "learning_rate": optimizer.param_groups[0]["lr"],
                      "physical_batch_min": min(physical_sizes), "physical_batch_max": max(physical_sizes),
                      "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                      "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
                      "elapsed_seconds": time.perf_counter()-begin}
            with (output / "batch_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            save_json(output.parents[1] / "log/current_progress.json", dict(record, model=name, seed=seed, stage="training"))
        bar.close()
        model.eval()
        save_json(output.parents[1] / "log/current_progress.json",dict(stage="validation",model=name,seed=seed,
            epoch=epoch+1,selection_metric=selection_metric,validation_samples=len(validation_set)))
        seed_everything(seed+999983)
        validation_details = {}
        if selection_metric == "energy_score":
            from .evaluation import evaluate
            validation_scores,_ = evaluate(model,data,val,device,cfg.get("validation_samples",32),cfg.get("eval_batch_size",8),cfg,desc=f"{name} seed={seed} validation | epoch {epoch+1}")
            # Select by the same independent-group aggregation as final reporting.
            val_groups = data["group"][val]
            validation_details = {"validation_"+key:float(np.mean([values[val_groups==g].mean() for g in np.unique(val_groups)]))
                                  for key,values in validation_scores.items()}
            validation = validation_details["validation_energy_score"]
        else:
          with torch.inference_mode():
            val_total = 0.
            val_longest = int((data["end"][val]-data["start"][val]).max())
            val_padded = int(np.ceil(val_longest/pad_multiple)*pad_multiple)
            val_batch = min(micro,cfg.get("eval_batch_size",8),max(1,model.prefix_capacity//val_padded))
            loader = DataLoader(validation_set, batch_size=val_batch, collate_fn=collate)
            for xb, yb, nb in tqdm(loader, desc=f"{name} seed={seed} validation | epoch {epoch+1}/{cfg['epochs']}", dynamic_ncols=True, ascii=True, file=sys.stdout):
                xb, yb, nb = xb.to(device), yb.to(device), nb.to(device)
                with amp_context(model, cfg, device):
                    loss = model.loss(xb, yb, nb)
                val_total += float(loss)*len(xb)
          validation = val_total/len(validation_set)
        if not np.isfinite(validation):
            raise FloatingPointError("Non-finite validation objective")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        seconds += time.perf_counter()-begin
        history.append({"epoch": epoch+1, "train_objective": total/len(dataset), "validation_objective": validation,
                        "selection_metric":selection_metric,**validation_details})
        if validation < best-1e-5:
            best, stale = validation, 0
            save_checkpoint(output / "best.pt", {"model": model.state_dict(), "epoch": epoch, "validation": best,
                            "objective":training_objective,"selection_metric":selection_metric})
        else:
            stale += 1
        save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                                "best": best, "stale": stale, "history": history, "seconds": seconds})
        save_json(output / "history.json", history)
        LOG.info("%s seed=%s epoch=%d train=%.4f val[%s]=%.4f", name, seed, epoch+1, total/len(dataset), selection_metric, validation)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    info = {"parameters": sum(p.numel() for p in model.parameters()), "epochs": len(history), "best_epoch": checkpoint["epoch"]+1,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "fit_seconds": seconds, "selection_objective": selection_metric if selection_metric != "objective" else training_objective,
            "best_validation_score":float(best),"validation_draws":cfg.get("validation_samples",32) if selection_metric == "energy_score" else None,
            "stop_reason":"patience" if stale >= cfg["patience"] else "maximum_epochs",
            "training_objective":training_objective,"conditional_rps_weight":getattr(model,"rps_weight",0),
            "micro_batch_size": micro, "effective_batch_size": cfg["batch_size"], "peak_allocated_bytes": peak_bytes,
            "batching": "length_sorted_dynamic" if cfg.get("length_bucketing",True) else "fixed_micro_batches", "pad_multiple": pad_multiple,
            "length_bucket_pool_batches": cfg.get("length_bucket_pool_batches",1),
            "prefix_frame_capacity": model.prefix_capacity,
            "amp_bfloat16": device.type == "cuda" and getattr(model, "supports_amp", False) and cfg.get("amp", True) and torch.cuda.is_bf16_supported()}
    save_json(output / "fit.json", info)
    return model, info
