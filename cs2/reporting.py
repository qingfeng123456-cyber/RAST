from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import bootstrap_summary, seed_summaries


def paper_tables(output: Path, aggregate: pd.DataFrame, cfg: dict, summary: dict):
    """Write ready-to-include tables without hiding seeds or ranking methods."""
    metrics = [("energy_score", "ES"), ("exact_count_nll", "NLL"), ("count_brier", "Count Brier"),
               ("pair_presence_brier", "Pair Brier"), ("count_mae", "Count MAE")]
    metrics = [(key, label) for key, label in metrics if key in set(aggregate.metric)]
    model_order = list(cfg.get("models", []))
    model_order.extend(name for name in aggregate.model.unique() if name not in model_order)
    rows, markdown = [], []
    for name in model_order:
        fields = [name]
        record = {"model": name}
        for metric, _ in metrics:
            subset = aggregate[(aggregate.model == name) & (aggregate.metric == metric)]
            if subset.empty:
                fields.append("--")
                continue
            value = subset.iloc[0]
            mean, std = float(value.seed_mean), float(value.seed_std)
            record.update({metric+"_mean": mean, metric+"_sample_std": std,
                           metric+"_match_ci_low": float(value.ci_low), metric+"_match_ci_high": float(value.ci_high),
                           metric+"_seed_count": int(value.seed_count)})
            fields.append(f"{mean:.4f} +/- {std:.4f}" if np.isfinite(std) else f"{mean:.4f}")
        rows.append(record)
        markdown.append("| "+" | ".join(fields)+" |")
    pd.DataFrame(rows).to_csv(output / "paper_results_table.csv", index=False)
    caption = ("Held-out match-macro scores, mean and sample standard deviation across "
               f"{len(cfg['seeds'])} configured seed runs. All scores are lower-is-better. "
               "Seed variability and match-bootstrap confidence intervals measure different sources of uncertainty; "
               "the latter are retained in the CSV files. A dash means no evaluable exact likelihood, not zero.")
    if cfg.get("verification_samples_per_split"):
        caption = "SYNTHETIC OR LIMITED SOFTWARE VERIFICATION ONLY; NOT PAPER PERFORMANCE EVIDENCE. "+caption
    table = ["# Saved evaluation results", "", caption, "", "| Model | "+" | ".join(label for _,label in metrics)+" |", "|"+"---|"*(len(metrics)+1), *markdown]
    (output / "RESULTS_TABLE.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    metadata = dict(data_kind=summary["kind"], software_verification=bool(cfg.get("verification_samples_per_split")),
                    configured_seeds=cfg["seeds"], configured_models=cfg.get("models", []),
                    aggregation="Mean per cutoff within each seed/match, equal-weight mean over matches, then mean and ddof=1 SD over configured seeds.",
                    selection="Each trained checkpoint is selected on validation energy score; test scores do not select a checkpoint.",
                    metrics={"energy_score": "Euclidean count-vector energy score: mean distance to truth minus half the mean distance of paired independent predictive draws.",
                             "exact_count_nll": "Negative log probability of the entire observed regional count vector, nats per cutoff; KNN has no evaluated exact likelihood.",
                             "count_brier": "Regional categorical count Brier score, summed over counts 0..5 then averaged over regions.",
                             "pair_presence_brier": "Brier score for simultaneous presence in each unordered pair of distinct regions, averaged over pairs.",
                             "count_mae": "Mean absolute error of the sampled predictive mean count, averaged over regions."},
                    limitations=summary.get("limitations", []))
    (output / "paper_results_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def report(output: Path, scores: pd.DataFrame, cfg: dict, summary: dict):
    import logging
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    aggregate, paired = bootstrap_summary(scores, cfg["bootstrap_repeats"])
    per_seed, seed_aggregate = seed_summaries(scores, cfg["seeds"])
    per_seed.to_csv(output / "per_seed_metrics.csv", index=False)
    seed_aggregate.to_csv(output / "seed_summary.csv", index=False)
    aggregate = aggregate.merge(seed_aggregate, on=["model", "metric"], validate="one_to_one")
    aggregate.to_csv(output / "comparison.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    paper_tables(output, aggregate, cfg, summary)
    for stratifier, bins in (("contact_age", [-2, -.5, 1, 5, 12, 30, 120, 300]), ("n", [0, 1, 2, 3, 4, 5]),
                             ("smoke_active", [-1, 0, 1, 3, 10]), ("ct_elo", [0, 1450, 1500, 1550, 3000]),
                             ("ct_alive", [0,1,2,3,4,5]), ("audio_available", [-.5,.5,1.5])):
        if stratifier not in scores: continue
        temp = scores.copy()
        if stratifier == "ct_elo" and "ct_elo_known" in temp:
            temp = temp[temp.ct_elo_known.astype(bool)]
        if not len(temp): continue
        temp["stratum"] = pd.cut(temp[stratifier], bins=bins)
        metrics = [m for m in ("energy_score", "count_brier", "pair_presence_brier") if m in scores]
        # Equal weight per test match inside each reported stratum.
        grouped = temp.groupby(["model", "group", "stratum"], observed=True)[metrics].mean().reset_index()
        grouped.groupby(["model", "stratum"], observed=True)[metrics].mean().to_csv(output / f"stratified_{stratifier}.csv")
    for team in ("ct_team", "t_team"):
        per_match = scores.groupby(["model", "group", team], as_index=False)[["energy_score", "count_brier"]].mean()
        per_match.groupby(["model", team]).agg(energy_score=("energy_score", "mean"), count_brier=("count_brier", "mean"),
                                               matches=("group", "nunique")).to_csv(output / f"by_{team}.csv")
    main = aggregate[aggregate.metric == "energy_score"].sort_values("match_macro_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(main.match_macro_mean.to_numpy(), np.arange(len(main)),
                xerr=[(main.match_macro_mean - main.ci_low).to_numpy(), (main.ci_high - main.match_macro_mean).to_numpy()], fmt="o", capsize=4)
    ax.set_yticks(np.arange(len(main)), main.model)
    ax.set_xlabel("Occupancy energy score (lower is better); match-cluster 95% CI")
    ax.set_title("Synthetic software check" if "SYNTHETIC" in summary["kind"] else "Held-out matches")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for model in scores.model.unique():
        paths = list(output.glob(f"*/{model}/calibration.csv"))
        if not paths:
            continue
        tables = pd.concat([pd.read_csv(p) for p in paths])
        for field in ("predicted", "observed"):
            tables[field] *= tables["count"]
        table = tables.groupby("bin")[["predicted", "observed", "count"]].sum()
        table[["predicted", "observed"]] = table[["predicted", "observed"]].div(table["count"], axis=0)
        ax.plot(table.predicted, table.observed, marker=".", label=model)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="Predicted regional presence probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=180)
    plt.close(fig)
    lines = ["CS2 anonymous occupancy experiment", "", "DATA KIND: " + summary["kind"],
             "All primary scores are lower-is-better. Exact count NLL is absent for methods without an evaluated exact likelihood, including KNN.",
             "CI resamples complete test match groups; seeds are averaged within each match.",
             "per_seed_metrics.csv gives equal-match scores for every configured seed.",
             "seed_summary.csv reports mean and sample SD across seed runs, separately from match-bootstrap CI.",
             "Seed-run SD includes predictive sampling variation; prior/KNN have no training randomness.",
             "These intervals do not establish generalization to all tournaments, patches or teams.", ""]
    for row in paired.itertuples(index=False):
        lines.append(f"{row.comparison}, {row.metric}: delta={row.delta:.5f}, 95% CI [{row.ci_low:.5f}, {row.ci_high:.5f}]")
    lines.extend(["", "No tactical quality, causal effect, player intention or win-rate improvement was evaluated.",
                  "This is a research starting point; no model is assumed to be the winner."])
    (output / "read_results.txt").write_text("\n".join(lines), encoding="utf-8")
