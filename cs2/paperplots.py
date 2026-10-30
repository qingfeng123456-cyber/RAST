"""Plot-only scientific diagnostics. Imports no torch and runs no model."""


from __future__ import annotations


import hashlib


import json


from pathlib import Path


import warnings


import numpy as np


import pandas as pd


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


from matplotlib.colors import TwoSlopeNorm, BoundaryNorm


KINDS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_file(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def linear_cka(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(0), y-y.mean(0)
    return np.square(x.T@y).sum() / max(1e-12, np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum()))


class Bundle:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.cfg = read_json(self.run / "resolved_config.json")
        self.selection = read_json(self.run / "diagnostic_selection.json")
        self.rows = pd.DataFrame(self.selection["rows"])
        self.seed = self.cfg["seeds"][0]
        self.root = self.run / f"seed_{self.seed}"
        self.primary_name = self.cfg.get("figure_primary","tactical_transformer")
        self.primary = self.root / self.primary_name
        self.trace = array_file(self.primary / "diagnostics/traces.npz")
        self.cases = array_file(self.primary / "diagnostics/cases.npz")
        self.regions = read_json(self.run / "plot_context/regions.json")
        self.summary = read_json(self.run / "plot_context/data_summary.json")
        self.k = len(self.regions["centers_scaled"])
        self.source_files = set()

    def control(self,kind):
        defaults = {"distance":"transformer_no_distance","gate":"transformer_no_gate","skill":"transformer_no_skill",
                    "readout":"improved_no_readout","score":"improved_no_score","allocation":"improved_transformer"}
        return self.cfg.get("figure_controls",{}).get(kind,defaults[kind])

    def reliability_reference(self):
        for name in ("tactical_multinomial",self.control("score"),"tactical_transformer","independent"):
            if name != self.primary_name and (self.root / name / "test_predictions.npz").exists():return name
        raise ValueError("No saved reference predictions for the reliability comparison")

    def cooccurrence_reference(self):
        for name in ("tactical_multinomial", "vanilla_transformer", "tactical_transformer",
                     "independent", self.control("readout"), self.control("score"), "knn"):
            if name != self.primary_name and (self.root / name / "test_predictions.npz").exists():
                return name
        raise ValueError("No saved reference predictions for the co-presence comparison")

    def predictions(self, name):
        path = self.root / name / "test_predictions.npz"
        self.source_files.add(path)
        return array_file(path)

    def traces(self, name):
        path = self.root / name / "diagnostics/traces.npz"
        self.source_files.add(path)
        return array_file(path)

    def base_caption(self, aggregate_seeds=False):
        if aggregate_seeds:
            return (f"Performance summaries use configured training seeds {self.cfg['seeds']}, "
                    "averaged equally within each held-out match. Match-bootstrap intervals are conditional "
                    "on these runs; separate seed summaries report run-to-run standard deviation. " +
                    ("SOFTWARE VERIFICATION SUBSET, not a full experiment. " if self.cfg.get("verification_samples_per_split") else ""))
        fit_path = self.primary / "fit.json"
        fit = read_json(fit_path) if fit_path.exists() else {}
        phase = (f"Validation-selected epoch {fit['best_epoch']} checkpoint after {fit['epochs']} completed training epochs; "
                 if "best_epoch" in fit and "epochs" in fit else "")
        return (phase + f"prespecified representative training seed {self.seed}; {len(self.rows)} diagnostic cutoffs from {self.rows.group.nunique()} held-out matches. "
                "Diagnostic cutoffs were prespecified by live time and match-balanced, without selecting successful predictions. "
                "Overlapping prefixes are not independent replicates. " +
                ("SOFTWARE VERIFICATION SUBSET, not a full experiment. " if self.cfg.get("verification_samples_per_split") else ""))


def panel(ax, letter):
    ax.text(-.12, 1.03, f"({letter})", transform=ax.transAxes, fontsize=10, va="bottom")


def matrix(ax, data, label, vmin=None, vmax=None, diverging=False):
    arr = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdBu_r" if diverging else "viridis").copy()
    cmap.set_bad(".85")
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=.046, pad=.035, label=label)
    return im


def match_macro(values, groups):
    return np.stack([values[np.asarray(groups) == group].mean(0) for group in np.unique(groups)]).mean(0)


def A1(b):
    case = b.selection["case_rows"][0]
    weights, no_bias, bias = [b.cases[f"case{case}_{key}"].mean(1) for key in ("spatial_attention", "spatial_no_bias", "distance_bias")]
    layers = sorted(set([0, len(weights)-1]))
    fig, axes = plt.subplots(len(layers), 3, figsize=(7.15, 2.4*len(layers)), squeeze=False, layout="constrained")
    delta = weights-no_bias
    bound = max(abs(delta).max(), 1e-5)
    for row, layer in enumerate(layers):
        for col, (data, label, lo, hi, div) in enumerate([
            (weights[layer], "Attention weight", 0, float(weights.max()), False),
            (bias[layer], "Fixed-distance logit bias", float(bias.min()), 0, False),
            (delta[layer], "Weight change (bias − no bias)", -bound, bound, True)]):
            ax = axes[row, col]
            matrix(ax, data, label, lo, hi, div)
            ax.set(xlabel="Key region", ylabel=f"Query region · layer {layer+1}")
            panel(ax, chr(97+row*3+col))
    return fig, (b.base_caption()+f"Case {case}, match {b.rows.iloc[case].group}. Head means of spatial attention at the final real timestep. "
                "The middle column shows a learned per-head negative bias strength multiplying fixed normalized centroid distances. "
                "The right column removes that local bias with Q/K held fixed; it is not the full-model bypass intervention. All heads are retained in the source tensors. Attention is a mechanism trace, not causal attribution.")


def temporal_bin_edges(time, width=5.):
    """Cover observed timestamps once; never append an unobserved future bin."""
    time = np.asarray(time, float)
    if not len(time) or not np.isfinite(time).all() or width <= 0:
        raise ValueError("Finite nonempty timestamps and positive bin width required")
    lo = np.floor(time.min()/width)*width
    hi = np.ceil(time.max()/width)*width
    if hi <= lo: hi = lo+width
    return np.arange(lo, hi+width*.01, width)


def A2(b):
    case = b.selection["case_rows"][0]
    attention = b.cases[f"case{case}_temporal_attention"].mean(1)
    time = b.cases[f"case{case}_time"]
    # Sum probability mass into prespecified 5 s bins: unequal token intervals
    # must not be presented as a probability density.
    edges = temporal_bin_edges(time)
    bins = np.clip(np.searchsorted(edges, time, side="right")-1, 0, len(edges)-2)
    mass = np.zeros((len(attention), b.k, len(edges)-1))
    for i in range(len(time)):
        mass[:, :, bins[i]] += attention[:, :, i]
    ncol = min(3, int(np.ceil(np.sqrt(len(mass))))); nrow = int(np.ceil(len(mass)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.15, 2.1*nrow), squeeze=False, layout="constrained")
    for layer, ax in enumerate(axes.flat):
        if layer >= len(mass):
            ax.set_visible(False); continue
        im = ax.pcolormesh(edges, np.arange(b.k+1)-.5, mass[layer], cmap="viridis", vmin=0, vmax=max(mass.max(), .01), shading="flat")
        fig.colorbar(im, ax=ax, label="Attention mass per 5-second bin")
        ax.axvline(0, color="white", linestyle="--", linewidth=.7)
        ax.set(xlabel="Seconds relative to freeze end", ylabel=f"Region · layer {layer+1}")
        ax.set_xlim(float(time.min()), float(time.max()) if time.max() > time.min() else float(time.max()+.01))
        panel(ax, chr(97+layer))
    (b.run / "figures/A2_bins.json").write_text(json.dumps({"case":int(case), "edges_seconds":edges.tolist(), "observed_range_seconds":[float(time.min()),float(time.max())], "mass_sum_by_layer_region":mass.sum(-1).tolist()},indent=2),encoding="utf-8")
    return fig, b.base_caption()+f"Case {case}: last-query temporal attention over the entire sampled prefix, summed into fixed 5-second bins and averaged over heads. The displayed range ends at the observed cutoff; boundary bins can contain less than 5 seconds of observations. Negative time is buy phase. Original token weights and timestamps are saved. This is not an attention rollout or evidence-importance score."


def projections(b):
    path = b.run / "projection_cache.npz"
    meta = b.run / "projection_parameters.json"
    sources = {}
    for name in ("vanilla_transformer", b.primary_name, b.control("skill")):
        p = b.root / name / "diagnostics/traces.npz"
        if p.exists():
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if path.exists() and meta.exists() and read_json(meta).get("source_sha256") == sources:
        return array_file(path)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import inspect
    result, settings = {}, []
    for name in sources:
        x = b.traces(name)["embeddings"]
        if len(x) < 8:
            continue
        reduced = PCA(n_components=min(50, x.shape[1], len(x)-1), random_state=1717).fit_transform(x)
        for seed in (17, 42):
            for requested in (15, 30):
                perplexity = min(float(requested), max(2., (len(x)-1)/3))
                kwargs = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
                kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
                key = f"{name}__s{seed}__p{requested}"
                result[key] = TSNE(**kwargs).fit_transform(reduced)
                settings.append(dict(key=key, pca_dimensions=reduced.shape[1], **kwargs))
    np.savez_compressed(path, **result)
    meta.write_text(json.dumps({"source_sha256": sources, "settings": settings,
                    "warning": "Each model is projected independently. Coordinates/global distances are not aligned between models. All four seed/perplexity combinations retained."}, indent=2), encoding="utf-8")
    return result


def A3(b):
    other = b.traces("vanilla_transformer")["representations"]
    ours = b.trace["representations"]
    similarities = np.array([[linear_cka(other[:, i], ours[:, j]) for j in range(ours.shape[1])] for i in range(other.shape[1])])
    coords = projections(b)
    pred = b.predictions(b.primary_name)
    fraction = pred["truth"][b.rows.test_position.to_numpy(), 0] / b.rows.n.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), layout="constrained")
    matrix(axes[0], similarities, "Linear CKA", 0, 1)
    axes[0].set(xlabel="Structured encoder state", ylabel="Vanilla encoder state")
    for ax, name in zip(axes[1:], ("vanilla_transformer", b.primary_name)):
        xy = coords[f"{name}__s17__p30"]
        im = ax.scatter(xy[:, 0], xy[:, 1], c=fraction, s=9, cmap="cividis", vmin=0, vmax=1, linewidths=0, alpha=.85)
        ax.set(xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, label="Target mass fraction in R0")
    for i, ax in enumerate(axes): panel(ax, chr(97+i))
    used_perplexity = min(30., max(2., (len(b.rows)-1)/3))
    return fig, b.base_caption()+f"Linear CKA uses corresponding per-cutoff region-mean layer states, with state 0 before the first block. t-SNE shows independently projected vanilla and structured final embeddings, projection seed 17/perplexity {used_perplexity:g} (other settings are saved as stability coordinates). Target labels color points only and do not fit the projection. Projection appearance alone cannot establish improved tactical representations."


def B1(b):
    e = b.trace["evidence"]
    if "gate_channels" not in b.trace: raise ValueError("Full channel gates were not saved in this older diagnostic run")
    gate = np.sqrt(np.square(b.trace["gate_channels"]-1).mean(-1))
    population = b.rows.groupby(["ct_alive", "t_alive"]).size().idxmax()
    retain = ((b.rows.ct_alive == population[0]) & (b.rows.t_alive == population[1])).to_numpy()
    condition = [([e[:, :, 0] == 0, e[:, :, 0] > 0], ["No active", "Active"]),
                 ([e[:, :, 3] < .5, (e[:, :, 3] > .5) & (e[:, :, 2]*120 <= 5), (e[:, :, 3] > .5) & (e[:, :, 2]*120 > 5)], ["Never", "≤5 s", ">5 s"]),
                 ([(e[:, :, 8] < .5) & (e[:, :, 9] > .5), (e[:, :, 8] > .5) & (e[:, :, 6] == 0) & (e[:, :, 9] > .5), (e[:, :, 6] > 0) & (e[:, :, 9] > .5)], ["Never", "Memory", "New steps"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), layout="constrained")
    rows = []
    for col, (masks, labels) in enumerate(condition):
        values = np.full((gate.shape[1], len(masks)), np.nan)
        support_labels = []
        for j, mask in enumerate(masks):
            mask = mask & retain[:, None]
            means = []
            for group in b.rows.group.unique():
                valid = mask & (b.rows.group.to_numpy() == group)[:, None]
                if valid.any():
                    means.append(np.stack([gate[:, layer][valid].mean() for layer in range(gate.shape[1])]))
            if len(means) >= 2 and mask.sum() >= 5:
                values[:, j] = np.mean(means, 0)
            support_labels.append(f"{labels[j]}\n{int(mask.sum())} / {len(means)}")
            for layer in range(gate.shape[1]):
                rows.append(dict(context=col, category=labels[j], layer=layer+1, gate_modulation_rms=values[layer, j], region_cutoffs=int(mask.sum()), matches=len(means)))
        bound = max(float(gate.max()), .002)
        matrix(axes[col], values, "Channel RMS(gate − 1)", 0, bound)
        axes[col].set(xticks=np.arange(len(labels)), xticklabels=support_labels, xlabel=["Friendly smoke state", "Radar contact age", "Audible enemy footsteps"][col], ylabel="Layer")
        axes[col].set_yticks(np.arange(gate.shape[1]), np.arange(1, gate.shape[1]+1))
        panel(axes[col], chr(97+col))
    pd.DataFrame(rows).to_csv(b.run / "figures/B1_source.csv", index=False)
    return fig, b.base_caption()+f"Final-timestep temporal gate modulation, measured as channel RMS deviation from unity, for the most frequent public population pair CT={population[0]}, T={population[1]}. Gate=0.5+sigmoid lies in (0.5,1.5), scales only the temporal update, and is not a modality probability. RMS does not distinguish amplification from attenuation. Region-cutoff means are averaged equally over supporting matches; cells with <2 matches or <5 region-cutoffs are gray. Category labels give region-cutoffs / supporting matches. Missing audio is excluded from footstep contexts. Exact per-cell support is in B1_source.csv. These are conditional associations, not effects of changing tactics."


def B2(b):
    ratio = b.trace["residual_norms"][:, :, :, 1:] / np.maximum(b.trace["incoming_norms"], 1e-8)
    mean = match_macro(ratio.mean(2), b.rows.group.to_numpy())
    case = b.selection["case_rows"][0]
    example = ratio[case, :, :, 1].T
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), layout="constrained")
    matrix(axes[0], mean, "Update RMS / preceding state RMS", 0, max(float(mean.max()), 1e-3))
    axes[0].set(xticks=[0, 1, 2], xticklabels=["Spatial", "Gated time", "FFN"], xlabel="Residual update", ylabel="Layer")
    axes[0].set_yticks(np.arange(len(mean)), np.arange(1, len(mean)+1))
    matrix(axes[1], example, "Temporal update / preceding state RMS", 0, max(float(example.max()), 1e-3))
    axes[1].set(xlabel="Layer", ylabel="Region")
    axes[1].set_xticks(np.arange(example.shape[1]), np.arange(1, example.shape[1]+1))
    for j, ax in enumerate(axes): panel(ax, chr(97+j))
    return fig, b.base_caption()+f"Final-timestep output-projected residual updates, including tanh(scale) and the temporal gate. Each update is normalized by its own immediately preceding state RMS, epsilon=1e-8. Left: equal-match mean, after within-cutoff region averaging. Right: predefined case {case}. Norm ratios do not add as causal contributions because update vectors can cancel."


def B3(b):
    table = pd.read_csv(b.primary / "diagnostics/interventions.csv", dtype={"group": str})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.5), layout="constrained")
    for j, key in enumerate(("neutral_gate_nll", "no_distance_nll")):
        delta = table.assign(delta=table[key]-table.baseline_nll).groupby("group").delta.mean()
        axes[j].axhline(0, color=".35", linestyle="--", linewidth=.8)
        axes[j].scatter(np.arange(len(delta)), delta, marker="o", s=30, color="#0072B2")
        axes[j].set(xticks=np.arange(len(delta)), xticklabels=[f"M{i+1}" for i in range(len(delta))],
                    xlabel=["Gate fixed to unity", "Distance bias bypassed"][j], ylabel="Paired Δ NLL (nats per cutoff)")
        panel(axes[j], chr(97+j))
    return fig, b.base_caption()+f"Frozen-checkpoint computational interventions on {len(table)} identical cutoffs from {table.group.nunique()} matches. All weights, inputs, full histories and modalities stay fixed. Each point is one match mean; Δ=intervened−ordinary, so positive values mean the original mechanism gives lower NLL. All signs are retained. This is a sensitivity probe, not a retrained ablation or a causal effect of gameplay changes. Match IDs map in lexicographic order to M1... in interventions.csv."


def cooccurrence(prediction, mask):
    samples = prediction["samples"][mask] > 0
    truth = prediction["truth"][mask] > 0
    predicted = np.einsum("bsi,bsj->bij", samples.astype(float), samples.astype(float))/samples.shape[1]
    observed = truth[:, :, None]*truth[:, None, :]
    groups = prediction["group"][mask]
    return match_macro(predicted, groups), match_macro(observed.astype(float), groups)


def C1(b):
    primary = b.predictions(b.primary_name)
    population = int(b.cfg.get("figure_population", 3))
    mask = primary["truth"].sum(-1) == population
    if mask.sum() < 5: raise ValueError("Insufficient fixed-population cutoffs for C1")
    baseline = b.cooccurrence_reference()
    names = [baseline, b.primary_name]
    entries = [cooccurrence(b.predictions(name), mask) for name in names]
    bound = max(max(abs(p-o).max() for p, o in entries), .01)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.5), layout="constrained")
    for row, (p, o) in enumerate(entries):
        for col, (data, label, lo, hi, div) in enumerate([(o, "Empirical co-presence", 0, 1, False), (p, "Predictive co-presence", 0, 1, False), (p-o, "Prediction − empirical", -bound, bound, True)]):
            matrix(axes[row, col], data, label, lo, hi, div)
            axes[row, col].set(xlabel="Region", ylabel=f"Region · {'Reference' if row == 0 else 'Proposed'}")
            panel(axes[row, col], chr(97+row*3+col))
    return fig, (b.base_caption()+f"Full held-out predictions restricted to n={population} hidden alive enemies: {mask.sum()} cutoffs from {np.unique(primary['group'][mask]).size} matches. Each matrix is an equal-match mean of pair-presence probabilities. Diagonal entries are marginal presence. Residual matrices are aggregate agreement diagnostics, not sufficient proof of calibration; pair-presence Brier and reliability data are also saved. Baseline={baseline}; "
                 +("encoder design and training protocol are shared; only the probabilistic head differs. " if baseline == "tactical_multinomial" and b.primary_name == "tactical_transformer" else
                   "encoder design and initial encoder parameters are shared, but head/readout and auxiliary training score can differ; this is an overall method comparison, with isolated factors tested separately. " if baseline in {"tactical_multinomial","tactical_transformer"} else "encoder differs, so effects cannot be attributed to the decoder alone. ")+
                 "A multinomial actor model already has dependent region counts due to fixed population; it is not independent region counts.")


def C2(b):
    p = b.predictions(b.primary_name)
    cases = b.selection["case_rows"][:4]
    ncol = min(2,len(cases)); nrow = int(np.ceil(len(cases)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.15, 2.7*nrow), squeeze=False, layout="constrained")
    descriptions = []
    for col, j in enumerate(cases):
        pos, n = int(b.rows.iloc[j].test_position), int(b.rows.iloc[j].n)
        draws = p["samples"][pos].astype(int)
        a, c = np.argsort(draws.var(0))[-2:]
        probability = np.zeros((n+1, n+1))
        np.add.at(probability, (draws[:, a], draws[:, c]), 1/len(draws))
        impossible = np.add.outer(np.arange(n+1), np.arange(n+1)) > n
        probability[impossible] = np.nan
        ax = axes.flat[col]
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(".85")
        im = ax.imshow(np.ma.masked_invalid(probability), origin="lower", aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.scatter([p["truth"][pos, c]], [p["truth"][pos, a]], marker="x", s=40, color="#D55E00", linewidths=1.5)
        ax.set(xlabel=f"Count in R{c}", ylabel=f"Count in R{a}", xticks=np.arange(n+1), yticks=np.arange(n+1))
        panel(ax, chr(97+col))
        descriptions.append(f"{chr(97+col)}: case {j}, R{a}/R{c}, n={n}")
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat)[:len(cases)], label="Predictive probability", shrink=.8)
    return fig, b.base_caption()+f"Discrete two-region predictive laws estimated from {p['samples'].shape[1]} draws per cutoff. "+"; ".join(descriptions)+". Within each predefined case, the two regions with greatest predictive count variance are selected without using the outcome. Cross marks the single observed realization, not a true conditional distribution. Gray cells are structurally impossible (sum>n); an estimated zero in a possible cell can be Monte Carlo sampling zero. Common [0,1] probability scale."


def C3(b):
    p = b.predictions(b.primary_name)
    cases = b.selection["case_rows"][:3]
    centers = np.asarray(b.regions["centers_scaled"])[:, :2]
    fig, axes = plt.subplots(3, len(cases), figsize=(7.15, 6), squeeze=False, layout="constrained")
    max_population = max(int(b.rows.iloc[j].n) for j in cases)
    scales = [max_population, max_population/2, max_population]
    row_images = {}
    support_path = b.run / "plot_context/map_support.npz"
    support = array_file(support_path) if support_path.exists() else None
    for col, j in enumerate(cases):
        pos = int(b.rows.iloc[j].test_position)
        values = [p["samples"][pos].mean(0), p["samples"][pos].std(0), p["truth"][pos]]
        for row, v in enumerate(values):
            ax = axes[row, col]
            if support is not None:
                gray = np.ma.masked_where(support["density"] <= .08, support["density"])
                ax.imshow(gray, extent=support["extent"], origin="lower", cmap="Greys", alpha=.22, zorder=0)
            im = ax.scatter(centers[:, 0], centers[:, 1], c=v, s=90, cmap="viridis", vmin=0, vmax=scales[row], edgecolors=".25", linewidths=.5)
            row_images[row] = im
            for region, (x, y) in enumerate(centers): ax.annotate(str(region), (x, y), xytext=(3, 5), textcoords="offset points", fontsize=7)
            ax.set(xlabel="Game X (units)", ylabel="Game Y (units)", aspect="equal")
            panel(ax, chr(97+row*len(cases)+col))
    for row in range(3):
        fig.colorbar(row_images[row], ax=list(axes[row]), label=["Mean count (players)", "Count SD (players)", "Observed count (players)"][row], shrink=.85)
    return fig, b.base_caption()+f"Predefined cases {cases}: predictive mean (top), predictive count standard deviation (middle), and observed target counts (bottom) at training-fitted region centroids. Shared row color limits use the displayed cases' theoretical support: count 0–{max_population}, SD 0–{max_population/2:g}. Region numbers are annotated. This is a 2D projection in original game units. If present, the gray backdrop is empirical training-trajectory density only; it is not a navigation mesh, complete walkability map or reconstructed scene. No test trajectories construct that backdrop. Region predictions are not spatially interpolated. The predictive sample count is {p['samples'].shape[1]}."


def D1(b):
    cases = b.selection["case_rows"][:4]
    arrays = [b.cases[f"case{j}_readout_attention"].mean(1) for j in cases]
    fig,axes = plt.subplots(2,2,figsize=(7.15,5.2),layout="constrained")
    uniform = 1/b.k
    bound = max(max(abs(values-uniform).max() for values in arrays),1e-6)
    cmap = plt.get_cmap("RdBu_r").copy();cmap.set_bad(".85")
    for ax,j,values in zip(axes.flat,cases,arrays):
        active = b.cases[f"case{j}_readout_remaining"] > 0
        shown = np.where(active[:,None],values-uniform,np.nan)
        im = ax.imshow(np.ma.masked_invalid(shown),origin="lower",aspect="auto",vmin=-bound,vmax=bound,cmap=cmap)
        ax.set(xlabel="Source region token",ylabel="Allocated region / decoder step")
        panel(ax,chr(97+cases.index(j)))
    for ax in list(axes.flat)[len(cases):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axes.flat)[:len(cases)],label="Attention weight − uniform (1/K)",shrink=.85)
    (b.run / "figures/D1_scale.json").write_text(json.dumps({"uniform_weight":uniform,"symmetric_difference_limit":float(bound),"cases":cases},indent=2),encoding="utf-8")
    return fig,b.base_caption()+f"Population-conditioned regional readout for predefined cases {cases}; head-mean attention minus the uniform 1/K reference, with shared symmetric color limits ±{bound:.4g}. Each query accesses the final real-time region tokens produced from the entire observed prefix. These are teacher-forced likelihood diagnostics: earlier region allocations use the observed target counts, not information given to deployed inference. Gray rows have zero remaining population and a deterministic zero count; the final region is determined by conservation and has no decoder query. Raw per-head weights, conditional probabilities and remaining counts are saved. Attention alone does not prove predictive benefit."


def D2(b):
    table = pd.read_csv(b.primary / "diagnostics/interventions.csv",dtype={"group":str})
    delta = table.assign(delta=table.no_readout_nll-table.baseline_nll).groupby("group").delta.mean()
    cases = b.selection["case_rows"][:4]
    updates = np.stack([np.where(b.cases[f"case{j}_readout_remaining"] > 0,b.cases[f"case{j}_readout_residual_rms"],np.nan) for j in cases])
    fig,axes = plt.subplots(1,2,figsize=(7.15,2.7),layout="constrained")
    axes[0].axhline(0,color=".5",linestyle="--",linewidth=.8)
    axes[0].scatter(np.arange(len(delta)),delta,color="#0072B2",s=28)
    axes[0].set(xticks=np.arange(len(delta)),xticklabels=[f"M{i+1}" for i in range(len(delta))],xlabel="Held-out match",ylabel="Readout bypass − full NLL (nats/cutoff)")
    matrix(axes[1],updates,"Actual readout residual RMS",0,float(np.nanmax(updates)))
    axes[1].set(xlabel="Allocated region / decoder step",yticks=np.arange(len(cases)),yticklabels=[str(j) for j in cases],ylabel="Predefined case")
    for j,ax in enumerate(axes):panel(ax,chr(97+j))
    delta.rename("delta_nll").to_csv(b.run / "figures/D2_source.csv")
    return fig,b.base_caption()+f"Left: frozen checkpoint readout bypass on {len(table)} identical cutoffs from {len(delta)} matches, holding every input, weight and modality fixed. Positive means the active readout gives lower exact NLL; all match signs are shown. Right: actual tanh-scaled update added to the decoder input, measured by channel RMS during the teacher-forced diagnostics in D1. Gray cells have no remaining mass. Update magnitude and frozen sensitivity are mechanism checks, not retrained benefits or game-level causal effects."


def reliability_data(b):
    from itertools import combinations
    entries = []
    for name in (b.reliability_reference(), b.primary_name):
        path = b.root / name / "test_predictions.npz"
        if not path.exists(): continue
        p = b.predictions(name)
        selected = p["truth"].sum(-1) == b.cfg.get("figure_population", 3)
        present, truth = p["samples"][selected] > 0, p["truth"][selected] > 0
        groups = p["group"][selected]
        for a, c in combinations(range(b.k), 2):
            prob = (present[:, :, a] & present[:, :, c]).mean(1)
            obs = truth[:, a] & truth[:, c]
            bins = np.minimum((prob*10).astype(int), 9)
            for group in np.unique(groups):
                for cell in range(10):
                    mask = (groups == group) & (bins == cell)
                    if mask.any(): entries.append(dict(model=name, group=group, region_a=a, region_b=c, bin=cell,
                                      count=int(mask.sum()), predicted=float(prob[mask].mean()), observed=float(obs[mask].mean())))
    pd.DataFrame(entries).to_csv(b.run / "figures/pair_reliability_source.csv", index=False)


def render(run, kinds=None, dpi=300, formats=("pdf", "svg", "png")):
    import logging
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})
    b = Bundle(run)
    destination = b.run / "figures"
    destination.mkdir(exist_ok=True)
    manifest_path = destination / "figure_manifest.json"
    previous = read_json(manifest_path) if kinds and manifest_path.exists() else {}
    # Standalone gen_A1.py etc. must not erase the documentation of other
    # already rendered figures.  Refuse a partial refresh after raw evidence
    # changed: that would leave stale pictures appearing to share new sources.
    previous_sources = []
    for source in previous.get("source_fingerprints", []):
        path = (b.run / source["path"]).resolve()
        if not path.is_relative_to(b.run) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Saved figure sources changed; regenerate all configured figures before a partial redraw")
        previous_sources.append(path)
    captions = read_json(destination / "captions.json") if previous and (destination / "captions.json").exists() else {}
    previous_figures = {row["figure"]: row for row in previous.get("figures", [])}
    includes, manifest = [], []
    for kind in (kinds or b.cfg.get("paper_figures",KINDS)):
        try:
            fig, caption = globals()[kind](b)
            for extension in formats:
                fig.savefig(destination / f"{kind}.{extension}", dpi=dpi, bbox_inches="tight", pad_inches=.04)
            plt.close(fig)
            captions[kind] = caption
            environment,width = ("figure","0.98\\columnwidth") if fig.get_size_inches()[0] <= 4 else ("figure*","0.96\\textwidth")
            manifest.append({"figure": kind, "status": "generated", "caption": caption, "formats": list(formats),
                             "latex_environment": environment, "latex_width": width, "dpi": dpi,
                             "training_seeds": b.cfg["seeds"] if kind in {"A4","B4","D3","D4","P3","S2"} else [b.seed]})
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.warn(f"{kind} skipped: {exc}")
            captions.pop(kind, None)
            manifest.append({"figure": kind, "status": "not_generated", "reason": str(exc)})
    if b.cfg.get("figure_reliability", True): reliability_data(b)
    previous_figures.update({row["figure"]: row for row in manifest})
    order = list(b.cfg.get("paper_figures", KINDS))
    order.extend(kind for kind in previous_figures if kind not in order)
    combined_manifest = [previous_figures[kind] for kind in order if kind in previous_figures]
    for row in combined_manifest:
        if row["status"] != "generated" or row["figure"] not in captions:
            continue
        kind, caption = row["figure"], captions[row["figure"]]
        narrow = kind in {"A4", "B4", "D3", "D4", "P3"}
        environment = row.get("latex_environment", "figure" if narrow else "figure*")
        width = row.get("latex_width", "0.98\\columnwidth" if narrow else "0.96\\textwidth")
        includes.append("## "+kind+"\n\n!["+kind+"]("+kind+".png)\n\n"+caption+"\n")
    (destination / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "FIGURE_INDEX.md").write_text("\n".join(includes), encoding="utf-8")
    source_paths = sorted(set([b.run / "diagnostic_selection.json", b.run / "resolved_config.json",
        b.primary / "fit.json", b.primary / "diagnostics/traces.npz", b.primary / "diagnostics/cases.npz",
        b.primary / "diagnostics/interventions.csv", b.run / "all_test_scores.csv", b.run / "projection_cache.npz",
        b.run / "projection_parameters.json", *b.source_files, *previous_sources, *(b.run / "plot_context").glob("*"),
        *destination.glob("*_source.csv"), destination / "A2_bins.json",destination / "D1_scale.json", *b.root.glob("*/fit.json")]))
    sources = [{"path": p.relative_to(b.run).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths if p.exists()]
    manifest_path.write_text(json.dumps({"figures": combined_manifest, "training_seed": b.seed, "source_fingerprints": sources,
        "representative_training_seed": b.seed, "configured_training_seeds": b.cfg["seeds"],
        "selection": b.selection["selection_rule"], "projection_stability": "2 seeds × 2 perplexities, cached independently for every available model",
        "aggregation": "within-match mean then equal-weight mean across matches; no token-level independent replication",
        "model_fit_records":{p.parent.name:read_json(p) for p in b.root.glob("*/fit.json")},
        "software_check": bool(b.cfg.get("verification_samples_per_split")), "dpi": dpi,
        "last_rendered_figures": [row["figure"] for row in manifest]}, indent=2), encoding="utf-8")
    return manifest

