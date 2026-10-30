# Regenerating RAST experimental figures

The repository contains saved numerical evidence and independent plotting scripts. Redrawing the figures does not require neural training, a GPU, or raw DEMO downloads. The model architecture overview is in `docs/architecture.png`; it is a conceptual illustration rather than experimental evidence.

## Original mechanism panels

From the repository root:

```bash
python plot_paper.py --run results/paper_model
```

This writes the original saved-diagnostic mechanism panels under `results/paper_model/figures/`. The entry point uses the final model's actual saved attention, evidence gates, residuals, count-vector draws, and frozen-checkpoint intervention records. It does not load trained weights for inference.

## Composite experimental figures

Install the plotting dependencies in the repository requirements, then run:

```bash
python figures/plot_support.py
python figures/gen_fig_evidence_gate.py
python figures/gen_compact_mechanisms.py
```

The first command redraws seven composite analyses. The second redraws the evidence-gate composite from the original diagnostic arrays. The third creates compact versions of the regional-readout and joint-count-law figures while keeping the same numerical evidence, case selection, and native font sizes. Each entry point saves PDF, SVG, and PNG files in `figures/`.

Individual figures can also be regenerated with the corresponding `figures/gen_fig_*.py` script:

| Entry point | Analysis |
|---|---|
| `gen_fig_deployment_archetypes.py` | Training-fitted regional deployment prototypes and spatial centroids |
| `gen_fig_context_shifts.py` | Equipment-conditioned deployment and paired temporal redistribution |
| `gen_fig_match_comparison.py` | Held-out match and hidden-population comparisons |
| `gen_fig_readout.py` | Regional attention and frozen readout-bypass likelihood changes |
| `gen_fig_joint_law.py` | Two-region joint integer-count distributions |
| `gen_fig_joint_cases.py` | Joint occupancy means, uncertainty, and observed counts |
| `gen_fig_space_time.py` | Complete-history temporal and spatial attention |
| `gen_fig_evidence_gate.py` | Context-conditioned temporal gate modulation |

The composite figures use `analysis/*.csv`, `analysis/*.json`, and `analysis/*.npz`; `analysis/figure_manifest.json` maps figures to sources. The compact layouts have their own `analysis/compact_figure_manifest.json`. The evidence-gate sources are also saved as `figures/evidence_gate_source.csv`.

## Recompute aggregate and descriptive analysis

The numeric arrays and summaries are already included. To reconstruct their aggregations from the frozen processed dataset and archived predictions:

```bash
python analysis/build_paper_analysis.py --processed data/processed --log results/paper_model
python analysis/validate_paper_analysis.py
```

This recomputes a small descriptive clustering of training-round snapshots and saved-score summaries. It does not train or run the neural model. The seven composite plotting scripts can then be run again. The analysis builder writes CSV/JSON/NPZ outputs and does not create a manuscript or LaTeX paper.

The empirical training-position density is included for immediate redraw. Rebuilding that background from parsed coordinates is optional and requires an external frame-cache directory supplied with `--frames-cache`. When that cache is absent, the existing saved background and its provenance are preserved. The density backdrop represents sampled training trajectories rather than a complete navigation mesh.

## Interpretation of diagnostic evidence

The case selection is fixed in `analysis/diagnostic_selection.json`. Regional-readout attention traces use the original teacher-forced likelihood computation; occupancy draws use free-running decoding. Frozen checkpoint bypasses measure reliance of an already trained model on a component and are identified as such in the source captions. Team and equipment profiles summarize observed deployment under the recorded conditions.

The actual manuscript, its LaTeX source, bibliography, and supplementary document are distributed separately from this code/data repository.
