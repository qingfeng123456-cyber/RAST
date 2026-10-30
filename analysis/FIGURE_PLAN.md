# Paper experiment figures

Every figure has PDF/SVG/PNG output, a standalone `gen_fig_*.py`, shared plot support, and raw sources recorded in `figure_manifest.json`. The retained methods are KNN, vanilla Transformer, and the final region-aligned model (`improved_transformer`).

| File | Content | Sources |
|---|---|---|
| fig_team_fingerprints.pdf | Training-fitted deployment prototypes, training trajectory support and 21 team deployment fingerprints | deployment_prototypes.csv, team_pattern_profiles.csv, training_position_density.npz |
| fig_context_shifts.pdf | Equipment-conditioned deployment usage, paired temporal occupancy and within-match equipment-conditioned regional differences | equipment_pattern_profiles.csv, paired_time_profiles.csv, equipment_region_match_differences.csv |
| fig_match_comparison.pdf | Held-out match-wise joint ES/NLL gains and all-population metric improvements | match_comparison.csv, population_conditioned_performance.csv |
| fig_readout.pdf | Final model's actual regional readout attention and all-seed frozen readout bypass NLL | paper_case_data.npz, frozen_mechanism_probes.csv |
| fig_joint_cases.pdf | Final model's saved free-running occupancy mean/SD and observed counts over training-only trajectory density | paper_case_data.npz, region_centroids.csv, training_position_density.npz |
| fig_space_time.pdf | Final model's complete-history temporal attention and spatial attention in first/last encoder layers | paper_case_data.npz, space_time_attention_bins.npz |
| fig_joint_law.pdf | Two-region joint integer count laws for four unchanged predefined cases; optional main/supplement figure | paper_case_data.npz, joint_count_law_source.csv |

The full six-metric results are saved as `analysis/paper_results.csv`.

The 15 s archetypes fit KMeans only on the 535 training rounds with known CT team and all five CT alive. The selected K=2 is determined by training silhouette among K=2..8. Every eligible round contributes one fixed-time observation. Team profiles average rounds within each match, then give each supporting match equal weight. Tables include all teams satisfying at least two matches and ten rounds. These descriptive profiles cover 51 named-team matches, including training matches; model predictive scores use only the 14 held-out matches.

The optional deployment-conditioned predictive table uses only cutoffs at or after 15 s, after the deployment classification is observable. It is saved in `deployment_conditioned_performance.csv`.
