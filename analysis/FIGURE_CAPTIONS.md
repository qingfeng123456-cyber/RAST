## fig_context_shifts

Context-conditioned deployment and temporal redistribution. Equipment is CT total equipment value at freeze end, with lower/middle/upper cutoffs 21500/28700, fitted on training rounds. (a) Match-balanced archetype usage at 15 s with five CT alive; (b) normalized survivor occupancy at 15/45/75 s on the same 667 rounds from 51 matches; (c) within-match upper-minus-lower equipment occupancy differences at 15 s, restricted to matches with at least two rounds in each tier; whiskers are match-bootstrap 95% intervals. All displayed profiles use equal match weighting.

## fig_match_comparison

Held-out predictive gains of the region-aligned model over the vanilla Transformer. (a) Each point is one of 14 held-out matches, after equal averaging across the three seeds; positive coordinates favor the final model. The diamond and marginal intervals summarize equally weighted matches (10,000 match resamples, seed 2026), conditional on these trained runs. (b) Gains at each known hidden-alive population, using the same within-match then across-match and seed aggregation. All five populations and all displayed metrics are retained. The evaluation contains 27,034 cutoff histories from 293 rounds.

## fig_readout

Regional readout mechanisms of the final model. (a,b) Head-mean attention minus uniform for predefined cases [64, 232]; gray rows have zero remaining population, and the final region is determined by conservation. (c) Exact NLL after frozen readout bypass minus the unmodified checkpoint, averaged within each of the 14 matches, for all three seeds on the same 128 prespecified cutoffs. Positive values show dependence of the learned predictor on its readout. Attention panels use seed 2026, validation-selected epoch 5 after 13 completed epochs. Readout panels are teacher-forced conditional likelihood traces; the right panel is a frozen-checkpoint intervention.

## fig_joint_cases

Final-model occupancy and uncertainty for the unchanged predefined cases [64, 232, 1]. Columns identify cases; rows show the mean and standard deviation of 256 free-running count-vector samples, then the single observed target count. All count vectors conserve the known hidden-alive population. Coordinates are training-fitted region centroids over a light backdrop of empirical training trajectory density. Common count/SD limits are set by the displayed populations. Cases use representative seed 2026, validation-selected epoch 5.

## fig_space_time

Spatial and complete-history temporal attention of the final model for predefined case 64, seed 2026. Top: last-query temporal attention, averaged over all heads and summed into fixed 5-second bins, for the first and last encoder layers. Negative time is the buy phase; the displayed history ends at the observed cutoff. Bottom: head-mean spatial attention at that last real time. Raw tokens, timestamps and all four layer/head arrays are saved.

## fig_joint_law

Two-region joint count laws of the final model in predefined cases [64, 232, 1, 8], estimated from 256 free-running samples. In each case the two regions with the largest predictive count variance are shown. Gray cells have total count greater than the known hidden-alive population; orange crosses mark the single observed target. The shared probability scale is 0–1. Seed2026, validation-selected epoch 5.

## fig_deployment_archetypes

Training-fitted regional deployment prototypes and spatial centroids; no team-style inference.