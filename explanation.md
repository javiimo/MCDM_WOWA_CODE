# From the paper to the code

This document maps the notation, results and experiments of the paper to the code. Paths
are relative to the repository root, and `output/` is created by the scripts.

## 1. Notation

| Paper | Meaning | Code |
|---|---|---|
| $C=\{c_1,\dots,c_7\}$ | criteria | `CRITERIA_DIRECTIONS` (`wowa_stability_analysis.py`) |
| $\mathbf X$, $x_{ij}$ | normalized decision matrix | `WOWAStabilityAnalyzer.normalized_criteria_matrix` |
| $\mathbf p$, $p_j$ | criteria weighting vector | `weights` (dict), `exact_weights()` (fractions adding up to one) |
| $\mathbf p(B)$ | weight of a subset $B\subseteq C$ | `subset_weights` |
| $\Omega_{\mathbf p}=\{t_0<\dots<t_K\}$ | complete weight grid (Definition 4) | `build_complete_weight_grid()`, `grid`, `grid_index` |
| grid of Definition 3 | cumulative weights of the alternatives | `minimal_grid_size()` |
| $t_k-t_{k-1}$, $\mathbf W^{\mathrm{lin}}$ | cell widths, weighted mean $Q(t)=t$ | `cell_widths` |
| $\mathbf z_i$ | unified vector of $a_i$ (Theorem 1) | row `i` of `z` |
| $\mathbf W\in\Delta^{K-1}$ | unified weights $W_k=Q(t_k)-Q(t_{k-1})$ | LP variables `w` |
| $\boldsymbol o$ | orness coefficients (Proposition 3) | `orness_coefficients` |
| $\mathbf G$ | Shapley matrix, $\boldsymbol\phi=\mathbf G\mathbf W$ (Proposition 2) | `shapley_matrix` |
| $[\underline o,\overline o]$ | orness interval of a profile | `PROFILES`, arguments `orness_min`, `orness_max` |
| $[\underline\phi_j,\overline\phi_j]$ | importance requirements | `shapley_bounds_around_weights(p, tolerance)` |
| $\mathcal W$ | admissible set, Equation (4) | `_build_ball_model()` |
| $\kappa(\mathbf n)$ | margin coefficient of a norm | `margin_coefficient(n, norm, widths)` |
| (LP$_i$), $\rho_i$, $\mathbf W_i^*$ | stability Linear Program, radius and center | `solve_stability_radius_lp()`: `Rho`, `Center` |
| $\rho_{\mathcal P}$ | largest ball in $\mathcal W$, without dominance | `reference_radius()`: `Rho_P` |
| $\theta_c$, $\mathbf d_c$ | fragility shares and tie direction (Proposition 6) | `Share_*`, `Rival`, `steepest_direction()`, `Tie_*` |
| $\delta^*$ | smallest $\max_j\lvert\phi_j-p_j\rvert$ in a profile | `min_shapley_deviation()` |
| $Q_{\mathbf W^*}$ | representative quantifier | `plot_quantifiers()` (`plots_and_tables/plot_stability_results.py`) |

## 2. The method (`wowa_stability_analysis.py`)

`WOWAStabilityAnalyzer(raw_decision_matrix_path, weights)` reads a raw decision matrix and
normalizes it. `build_unified_representation()` then builds, for the weighting vector $\mathbf p$:

1. **The complete grid** $\Omega_{\mathbf p}$, from the $2^M$ subset weights in exact rational
   arithmetic. Weights are read with ten decimals and rescaled to add up to one, so equal
   subset weights always give the same grid point and floating-point round-off cannot create
   spurious cells.
2. **The coefficients** $\boldsymbol o$ and $\mathbf G$, by adding the contribution of every
   subset $B$ to the cells $t_k\le\mathbf p(B)$ (orness) and $\mathbf p(B)<t_k\le\mathbf p(B)+p_j$ (Shapley).
3. **The vectors** $\mathbf z_i$: if $B_{ij}$ holds the $j$ best criteria of $\mathbf x_i$, the
   cells between $\mathbf p(B_{i,j-1})$ and $\mathbf p(B_{ij})$ take the $j$-th largest value of $\mathbf x_i$.
   The cells are found by exact lookup of $\mathbf p(B_{ij})$ in the grid.

`check_wowa_identities()` verifies Theorem 1 against direct WOWA evaluations with
$Q(t)=t^{1/2}$ and $Q(t)=t^2$, Proposition 2 with a random capacity, and the identities
$\mathbf G\mathbf W^{\mathrm{lin}}=\mathbf p$, $\boldsymbol o^T\mathbf W^{\mathrm{lin}}=1/2$ and $o_1=1$.

`solve_stability_radius_lp()` solves (LP$_i$) for every alternative of the choice set:
maximize $\rho$ subject to $\mathbf 1^T\mathbf W=1$ and one margin constraint
$\mathbf n_c^T\mathbf W-\kappa(\mathbf n_c)\rho\ge b_c$ per face $W_k\ge0$, orness bound, importance bound
and competitor. The constraints without margin are implied by $\rho\ge0$. The norms are:

| `norm` | $\lVert\mathbf d\rVert$ | $\kappa(\mathbf n)$ |
|---|---|---|
| `"L2"` | $\lVert\mathbf d\rVert_2$ | $\lVert\Pi\mathbf n\rVert_2$ |
| `"L1"` | $\lVert\mathbf d\rVert_1$ | $(\max_k n_k-\min_k n_k)/2$ |
| `"capacity"` | $\max_k\lvert d_1+\dots+d_k\rvert$ | $\sum_k\lvert n_{k+1}-n_k\rvert$ |
| `"slope"` | $\max_k\lvert d_k\rvert/(t_k-t_{k-1})$ | $\sum_k(t_k-t_{k-1})\lvert n_k-m\rvert$, $m$ a weighted median of $\mathbf n$ |

With `details=True`, each row also reports the center, its orness, Shapley importance and
entropy, the fragility shares $\theta_c=\lambda_c\kappa_c$ of each type of constraint, the
competitor with the largest share and the changes of orness and importance in the direction
of the tie with it. `MAIN_NORM` sets the norm of every experiment.

## 3. Pipeline

| Step | Script | Produces | Paper |
|---|---|---|---|
| 1 | `data/Alternatives/Solvers/execute.py`, `extract_results.py` | candidate schedules | experimental setup, Appendix D |
| 2 | `spatial_reconstruction.py` | MDS map of X12 (seed 42), tracked in `data/spatial_reconstruction/` | Appendix B |
| 3 | `decision_matrix_builder.py` | raw matrix of X12 | decision model, Appendix C |
| 3' | `decision_matrix_builder.py --extension` | raw matrices of X03, X12 and X13 (all $\alpha$) | extensions |
| 4 | `wowa_stability_analysis.py` | radii, centers, fragility, Shapley tolerances, $\delta^*$, choice-set size, grid size | results |
| 5 | `wowa_extension_analysis.py` | selection on the extended candidate sets, LP size | extensions, computational cost |
| 6 | `plots_and_tables/generate_appendix_tables.py` | decision matrix and solver runs with competition scores | Appendices C and D |
| 7 | `mds_stability_analysis.py` | 20 MDS seeds: Procrustes alignment, containment, memberships, selection | map uncertainty, Appendix B |
| 8 | `supplementary_analyses/mds_checks.py` | reproducibility, stress, orientations, rank correlations, ablation of $c_5,c_6$ | map uncertainty, Appendix B |
| 9 | `supplementary_analyses/weighted_dominance.py` | alternatives dominated for every quantifier | selection and its margins |
| 10 | `supplementary_analyses/patterns.py` | correlations of the radius with entropy, score, time limit and distinctiveness | patterns of stability |
| 11 | `supplementary_analyses/norm_and_weight_sensitivity.py` | selection under each norm, refinement, perturbations of $\mathbf p$, 0.05-lattice neighbours | sensitivity to $\mathbf p$ and to the norm |
| 12 | `supplementary_analyses/fragility_dual_check.py` | range of $\theta_c$ over the optimal duals | direction of fragility |
| 13 | `plots_and_tables/*.py` | LaTeX tables and figures | all tables and figures |

## 4. Decision matrix (`maintenance_scheduling_model.py`)

All criteria are computed by `MaintenanceSchedule.compute_raw_criteria(instance, mu_ski, mu_park)`
from the daily risk $R_\tau$ (`compute_risk_profile`) and the active interventions
$\mathcal A_\tau$ (`compute_active_interventions`):

| Criterion | Paper | Code |
|---|---|---|
| $c_1$ Reliability Stress | $\sum_\tau\max(0,R_\tau-q_{90})$ | `compute_raw_criteria` |
| $c_2$ Max Critical Overlap | $\max_\tau\lvert\mathcal A_\tau\cap\mathcal I_{crit}\rvert$ | `compute_max_critical_overlap_criterion` |
| $c_3$ Risk Volatility | $\sum_\tau\lvert R_\tau-R_{\tau-1}\rvert$ | `compute_risk_volatility_criterion` |
| $c_4$ Resource Slack | $\frac1T\sum_\tau\min_r\frac{U^{\max}_{r,\tau}-L_{r,\tau}}{U^{\max}_{r,\tau}-U^{\min}_{r,\tau}}$ | `_compute_bottleneck_slack_profile` |
| $c_5$ Ski Season Risk | $\sum_{\tau\in\mathcal T_{winter}}\sum_{v\in\mathcal A_\tau}\bar r_{v,\tau}\,\mu_{ski}(v)$ | `compute_raw_criteria` |
| $c_6$ Summer Eco-Stress | $\sum_{\tau\in\mathcal T_{summer}}\max(0,E_\tau-\tilde E)$ | `compute_summer_eco_stress_criterion` |
| $c_7$ Social Weekend Risk | $\sum_v\mathbb I(S_v\in\mathcal T_{weekend}\wedge\bar r_v>P_{80})$ | `compute_raw_criteria` |

Timesteps are mapped to calendar days by `timestep_to_day_interval`, with the seasonal blocks
of each instance read from its `Seasons` field (`configure_season_blocks`). Instances are
loaded by `load_maintenance_instance_from_json`, and schedules by `MaintenanceSchedule(path)`.
`decision_matrix_builder.build_raw_decision_matrix` keeps one row per distinct vector of criteria.

Normalization (`load_and_normalize_decision_matrix`): benefit criteria ($c_4$) are divided by
their largest value, and cost criteria are mapped to $1-(x-\min x)/\max x$. A criterion whose
largest value is zero receives the value 1 for every alternative.

## 5. Spatial reconstruction (`spatial_reconstruction.py`, `spatial_memberships.py`)

| Paper | Code |
|---|---|
| sign concordance $\sigma_{vw}$ of the risk increments | `compute_sign_concordance`, `compute_sign_concordance_matrix` |
| $d_{vw}=(1-\hat\sigma_{vw})/2$, only for positive concordance | `transform_correlation_to_dissimilarity`, `compute_intervention_dissimilarity_matrix` |
| non-metric SMACOF in $\mathbb R^2$ | `compute_weighted_mds_embedding` (rpy2), `compute_mds_embeddings_rscript` (Rscript, several seeds) |
| placement on the map of France | `scale_embedding_to_france` |
| fuzzy memberships $\mu_{ski}$, $\mu_{park}$ | `compute_ski_area_proximity_membership` (30–80 km), `compute_park_proximity_membership` (1–10 km) |
| Procrustes alignment and containment in France | `align`, `containment` (`mds_stability_analysis.py`) |
