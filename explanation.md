# Implementation Traceability: Paper Sections -> Repository Code

This document maps the manuscript content to the actual implementation in this repository, in execution order.

It is written to answer, with high traceability:
- where each formula is implemented,
- which class/method computes each decision-matrix attribute,
- how the WOWA transformation and the stability LP are encoded,
- and what practical differences currently exist between manuscript equations and code.

Naming note:
- the methodology used is WOWA.

---

## 1. End-to-End Execution Order (What Runs First, What Produces What)

### Step 1. Generate candidate schedules with external solvers
- File: `data/Alternatives/Solvers/execute.py`
- Key functions:
  - `create_run_all_sh()`
  - `create_run_all_bat()`
  - `main()`
- What it does:
  - Builds a batch/shell execution harness for the compiled Team1, Team2, Team3 binaries published by the competition teams.
  - Runs each solver for `(time_limit in {180,300,500,700,900}) x (seed in {42,33,73})`.
  - Writes `sol*_t*_X_12_s*.txt` and organizes them by team folders.

### Step 2. Build intervention spatial embedding (MDS)
- File: `spatial_reconstruction.py`
- Key functions:
  - `compute_sign_concordance()`
  - `compute_sign_concordance_matrix()`
  - `transform_correlation_to_dissimilarity()`
  - `compute_intervention_dissimilarity_matrix()`
  - `compute_weighted_mds_embedding()`
  - `compute_and_save_spatial_reconstruction()`
- Output artifacts:
  - `points_*.npy`
  - `points_keys_*.npy`

### Step 3. Build the raw decision matrix from schedules
- File: `decision_matrix_builder.py`
- Key function:
  - `build_raw_decision_matrix(...)`
- Required inputs:
  - schedule files under `data/Alternatives/X12_Original_Execution/`
  - spatial reconstruction files `points_*.npy` and `points_keys_*.npy` generated in Step 2
- Uses:
  - `maintenance_scheduling_model.load_maintenance_instance_from_json(...)`
  - `maintenance_scheduling_model.MaintenanceSchedule(...).compute_raw_criteria(...)`
  - `spatial_memberships.compute_ski_area_proximity_membership(...)`
  - `spatial_memberships.compute_park_proximity_membership(...)`
- Output artifacts:
  - `output/csv/decision_matrix_raw.csv`
  - `output/md/decision_matrix_raw.md`

### Step 4. Normalize matrix + build transformed space + solve WOWA stability LP
- File: `wowa_stability_analysis.py`
- Key class:
  - `WOWAStabilityAnalyzer`
- Key methods:
  - `load_and_normalize_decision_matrix()`
  - `build_unified_importance_grid()`
  - `build_transformed_alternative_matrix()`
  - `solve_stability_radius_lp(alpha_min, alpha_max)`
- Output artifacts:
  - `output/csv/wowa_scenarios_comparison.csv`
  - scenario/profile winner tables in console

### Step 5. Importance/orness sensitivity sweeps (optional analysis)
- Files:
  - `archive/scripts_exploration/choose_importance.py`
  - `archive/scripts_exploration/analyze_importance_search.py`
- Purpose:
  - exhaustive search over importance vectors, profile winners, stability-radii changes.

---

## 2. Paper Section -> Code Mapping

## 2.1 Section 03 (manuscript Section 03): Spatial and Temporal Reconstruction

### Formula block: sign-concordance similarity and distance transform
Paper equations:
- `sigma_ij = sum_t s(Delta r_i,t, Delta r_j,t)`
- `d_ij = (1 - sigma_hat_ij) / 2`

Code mapping:
- `spatial_reconstruction.py::compute_sign_concordance` implements sign-concordance contributions `+1/-1/0`.
- `spatial_reconstruction.py::compute_sign_concordance_matrix` normalizes by diagonal terms.
- `spatial_reconstruction.py::transform_correlation_to_dissimilarity(method="linear")` implements `(1-r)/2`.
- `spatial_reconstruction.py::compute_intervention_dissimilarity_matrix` sets off-diagonal distances to `NaN` when correlation `<= 0`, matching the "keep positive correlation" design.

### Formula block: non-metric MDS (SMACOF)
Paper concept:
- non-metric MDS in `R^2` from dissimilarities.

Code mapping:
- `spatial_reconstruction.py::compute_weighted_mds_embedding`
  - calls R `smacofSym(..., type="ordinal")` through `rpy2`.
  - uses weighted stress with `weightmat`.

### Formula block: affine scaling to France map
Paper concept:
- projected cloud is affinely adjusted to French geography.

Code mapping:
- `spatial_memberships.py::scale_embedding_to_france`
  - rescales to France bounding box + empirical offsets.

### Formula block: fuzzy proximity membership `mu_Z(i)`
Paper piecewise function:
- `1` if near enough,
- linear decay in transition band,
- `0` if far enough.

Code mapping:
- `spatial_memberships.py::compute_feature_proximity_membership`
  - exact piecewise linear membership with `(near_full_dist, near_zero_dist)`.
- `spatial_memberships.py::compute_ski_area_proximity_membership`
  - defaults `near_full_km=30`, `near_zero_km=80`.
- `spatial_memberships.py::compute_park_proximity_membership`
  - defaults `near_full_m=1000`, `near_zero_m=10000`.

Data loading used by these memberships:
- `spatial_memberships.py::load_regional_natural_parks`
- `spatial_memberships.py::load_ski_stations` (GeoJSON or XLSX fallback with `X,Y` in EPSG:2154).

### Temporal mapping needed by seasonal/weekend criteria
Code mapping:
- `maintenance_scheduling_model.py::timestep_to_day_interval`
- `maintenance_scheduling_model.py::model_day_to_date_2025`
- helper predicates:
  - `timestep_is_winter`
  - `timestep_is_summer`
  - `timestep_has_weekend`
  - `timestep_intersects_model_days`

---

## 2.2 Section 03: Decision-Matrix Criteria (`c1` ... `c7`)

All criteria are assembled in:
- `maintenance_scheduling_model.py::MaintenanceSchedule.compute_raw_criteria`

Support computations:
- active interventions by timestep: `MaintenanceSchedule.compute_active_interventions`
- daily risk curve `R_t`: `MaintenanceSchedule.compute_risk_profile`
- resource slack curve: `MaintenanceSchedule._compute_bottleneck_slack_profile`
- critical overlap helper: `MaintenanceSchedule.compute_max_critical_overlap_criterion`
- volatility helper: `MaintenanceSchedule.compute_risk_volatility_criterion`
- eco-stress helper: `MaintenanceSchedule.compute_summer_eco_stress_criterion`

### Criterion mapping table

| Paper criterion | Paper expression | Code implementation | Notes |
|---|---|---|---|
| `c1` Reliability Stress | `sum_t max(0, R_t - tau_90)` | `compute_raw_criteria`: `safety_threshold=np.percentile(non_zero_risks,90)` then `np.sum(np.maximum(0, risk_profile - safety_threshold))` | direct match |
| `c2` Max Critical Overlap | `max_t \lvert A_t \cap I_{crit} \rvert` | `compute_max_critical_overlap_criterion`: `top_10_count=max(1,int(len(all_interventions)*0.10))`; `critical_names=set(...)`; `count=sum(1 for name in active_intvs if name in critical_names)`; `max_overlap=max(overlap_curve)` | Exact implementation in `maintenance_scheduling_model.py:1037-1070` |
| `c3` Risk Volatility | `sum_{t=2}^T \lvert R_t - R_{t-1} \rvert` | `compute_risk_volatility_criterion`: `risk_profile=np.array(self.risk_profile)`; `diffs=np.abs(np.diff(risk_profile))`; `total_jerk=float(np.sum(diffs))` | Exact implementation in `maintenance_scheduling_model.py:1022-1035` |
| `c4` Robustness Slack | `min_{t,r} (Cmax - L)/(Cmax - Cmin)` | `_compute_bottleneck_slack_profile` and `min(...)` in `compute_raw_criteria` | direct match |
| `c5` Ski Season Risk | `sum_{t in winter} sum_{i in A_t} rbar_{i,t} * mu_ski(i)` | winter loop + membership weighting in `compute_raw_criteria` | uses `ski_area_proximity_membership` from `spatial_memberships.py` |
| `c6` Summer Eco-Stress | `sum_{t in summer} max(0, E_t - median(E))` | `compute_summer_eco_stress_criterion` | `E_t` is sum of park-memberships over active interventions |
| `c7` Social Weekend Risk | count of risky interventions starting on weekends | `compute_raw_criteria` | risky = above P80 intervention mean risk |

### Important object attributes used in criteria computation

Classes and fields:
- `InterventionScheduleOption` (`maintenance_scheduling_model.py`)
  - `start_time`, `duration`, `workloads`, `risks`, `mean_risk`, `worst_risk`, `total_workload`
- `Intervention`
  - `name`, `latest_start_timestep`, `options`, `mean_risk_profile`, `mean_workload_size`
- `MaintenanceSchedulingInstance`
  - `T`, `scenarios_number`, `resources`, `seasons`, `interventions`, `exclusions`
- `MaintenanceSchedule`
  - `intervention_start_times`
  - derived runtime fields:
    - `active_interventions_by_timestep`
    - `active_intervention_counts`
    - `risk_profile`
    - `raw_criteria_values`

---

## 2.3 Section 03: Alternative Generation and Raw Matrix Assembly

### Schedule pool generation from solver outputs
- File: `data/Alternatives/Solvers/execute.py`
- Produced files consumed by decision-matrix builder:
  - `data/Alternatives/X12_Original_Execution/Team*/sol*_t*_X_12_s*.txt`

### Schedule loading into `MaintenanceSchedule` objects
- File: `decision_matrix_builder.py`
- `build_raw_decision_matrix(...)` creates one `MaintenanceSchedule` per file path.
- `MaintenanceSchedule.__init__` parses each text file and builds `intervention_start_times`.

### Instance loading into structured classes
- File: `maintenance_scheduling_model.py`
- `load_maintenance_instance_from_json(...)`
  - creates `MaintenanceSchedulingInstance`
  - populates `Resource`, `Season`, `Intervention`, `InterventionScheduleOption`, `Exclusion`.

### Deduplication logic (post-metrics)
- File: `decision_matrix_builder.py::build_raw_decision_matrix`
- Dedup key = tuple of raw metric values.
- Keeps only unique metric rows before writing matrix.
- In the current repository state, this matches the schedule-level uniqueness:
  - `41` non-empty solver files,
  - `31` unique schedules,
  - `31` unique raw-metric rows.

---

## 2.4 Section 03: Matrix Normalization

Paper normalization:
- benefit criterion (`c4`): divide by max
- cost criteria: inverted max-normalization

Code mapping:
- `wowa_stability_analysis.py::WOWAStabilityAnalyzer.load_and_normalize_decision_matrix`
- `directions` explicitly marks:
  - maximize: `"Robustness Slack"`
  - minimize: `"Reliability Stress"`, `"Ski Season Risk"`, `"Social Weekend Risk"`, `"Max Critical Overlap"`, `"Risk Volatility"`, `"Summer Eco-Stress"`

Implemented formulas in the current repository state:
- maximize:
  - `x_norm = x / max(x)`
- minimize:
  - `x_norm = 1 - (x - min(x)) / max(x)`

Important implementation note:
- the minimizing formula above is the exact code path currently used.
- therefore minimizing criteria are not necessarily mapped to `[0,1]`; their lower endpoint is `min(x)/max(x)` when `max(x) > 0`.

Stored outputs:
- `self.normalized_decision_matrix`: full raw dataframe with normalized criterion columns
- `self.normalized_criteria_matrix`: only the seven normalized criterion columns, in fixed order

---

## 2.5 Section 02 (manuscript Section 02): WOWA Unified Grid Representation

Paper theorem ingredients:
- unified cumulative-importance grid `Omega`
- transformed vectors `z_i`
- common-weight representation over that grid.

Code mapping:
- `wowa_stability_analysis.py::build_unified_importance_grid`
  - constructs union of breakpoints from cumulative importance after sorting criteria values for each alternative.
  - this is the implementation equivalent of `Omega`.
- `wowa_stability_analysis.py::build_transformed_alternative_matrix`
  - builds `Psi` (rows are transformed alternative vectors over grid segments).
  - each segment gets the criterion value covering that cumulative-importance interval.
  - this is the implementation analogue of `z_i`.

Implementation note:
- In this implementation, the LP works directly with the unified weight vector (`w[k]` over the grid segments).
- A quantifier `Q` is not required for this optimization path: `Q` would only be needed if one wants to pre-parameterize/fix `W` via `W_k = Q(t_k)-Q(t_{k-1})`.
- Here, `W` is determined by optimization under simplex + orness + dominance constraints.

### Where the unified grid `Omega` is built, step by step

The exact execution path is:
1. `load_and_normalize_decision_matrix()` reads `output/csv/decision_matrix_raw.csv` and prepares `self.normalized_criteria_matrix`.
2. `build_unified_importance_grid()` gets the criterion-importance vector in that exact column order through `get_ordered_importance_vector()`.
3. For each alternative row:
   - sort the seven normalized criterion values in descending order,
   - reorder the importance vector with the same permutation,
   - compute cumulative importances with `np.cumsum(...)`.
4. Collect all those cumulative sums across all alternatives, then add `0.0` and `1.0`.

Result:
- the final sorted breakpoint list is stored in `self.unified_importance_grid`
- this is the actual `Omega` used later by the LP

### Where the transformed matrix `Psi` is built

`build_transformed_alternative_matrix()` uses `self.unified_importance_grid` and creates one row of `Psi` per alternative:
1. For each alternative, sort normalized criterion values descending.
2. Reorder criterion importances the same way and compute cumulative importances.
3. For each segment `[Omega_k, Omega_{k+1}]`, find which sorted criterion covers the segment end.
4. Assign that sorted criterion value to `Psi[i, k]`.

Stored arrays:
- `self.transformed_alternative_matrix`: the full `Psi`
- `self.orness_coefficients`: discrete orness coefficients `nu_k = (K-k)/(K-1)` used by the current LP

Important note:
- the active orness constraints use `self.orness_coefficients`, because this implementation optimizes the segment-weight vector `w` directly rather than recovering it from a separate BUM quantifier `Q`.

---

## 2.6 Section 04 (manuscript Section 04): Stability Radius LP

Paper LP variables:
- weight vector `W`
- stability radius `rho_i` per candidate alternative.

Code mapping:
- `wowa_stability_analysis.py::solve_stability_radius_lp(alpha_min, alpha_max)`
  - loops each candidate alternative `i_cand`
  - builds one LP model per candidate.

### Exact Gurobi objects created in code

For each candidate alternative:
- `model = gp.Model("Stability")`
- `w = model.addVars(k_segments, lb=0.0, ub=1.0, name="w")`
- `rho = model.addVar(lb=0.0, name="rho")`

So the LP has:
- one continuous variable `w_k` per segment of `Omega`
- one continuous scalar `rho`

### Objective function

Defined exactly with:
```text
model.setObjective(rho, GRB.MAXIMIZE)
```

So the solver maximizes the largest admissible stability radius around a candidate winner.

### Constraint block 1: simplex feasibility of the weight vector

Defined with:
```text
sum_k w_k = 1
0 <= w_k <= 1
```

Code location:
- `model.addConstr(gp.quicksum(w[k] for k in range(k_segments)) == 1.0, "SumWeights")`
- lower/upper bounds are already embedded in `model.addVars(...)`

### Constraint block 2: orness interval

The current LP uses the discrete coefficient vector:
```text
nu_k = (K-k)/(K-1)
```
with `K = k_segments`.

Code path:
1. `build_transformed_alternative_matrix()` computes `self.orness_coefficients`
2. `solve_stability_radius_lp(...)` builds
```text
orness_expr = sum_k w_k * nu_k
```
3. Then adds:
```text
alpha_min <= orness_expr <= alpha_max
```

### Constraint block 3: simplex-face margin

Purpose:
- `rho` is not only a feasibility radius.
- it is forced to stay at least `rho` away from every simplex face `w_k = 0`.

Code:
```text
face_margin = ||P e_k||_2
w_k >= rho * face_margin
```

Implementation detail:
- the code computes `face_margin` once using the first basis vector because all simplex faces are symmetric after projection onto the simplex hyperplane.

### Constraint block 4: orness-boundary margin

Purpose:
- force the solution to stay at least `rho` away from both hyperplanes
  - `orness = alpha_min`
  - `orness = alpha_max`

Code:
```text
nu_margin = ||P nu||_2
orness_expr - alpha_min >= rho * nu_margin
alpha_max - orness_expr >= rho * nu_margin
```

These constraints are skipped only if `nu_margin == 0`.

### Constraint block 5: dominance margin against every rival

For each rival alternative `j != i`:
1. build the transformed-space difference
```text
diff = Psi_i - Psi_j
```
2. compute the projected norm
```text
dist = ||P diff||_2
```
3. add the dominance-margin constraint
```text
sum_k w_k * diff_k >= rho * dist
```

This is the exact place where pairwise robust dominance is encoded in Gurobi.

### Why projected norms appear in the LP

The helper functions:
- `project_to_simplex_hyperplane(vec)`
- `projected_simplex_norm(vec)`

implement the geometry of distances inside the affine simplex `sum_k w_k = 1`.

Interpretation:
- `||P e_k||_2` gives the correct Euclidean scaling for distance to simplex faces
- `||P nu||_2` gives the correct Euclidean scaling for distance to the orness hyperplanes inside that same affine space
- `||P(Psi_i - Psi_j)||_2` gives the correct Euclidean scaling for distance to a dominance-separating hyperplane

### Solver output and winner selection

If the model is optimal, the script stores:
- `Alternative`
- `Rho`
- `Optimized_Orness`
- `Is_Stable = (Rho > 1e-5)`

Then `main()`:
1. defines two importance scenarios (`Baseline`, `Sustainability`)
2. defines four orness profiles:
   - `Risk_Averse = [0.00, 0.35]`
   - `Neutral = [0.35, 0.65]`
   - `Optimistic = [0.65, 1.00]`
   - `Full_Spectrum = [0.00, 1.00]`
3. solves one LP per candidate alternative and per profile
4. sorts results by `Rho` descending
5. keeps the first row of each profile as the robust winner

This means the winner is **not** the alternative with the highest score under one fixed weight vector.
It is the alternative whose best admissible Chebyshev-like safety radius `rho` is largest in the feasible weight region.

---

## 3. Pseudocode of Core Long Routines

## 3.1 Decision matrix build (`decision_matrix_builder.build_raw_decision_matrix`)

```text
load instance JSON -> MaintenanceSchedulingInstance
load all solution txt files -> MaintenanceSchedule objects

load intervention points + keys
load park geometries and ski geometries
compute fuzzy proximities:
    mu_ski(i), mu_park(i)

for each solution:
    compute active interventions by timestep
    compute raw metrics c1..c7

deduplicate solutions by raw metric tuple
create dataframe rows {Alternative, raw_criteria_values}
save output/csv/decision_matrix_raw.csv/.md
```

## 3.2 Raw metrics (`MaintenanceSchedule.compute_raw_criteria`)

```text
ensure active_intervention_counts and daily risk are precomputed

c1 reliability_stress = sum max(0, R_t - P90(R_t>0))
c4 robustness_slack = min_t,r normalized slack(t,r)
c5 ski_season_risk = winter sum of risk(i,t) * mu_ski(i)
c7 social_weekend_risk = count starts on weekend with mean_risk > P80

c2 max_critical_overlap = max_t count(active critical interventions)
c3 risk_volatility = sum |R_t - R_t-1|
c6 eco_stress = summer sum max(0, E_t - median(E))

return dict with 7 criteria
```

## 3.3 Transformed representation + stability LP (`WOWAStabilityAnalyzer`)

```text
normalize raw matrix to benefit scale
for each alternative:
    sort criteria values descending
    accumulate importances in that sorted order
collect all cumulative points -> Omega grid

for each alternative and each grid segment:
    assign segment value from covering sorted criterion
store in Psi

for each candidate alternative i:
    maximize rho
    subject to:
      simplex constraints on w
      orness interval bounds
      simplex-face margin constraints
      orness-boundary margin constraints
      dominance margin vs every rival using projected Psi differences
solve with Gurobi
```

---

## 4. Formula-to-Code Quick Index

| Paper source | Formula / concept | Code reference |
|---|---|---|
| manuscript Section 03 | sign-concordance similarity | `spatial_reconstruction.py::compute_sign_concordance` |
| manuscript Section 03 | `d=(1-r)/2` | `spatial_reconstruction.py::transform_correlation_to_dissimilarity(method="linear")` |
| manuscript Section 03 | non-metric SMACOF MDS | `spatial_reconstruction.py::compute_weighted_mds_embedding` |
| manuscript Section 03 | fuzzy spatial membership `mu_Z` | `spatial_memberships.py::compute_feature_proximity_membership` |
| manuscript Section 03 | `c1..c7` | `maintenance_scheduling_model.py::MaintenanceSchedule.compute_raw_criteria` + helper methods |
| manuscript Section 03 | raw matrix build | `decision_matrix_builder.py::build_raw_decision_matrix` (alias `build_raw_decision_matrix`) |
| manuscript Section 03 | normalization | `wowa_stability_analysis.py::WOWAStabilityAnalyzer.load_and_normalize_decision_matrix` |
| manuscript Section 02 | unified grid `Omega` | `wowa_stability_analysis.py::build_unified_importance_grid` |
| manuscript Section 02 | transformed vectors (`z_i`) | `wowa_stability_analysis.py::build_transformed_alternative_matrix` (`Psi`) |
| manuscript Section 04 | stability LP objective and constraints | `wowa_stability_analysis.py::solve_stability_radius_lp` |

---

## 5. Practical Differences Between Manuscript and Current Code

These are important for strict reproducibility claims.

1. The LP is intentionally formulated on the unified weight vector `W` (segment weights `w`).
- This is consistent with the robust-selection objective in this repository.
- A quantifier `Q` is optional and only needed to generate a fixed parametric `W`, not to solve the stability LP used here.

2. The current LP implementation matches the repository's active Section 04 geometry.
- It includes:
  - simplex feasibility,
  - orness bounds using `nu_k = (K-k)/(K-1)`,
  - simplex-face projected margins,
  - projected orness-boundary margins,
  - projected dominance margins.

3. Current decision matrix has 7 raw criteria.
- Active matrix columns are 7 (`Reliability Stress`, `Robustness Slack`, `Ski Season Risk`, `Social Weekend Risk`, `Max Critical Overlap`, `Risk Volatility`, `Summer Eco-Stress`).

4. Current raw metrics are based on ski/park proximity memberships and the corrected 2025 calendar mapping.
- The active pipeline uses `compute_ski_area_proximity_membership(...)` and `compute_park_proximity_membership(...)` for spatial context.
- `c5` uses the winter window reconstructed from the calendar mapping.
- `c6` uses the May-August summer window.
- `c7` checks weekend overlap at timestep-interval level.

---

## 6. File-Level Responsibility Map

- `data/Alternatives/Solvers/execute.py`
  - solver-run orchestration and raw schedule generation.
- `spatial_reconstruction.py`
  - risk-based similarity, dissimilarity, weighted non-metric MDS, point export.
- `spatial_memberships.py`
  - geospatial loading (France, parks, ski stations), proximity membership computation.
- `maintenance_scheduling_model.py`
  - domain classes, instance loading, schedule parsing, criterion computation (`c1..c7`).
- `decision_matrix_builder.py`
  - end-to-end decision-matrix assembly and dedup.
- `wowa_stability_analysis.py`
  - WOWA normalization, transformed-space construction, stability-radius LP.
- `plots_and_tables/`
  - scripts used to generate paper figures and appendix tables.
- `archive/scripts_exploration/choose_importance.py`
  - exhaustive importance-vector search over orness profiles.
- `archive/scripts_exploration/analyze_importance_search.py`
  - post-analysis of winner variability across importance blocks.
- `plots_and_tables/export_normalized_decision_matrix_to_latex.py`
  - export normalized matrix to LaTeX table for appendix material.

---

## 7. Minimal Reproduction Commands

```bash
# 1) (Optional) run external solver batch harness for the compiled binaries
python "data/Alternatives/Solvers/execute.py"

# 2) Build or refresh spatial embedding (.npy files required by the decision-matrix builder)
python spatial_reconstruction.py

# 3) Build raw decision matrix
python decision_matrix_builder.py

# 4) Run WOWA stability-radius analysis across scenarios/profiles
python wowa_stability_analysis.py

# 5) (Optional) importance sensitivity search
python archive/scripts_exploration/choose_importance.py
python archive/scripts_exploration/analyze_importance_search.py
```
