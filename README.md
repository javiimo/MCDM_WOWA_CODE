# Stability Radii for Multi-Criteria Choice under WOWA

Code and data of the paper *Stability Radii for Multi-Criteria Choice under Weighted Ordered
Weighted Averaging, with an Application to Power Grid Maintenance Scheduling*.

The method selects, from a finite set of alternatives evaluated on several criteria, the one
that remains the best under the widest range of WOWA aggregation models. For a weighting
vector $\mathbf p$, every WOWA operator is a point $\mathbf W$ of a common simplex defined on
the grid of all subset weights $\mathbf p(B)$, the score of every alternative is linear in
$\mathbf W$, and the orness and the Shapley importance of the criteria are linear in $\mathbf W$
too. The stability radius of an alternative is the radius of the largest ball of admissible
models in which it is the best, and it is computed with one Linear Program. No quantifier has
to be chosen: the decision maker gives $\mathbf p$ and, optionally, an orness interval and
bounds on the importance of the criteria.

The application selects maintenance schedules for the ROADEF/EURO 2020 challenge (instance
X12, with X03 and X13 as extensions), from candidates produced by three competition solvers.

[explanation.md](explanation.md) maps the notation, results and experiments of the paper to the code.

## Repository Layout

| Path | Content |
|---|---|
| `wowa_stability_analysis.py` | the method (`WOWAStabilityAnalyzer`) and the main experiment on X12 |
| `wowa_extension_analysis.py` | selection on the candidate sets of X03, X12 and X13 for several objective functions |
| `mds_stability_analysis.py` | uncertainty of the spatial reconstruction across MDS initializations |
| `supplementary_analyses/` | checks of the map, patterns of stability, sensitivity to $\mathbf p$ and to the norm, dual multipliers |
| `plots_and_tables/` | every table (LaTeX) and figure of the paper |
| `maintenance_scheduling_model.py` | instances, schedules and criteria $c_1,\dots,c_7$ |
| `decision_matrix_builder.py` | raw decision matrices |
| `spatial_reconstruction.py`, `spatial_memberships.py`, `fuzzy_sets.py` | MDS map of the interventions and fuzzy proximity memberships |
| `RTE_ChallengeROADEF2020_checker.py` | official checker of the challenge |
| `project_paths.py` | input and output paths |
| `data/` | candidate schedules, solver runner, map layers, reference MDS map of X12 |
| `output/` | generated files (`csv/`, `tex/`, `figures/`, `npy/`, `md/`), not tracked |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Linear Programs are solved with Gurobi (`gurobipy`), which needs a license (free for
academic use). The MDS map uses the R package `smacof`, through `rpy2`
(`spatial_reconstruction.py`) or through the `Rscript` executable given by the `RSCRIPT`
environment variable (MDS seeds and checks of the map).

## Using the Method

```python
from wowa_stability_analysis import WOWAStabilityAnalyzer, shapley_bounds_around_weights

p = {"Reliability Stress": 0.10, "Max Critical Overlap": 0.25, "Risk Volatility": 0.10,
     "Resource Slack": 0.20, "Ski Season Risk": 0.10, "Summer Eco-Stress": 0.10,
     "Social Weekend Risk": 0.15}
analyzer = WOWAStabilityAnalyzer("output/csv/decision_matrix_raw.csv", p)
radii = analyzer.solve_stability_radius_lp(
    orness_min=0.35, orness_max=0.65,                     # attitudinal profile
    shapley_bounds=shapley_bounds_around_weights(p, 0.1),  # optional importance requirements
)
print(radii.sort_values("Rho", ascending=False).head())
```

Each row gives the radius $\rho_i$, the center $\mathbf W_i^*$ with its orness and Shapley
importance, and the constraints and competitor that limit the radius. The criteria and
their directions are listed in `CRITERIA_DIRECTIONS`.

## Reproducing the Paper

From the repository root, in this order:

```bash
python decision_matrix_builder.py                 # raw decision matrix of X12
python decision_matrix_builder.py --extension     # raw decision matrices of X03, X12 and X13
python wowa_stability_analysis.py                 # main experiment
python wowa_extension_analysis.py                 # extended candidate sets
python plots_and_tables/generate_appendix_tables.py   # appendix tables (competition scores, read below)
python mds_stability_analysis.py                  # MDS seeds (needs RSCRIPT)
python supplementary_analyses/mds_checks.py       # checks of the map (needs RSCRIPT)
python supplementary_analyses/weighted_dominance.py
python supplementary_analyses/patterns.py
python supplementary_analyses/norm_and_weight_sensitivity.py   # about 30 min on 8 cores
python supplementary_analyses/fragility_dual_check.py
python plots_and_tables/plot_stability_results.py
python plots_and_tables/supplementary_tables.py
python plots_and_tables/plot_robust_alternatives_radar.py
```

The reference MDS map of the X12 interventions (seed 42) is provided in
`data/spatial_reconstruction/`, so `spatial_reconstruction.py` is not needed to reproduce
the paper; the maps of X03 and X13 are computed by the extension (needs RSCRIPT).
The decision matrices need the instances (see below), and the extension needs the
calendars of `data/Alternatives/Results/`. Tables are written to `output/tex/` and figures
to `output/figures/plots_and_tables/`. The remaining scripts of `plots_and_tables/` draw the
illustrative figures of the theory (orness level curves, weight simplex, OWA and WOWA
weights) and the map of the interventions.

## Data and External Assets

- Schedule pool: `data/Alternatives/X12_Original_Execution/`
- Instance: `data/Alternatives/downloaded/X_12.json` (download it with `python data/Alternatives/Solvers/download_instances.py`)
- Solver runner and binaries: `data/Alternatives/Solvers/`
  The repository uses the compiled solver binaries published by the competition teams in their public releases/repositories.
- France map and locations data (parks + ski + map layers): `data/france_map_and_locations/`
- Constraint/objective checker from ROADEF context: `RTE_ChallengeROADEF2020_checker.py`

Please verify redistribution rights for third-party binaries/data before publishing derivative artifacts.

`data/Alternatives/X12_Original_Execution/execution.log` is the log of the original
X12 run. Team labels are renamed (original Team1/Team3/Team5 -> Team1/Team2/Team3),
and license metadata and personal paths are removed. The header records the SHA256
of the original file. Warnings and exceptions are kept as they were. All solvers
were run with 4 threads.

## Solver Experiment Reproduction

The solver experiment can be reproduced on x86-64 Linux with glibc. It was run on
LMDE (Debian-based), but nothing in it is specific to Debian.
Requirements:

- Python 3.11+ with NumPy, and `7z` (`p7zip-full` on Debian/Ubuntu).
- The shared libraries used by the solver binaries: `libstdc++`, `libgomp` (Team2) and `zlib` (Team3).
- Gurobi 9.1 (`libgurobi91.so`) and a valid license for Team1. Set them in
  `data/Alternatives/Solvers/config/runtime.json` (not tracked by Git):

  ```json
  {"team1": {"gurobi_home": "/path/to/gurobi912/linux64", "license_file": "/path/to/gurobi.lic"}}
  ```

From the repository root:

```bash
python data/Alternatives/Solvers/download_instances.py   # download and verify X03/X12/X13, check solver libraries
python -u data/Alternatives/Solvers/execute.py --run     # run every configuration
python data/Alternatives/Solvers/extract_results.py      # build data/Alternatives/Results/
```

`data/Alternatives/Solvers/config-target.json` defines the experiment:
3 instances x 3 solvers x 3 alphas x 3 seeds x 3 time budgets = 243 configurations,
all run with 4 threads (`omp_num_threads`).
If you run `execute.py` again, it resumes the run and skips completed jobs.
Raw runs are written to `data/Alternatives/Executions/` and are not tracked by Git.
Results may differ on other hardware; the goal is to reproduce the protocol,
not to get bit-identical solutions.

`data/Alternatives/Results/` contains:

- `<instance>/<team>/alpha<a>_seed<s>_time<t>.txt`: one calendar per configuration,
  validated with the official checker.
- `results.csv`: all 243 configurations, with objective values, durations, status and warnings.
- `execution.log`: all solver logs, in the order they ran. Personal paths, email
  addresses and license metadata are removed.

The published run gives 239 valid calendars out of 243 configurations. The four
without a calendar are Team3 on X12 with 300 s (alpha 0.5 and 0.75, seeds 33 and 42).
Team3 on X13 runs out of memory on the test machine (15 GiB RAM + 15 GiB swap).
Instead of letting the kernel OOM killer end the process, which could also kill the
supervisor or leave a truncated solution file, a memory guard stops the job when
available RAM plus free swap falls below 3 GiB. This happened in 26 runs, all after
about 150-230 s, so they did not reach their requested budget. Each one had already
written a valid calendar, which is kept. The `calendar_status`, `early_resource_stop`
and `solver_wall_seconds` columns of `results.csv` show which runs these are.

## External References

- ROADEF/EURO Challenge 2020 (Instance X12): https://roadef.org/challenge/2020/en/
- `RTE_ChallengeROADEF2020_checker.py` source: https://github.com/rte-france/challenge-roadef-2020
- Team 1 solver source: https://github.com/mbmv7/rc
- Team 2 solver source: https://github.com/FranciscoParrenoTorres/Roadef2020
- Team 3 solver source reference: https://github.com/mlangiu/challenge-roadef-2020
- Map of France geojson: https://www.data.gouv.fr/datasets/contours-administratifs
- French parks dataset source: https://www.data.gouv.fr/fr/datasets/parcs-naturels-regionaux-pnr-france-metropolitaine
- French ski stations dataset source: https://www.statistiques.developpement-durable.gouv.fr/pres-de-300-000-hectares-mobilises-par-les-stations-de-ski-au-coeur-despaces-naturels-0 

## Citation and License

If you use this repository, cite the associated paper and the external solver/data sources listed in the manuscript.  
License: MIT (see [LICENSE](LICENSE)).
