# Fuzzy-MCDM for Maintenance Scheduling

This repository contains the code used to build and analyze a multi-criteria decision model for maintenance schedules derived from the ROADEF/EURO 2020 challenge context.
The current decision matrix is built from 31 unique alternatives obtained from Team 1, Team 2, and Team 3 solver runs.

The public workflow in this repository is:
1. Generate or reuse the spatial reconstruction `.npy` files.
2. Build the raw decision matrix from a solution pool.
3. Run WOWA-based stability analysis.
4. Generate paper plots and tables (optional, via `plots_and_tables/`).

Detailed paper-to-code traceability is documented in [explanation.md](explanation.md).

## Repository Layout

- `decision_matrix_builder.py`: builds the raw decision matrix.
- `wowa_stability_analysis.py`: runs normalization and WOWA stability LP analysis.
- `plots_and_tables/`: scripts used to generate paper figures and appendix tables.
- `maintenance_scheduling_model.py`, `spatial_memberships.py`, `fuzzy_sets.py`: core domain, spatial, and fuzzy-set utilities.
- `data/`: static inputs (solutions, instance, geospatial files).
- `output/`: generated artifacts (`csv/`, `md/`, `tex/`, `figures/`, `npy/`, `reports/`, `logs/`).
- `archive/`: exploratory or historical material kept for reference.
- `plots_and_tables/export_normalized_decision_matrix_to_latex.py`: helper script to export the normalized WOWA matrix to LaTeX.

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not use the provided virtual environment, ensure Python and required libraries are available.  
`wowa_stability_analysis.py` requires a working `gurobipy` installation and license.
`spatial_reconstruction.py` is required if you need to generate the `output/npy/points_*.npy` and `output/npy/points_keys_*.npy` files used by `decision_matrix_builder.py`; it requires R + `rpy2` + R package `smacof`.

## Minimal Execution

From the repository root:

```bash
python spatial_reconstruction.py
python decision_matrix_builder.py
python wowa_stability_analysis.py
```

`decision_matrix_builder.py` expects spatial reconstruction files (`points_*.npy` and `points_keys_*.npy`) under `output/npy/`. If they are not already present, generate them first with `python spatial_reconstruction.py`.

Main outputs are written under `output/`.

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
