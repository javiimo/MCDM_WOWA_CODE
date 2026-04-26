# Fuzzy-MCDM for Maintenance Scheduling

This repository contains the code used to build and analyze a multi-criteria decision model for maintenance schedules derived from the ROADEF/EURO 2020 challenge context.
The current decision matrix is built from 31 unique alternatives obtained from Team 1, Team 2, and Team 3 solver runs.

The public workflow in this repository is:
1. Build the raw decision matrix from a solution pool.
2. Run WOWA-based stability analysis.
3. Generate paper plots and tables (optional, via `plots_and_tables/`).

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
`spatial_reconstruction.py` is optional and requires R + `rpy2` + R package `smacof`.

## Minimal Execution

From the repository root:

```bash
python decision_matrix_builder.py
python wowa_stability_analysis.py
```

Main outputs are written under `output/`.

## Data and External Assets

- Schedule pool and instance: `data/Alternatives/X12/`
- Solver runner and binaries: `data/Alternatives/Solvers/`
- France map and locations data (parks + ski + map layers): `data/france_map_and_locations/`
- Constraint/objective checker from ROADEF context: `RTE_ChallengeROADEF2020_checker.py`

Please verify redistribution rights for third-party binaries/data before publishing derivative artifacts.

## External References

- ROADEF/EURO Challenge 2020 (Instance X12): https://roadef.org/challenge/2020/en/
- `RTE_ChallengeROADEF2020_checker.py` source: https://github.com/rte-france/challenge-roadef-2020
- Team 1 solver source: https://github.com/mbmv7/rc
- Team 2 solver source: https://github.com/FranciscoParrenoTorres/Roadef2020
- Team 3 solver source reference: https://rwth-aachen.sciebo.de/s/BoMYqRmy7GbwlIm
- Map of France geojson: https://www.data.gouv.fr/datasets/contours-administratifs
- French parks dataset source: https://www.data.gouv.fr/fr/datasets/parcs-naturels-regionaux-pnr-france-metropolitaine
- French ski stations dataset source: https://www.statistiques.developpement-durable.gouv.fr/pres-de-300-000-hectares-mobilises-par-les-stations-de-ski-au-coeur-despaces-naturels-0 

## Citation and License

If you use this repository, cite the associated paper and the external solver/data sources listed in the manuscript.  
License: MIT (see [LICENSE](LICENSE)).
