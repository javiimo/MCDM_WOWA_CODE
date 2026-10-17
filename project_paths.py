from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
ALTERNATIVES_DIR = DATA_DIR / "Alternatives"
X12_DIR = ALTERNATIVES_DIR / "X12_Original_Execution"
SOLVERS_DIR = ALTERNATIVES_DIR / "Solvers"
INSTANCE_PATH = ALTERNATIVES_DIR / "downloaded" / "X_12.json"
RESULTS_DIR = ALTERNATIVES_DIR / "Results"
RESULTS_CSV_PATH = RESULTS_DIR / "results.csv"

FRANCE_MAP_LOCATIONS_DIR = DATA_DIR / "france_map_and_locations"
FRANCE_REGIONS_GEOJSON_PATH = FRANCE_MAP_LOCATIONS_DIR / "france.json"
REGIONAL_NATURAL_PARKS_CSV_PATH = FRANCE_MAP_LOCATIONS_DIR / "pnr_polygon.csv"
SKI_STATIONS_GEOJSON_PATH = FRANCE_MAP_LOCATIONS_DIR / "stations.geojson"
SKI_STATIONS_XLSX_PATH = FRANCE_MAP_LOCATIONS_DIR / "ski_stations.xlsx"

OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_CSV_DIR = OUTPUT_DIR / "csv"
OUTPUT_MD_DIR = OUTPUT_DIR / "md"
OUTPUT_TEX_DIR = OUTPUT_DIR / "tex"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_PLOTS_AND_TABLES_DIR = OUTPUT_FIGURES_DIR / "plots_and_tables"
OUTPUT_NPY_DIR = OUTPUT_DIR / "npy"
OUTPUT_LOGS_DIR = OUTPUT_DIR / "logs"

RAW_DECISION_MATRIX_CSV_PATH = OUTPUT_CSV_DIR / "decision_matrix_raw.csv"
RAW_DECISION_MATRIX_MD_PATH = OUTPUT_MD_DIR / "decision_matrix_raw.md"
NORMALIZED_DECISION_MATRIX_CSV_PATH = OUTPUT_CSV_DIR / "decision_matrix_normalized_wowa.csv"
WOWA_STABILITY_SCENARIOS_CSV_PATH = OUTPUT_CSV_DIR / "wowa_scenarios_comparison.csv"
WOWA_STABILITY_RESULTS_CSV_PATH = OUTPUT_CSV_DIR / "wowa_stability_results.csv"
WOWA_GRID_SUMMARY_CSV_PATH = OUTPUT_CSV_DIR / "wowa_grid_summary.csv"
WOWA_CHOICE_SET_SIZE_CSV_PATH = OUTPUT_CSV_DIR / "wowa_choice_set_size.csv"
WOWA_SHAPLEY_DEVIATION_CSV_PATH = OUTPUT_CSV_DIR / "wowa_shapley_deviation.csv"
EXTENSION_DECISION_MATRIX_CSV_PATH = OUTPUT_CSV_DIR / "decision_matrix_raw_extension.csv"
WOWA_EXTENSION_CSV_PATH = OUTPUT_CSV_DIR / "wowa_extension_results.csv"
WOWA_EXTENSION_DRAWS_CSV_PATH = OUTPUT_CSV_DIR / "wowa_extension_mixed_draws.csv"
WOWA_EXTENSION_LP_SIZE_CSV_PATH = OUTPUT_CSV_DIR / "wowa_extension_lp_size.csv"
MDS_STABILITY_CSV_PATH = OUTPUT_CSV_DIR / "mds_stability.csv"
MDS_SELECTION_CSV_PATH = OUTPUT_CSV_DIR / "mds_selection.csv"
SUPPLEMENTARY_CSV_DIR = OUTPUT_CSV_DIR / "supplementary"


PLOTS_AND_TABLES_SPATIAL_RECONSTRUCTION_MAP_PATH = OUTPUT_PLOTS_AND_TABLES_DIR / "interventions_map.png"
PLOTS_AND_TABLES_WINNERS_RADAR_PATH = OUTPUT_PLOTS_AND_TABLES_DIR / "winners_radar_chart.png"
ERROR_LOG_PATH = OUTPUT_LOGS_DIR / "error.log"

DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH = OUTPUT_NPY_DIR / "points_20250329_203043.npy"
DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH = OUTPUT_NPY_DIR / "points_keys_20250329_203043.npy"


def instance_path(name: str) -> Path:
    return ALTERNATIVES_DIR / "downloaded" / f"X_{name[1:]}.json"


def embedding_paths(name: str) -> tuple[Path, Path]:
    if name == "X12":
        return DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH, DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH
    return OUTPUT_NPY_DIR / f"points_{name}.npy", OUTPUT_NPY_DIR / f"points_keys_{name}.npy"


def ensure_output_tree() -> None:
    for folder in (
        OUTPUT_DIR,
        OUTPUT_CSV_DIR,
        SUPPLEMENTARY_CSV_DIR,
        OUTPUT_MD_DIR,
        OUTPUT_TEX_DIR,
        OUTPUT_FIGURES_DIR,
        OUTPUT_PLOTS_AND_TABLES_DIR,
        OUTPUT_NPY_DIR,
        OUTPUT_LOGS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)

