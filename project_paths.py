from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
ALTERNATIVES_DIR = DATA_DIR / "Alternatives"
X12_DIR = ALTERNATIVES_DIR / "X12_Original_Execution"
SOLVERS_DIR = ALTERNATIVES_DIR / "Solvers"
INSTANCE_PATH = ALTERNATIVES_DIR / "downloaded" / "X_12.json"

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
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_LOGS_DIR = OUTPUT_DIR / "logs"

RAW_DECISION_MATRIX_CSV_PATH = OUTPUT_CSV_DIR / "decision_matrix_raw.csv"
RAW_DECISION_MATRIX_MD_PATH = OUTPUT_MD_DIR / "decision_matrix_raw.md"
NORMALIZED_DECISION_MATRIX_CSV_PATH = OUTPUT_CSV_DIR / "decision_matrix_normalized_wowa.csv"
WOWA_STABILITY_SCENARIOS_CSV_PATH = OUTPUT_CSV_DIR / "wowa_scenarios_comparison.csv"
IMPORTANCE_SEARCH_CSV_PATH = OUTPUT_CSV_DIR / "importance_search_results.csv"
FINAL_DISCRIMINATION_REPORT_PATH = OUTPUT_REPORTS_DIR / "final_discrimination_report.md"
SPATIAL_RECONSTRUCTION_MAP_PATH = OUTPUT_FIGURES_DIR / "interventions_map.png"
PLOTS_AND_TABLES_SPATIAL_RECONSTRUCTION_MAP_PATH = (
    OUTPUT_PLOTS_AND_TABLES_DIR / "interventions_map.png"
)
PLOTS_AND_TABLES_WINNERS_RADAR_PATH = (
    OUTPUT_PLOTS_AND_TABLES_DIR / "winners_radar_chart.png"
)
ERROR_LOG_PATH = OUTPUT_LOGS_DIR / "error.log"

DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH = OUTPUT_NPY_DIR / "points_20250329_203043.npy"
DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH = OUTPUT_NPY_DIR / "points_keys_20250329_203043.npy"


def ensure_output_tree() -> None:
    for folder in (
        OUTPUT_DIR,
        OUTPUT_CSV_DIR,
        OUTPUT_MD_DIR,
        OUTPUT_TEX_DIR,
        OUTPUT_FIGURES_DIR,
        OUTPUT_PLOTS_AND_TABLES_DIR,
        OUTPUT_NPY_DIR,
        OUTPUT_REPORTS_DIR,
        OUTPUT_LOGS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def latest_output_file(pattern: str, directory: Path = OUTPUT_NPY_DIR) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None
