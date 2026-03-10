from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_memberships import (
    compute_park_proximity_membership,
    compute_ski_area_proximity_membership,
    load_regional_natural_parks,
    load_ski_stations,
)
from maintenance_scheduling_model import (
    MaintenanceSchedule,
    load_maintenance_instance_from_json,
)
from project_paths import (
    DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH,
    DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
    INSTANCE_PATH,
    REGIONAL_NATURAL_PARKS_CSV_PATH,
    RAW_DECISION_MATRIX_CSV_PATH,
    RAW_DECISION_MATRIX_MD_PATH,
    X12_DIR,
    ensure_output_tree,
)


RAW_CRITERION_COLUMNS = [
    "Reliability Stress",
    "Robustness Slack",
    "Ski Season Risk",
    "Social Weekend Risk",
    "Max Critical Overlap",
    "Risk Volatility",
    "Summer Eco-Stress",
]


def compute_schedule_difference_matrix(
    schedules: List[MaintenanceSchedule], plot: bool = False
) -> np.ndarray:
    """
    Build the pairwise difference matrix for intervention start times.
    """
    n = len(schedules)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i, n):
            keys = set(schedules[i].intervention_start_times.keys()).union(
                schedules[j].intervention_start_times.keys()
            )
            diff_count = sum(
                1
                for key in keys
                if schedules[i].intervention_start_times.get(key)
                != schedules[j].intervention_start_times.get(key)
            )
            matrix[i, j] = diff_count
            matrix[j, i] = diff_count

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.title("Intervention Start Time Differences")
        plt.xlabel("Schedule Index")
        plt.ylabel("Schedule Index")
        plt.colorbar(label="Difference Count")
        plt.tight_layout()
        plt.show()

    return matrix


def build_raw_decision_matrix(
    instance_path: str | Path,
    schedule_paths: Sequence[str | Path],
    embedding_points_path: str | Path = DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
    embedding_keys_path: str | Path = DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH,
    parks_path: str | Path = REGIONAL_NATURAL_PARKS_CSV_PATH,
    plots: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the raw decision matrix for maintenance scheduling alternatives.
    """
    instance_path = Path(instance_path)
    embedding_points_path = Path(embedding_points_path)
    embedding_keys_path = Path(embedding_keys_path)
    parks_path = Path(parks_path)

    if verbose:
        print(f"Loading instance: {instance_path}")
    with instance_path.open("r", encoding="utf-8") as f:
        instance_data = json.load(f)
    instance = load_maintenance_instance_from_json(instance_data, verbose=False)

    if verbose:
        print(f"Loading {len(schedule_paths)} solution files...")
    schedules: List[MaintenanceSchedule] = []
    for schedule_path in schedule_paths:
        schedule_path = Path(schedule_path)
        schedule = MaintenanceSchedule(str(schedule_path))
        setattr(schedule, "schedule_id", schedule_path.name)
        schedules.append(schedule)

    if verbose:
        print("Computing fuzzy spatial memberships...")
    embedding_points = np.load(embedding_points_path)
    intervention_keys = np.load(embedding_keys_path, allow_pickle=True)

    try:
        parks_gdf = load_regional_natural_parks(parks_path)
    except Exception as e:
        print(f"Error loading parks data from {parks_path}: {e}")
        return pd.DataFrame()

    try:
        ski_stations = load_ski_stations()
        ski_area_proximity_membership = compute_ski_area_proximity_membership(
            embedding_points,
            intervention_keys,
            ski_stations,
        )
    except Exception as e:
        print(f"Error loading ski stations or memberships: {e}")
        ski_area_proximity_membership = pd.Series(dtype=float)

    park_proximity_membership = compute_park_proximity_membership(
        embedding_points,
        intervention_keys,
        parks_gdf,
        near_full_m=1000.0,
        near_zero_m=10000.0,
    )

    for schedule in tqdm(schedules, desc="Computing criteria"):
        schedule.compute_active_interventions(instance)
        schedule.compute_raw_criteria(
            instance,
            ski_area_proximity_membership,
            park_proximity_membership,
        )

    unique_schedules: List[MaintenanceSchedule] = []
    seen_criteria = set()
    for schedule in schedules:
        criteria_tuple = tuple(
            schedule.raw_criteria_values.get(column, 0.0)
            for column in RAW_CRITERION_COLUMNS
        )
        if criteria_tuple in seen_criteria:
            continue
        seen_criteria.add(criteria_tuple)
        unique_schedules.append(schedule)

    if verbose:
        removed = len(schedules) - len(unique_schedules)
        print(
            f"Deduplication complete: removed {removed} duplicates, "
            f"kept {len(unique_schedules)} alternatives."
        )
    schedules = unique_schedules

    if plots:
        compute_schedule_difference_matrix(schedules, plot=True)

    alternatives = []
    for schedule in schedules:
        row_data = {}
        schedule_id = cast(str, getattr(schedule, "schedule_id", ""))
        match = re.match(r"sol(\d+)_t(\d+)_X_12_s(\d+)\.txt", schedule_id)
        if match:
            duration, team, seed = match.groups()
            alt_name = f"T{team}_D{duration}_S{seed}"
        else:
            alt_name = schedule_id

        row_data["Alternative"] = alt_name
        row_data.update(getattr(schedule, "raw_criteria_values", {}))
        row_data.update(getattr(schedule, "fuzzy_criteria_values", {}))
        alternatives.append(row_data)

    decision_matrix = pd.DataFrame(alternatives)
    if not decision_matrix.empty:
        decision_matrix.set_index("Alternative", inplace=True)
    return decision_matrix


def collect_solution_paths(base_path: str | Path = X12_DIR) -> List[str]:
    """
    Return non-empty solution text files under the X12 directory tree.
    """
    base_path = Path(base_path)
    solution_paths: List[str] = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".txt"):
                continue
            full_path = Path(root) / file
            if full_path.stat().st_size > 0:
                solution_paths.append(str(full_path))

    return sorted(solution_paths)


def write_markdown_table(df: pd.DataFrame, filename: str | Path, title: str) -> None:
    markdown_table = f"# {title}\n\n"
    header = "| |" + "|".join(df.columns) + "|\n"
    separator = "|---|" + "|".join([":---:" for _ in df.columns]) + "|\n"
    markdown_table += header + separator

    for idx, row in df.iterrows():
        formatted_row = [
            f"{val:.3f}" if isinstance(val, (int, float)) else str(val) for val in row
        ]
        row_str = f"|{idx}|" + "|".join(formatted_row) + "|\n"
        markdown_table += row_str

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(markdown_table, encoding="utf-8")


def main() -> None:
    ensure_output_tree()

    print("Building raw decision matrix...")
    schedule_paths = collect_solution_paths(base_path=X12_DIR)
    if not schedule_paths:
        print(f"No non-empty solution files found in {X12_DIR}.")
        return

    decision_matrix = build_raw_decision_matrix(
        instance_path=INSTANCE_PATH,
        schedule_paths=schedule_paths,
        embedding_points_path=DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
        embedding_keys_path=DEFAULT_SPATIAL_RECONSTRUCTION_KEYS_PATH,
        parks_path=REGIONAL_NATURAL_PARKS_CSV_PATH,
        plots=False,
    )

    if decision_matrix.empty:
        print("Decision matrix could not be generated.")
        return

    raw_columns = [c for c in decision_matrix.columns if c in RAW_CRITERION_COLUMNS]
    decision_matrix_raw = pd.DataFrame(decision_matrix.loc[:, raw_columns])
    decision_matrix_raw.to_csv(RAW_DECISION_MATRIX_CSV_PATH)
    write_markdown_table(
        decision_matrix_raw,
        RAW_DECISION_MATRIX_MD_PATH,
        "Raw Decision Matrix",
    )

    print(f"Saved CSV: {RAW_DECISION_MATRIX_CSV_PATH}")
    print(f"Saved Markdown: {RAW_DECISION_MATRIX_MD_PATH}")


if __name__ == "__main__":
    main()
