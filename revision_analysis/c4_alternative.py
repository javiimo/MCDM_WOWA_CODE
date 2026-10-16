"""
Sensitivity of the X12 selection to the definition of Resource Slack (c4).
The minimum bottleneck slack is zero for every candidate, so c4 is replaced by the
mean of the per-timestep bottleneck slack (low quantiles are also zero up to round-off). Writes output/csv/revision/c4_alternative.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mds_checks import all_radii, load_schedules  # noqa: E402
from maintenance_scheduling_model import load_maintenance_instance_from_json  # noqa: E402
from project_paths import OUTPUT_CSV_DIR, OUTPUT_DIR, RAW_DECISION_MATRIX_CSV_PATH  # noqa: E402

OUT_DIR = OUTPUT_CSV_DIR / "revision"


def main() -> None:
    with (OUTPUT_DIR / "X_12_mean_risk.json").open("r", encoding="utf-8") as f:
        instance = load_maintenance_instance_from_json(json.load(f))
    raw = pd.read_csv(RAW_DECISION_MATRIX_CSV_PATH, index_col=0)
    schedules = load_schedules(instance, set(raw.index))
    alt = raw.copy()
    for name, schedule in schedules.items():
        alt.loc[name, "Resource Slack"] = np.mean(schedule._compute_bottleneck_slack_profile(instance))
    rows = []
    for label, matrix in (("min slack (paper)", raw), ("mean slack", alt)):
        radii = all_radii(matrix)
        rows.append(radii[radii["Winner"]].assign(Definition=label))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_DIR / "c4_alternative.csv", index=False)
    print(alt["Resource Slack"].describe())
    print(out.pivot_table(index=["Scenario", "Profile"], columns="Definition", values="Alternative", aggfunc="first"))


if __name__ == "__main__":
    main()
