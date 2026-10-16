"""
Importance of the criteria in the Shapley sense (Section 5, answer to Reviewer 3).

For each scenario and profile:
- delta*: smallest max_j |phi_j - p_j| attainable in the profile (one LP);
- max_j |phi_j - p_j| at the center of the most stable alternative, for each Shapley tolerance;
- the most stable alternative with free importance and with tolerances 0.10, 0.05 and 0.01.

Reads output/csv/wowa_stability_results.csv; writes output/csv/revision/shapley_deviation.csv
and output/tex/revision/shapley_deviation_table.tex.
"""

import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_and_tables.plot_stability_results import CRITERIA, PROFILE_NAMES, tex_id  # noqa: E402
from project_paths import OUTPUT_CSV_DIR, OUTPUT_TEX_DIR, RAW_DECISION_MATRIX_CSV_PATH  # noqa: E402
from project_paths import WOWA_STABILITY_RESULTS_CSV_PATH  # noqa: E402
from wowa_stability_analysis import PROFILES, WOWAStabilityAnalyzer, build_default_importance_scenarios  # noqa: E402

SETTINGS = [("Free_Shapley", "free"), ("Main", "0.10"), ("Shapley_0.05", "0.05"), ("Shapley_0.01", "0.01")]


def min_shapley_deviation(analyzer, p, a_min, a_max):
    """delta* = min over admissible W of max_j |g_j^T W - p_j|."""
    k_segments = len(analyzer.unified_importance_grid) - 1
    model = gp.Model("ShapleyDeviation")
    model.setParam("OutputFlag", 0)
    w = model.addMVar(k_segments, lb=0.0)
    delta = model.addVar(lb=0.0)
    model.addConstr(w.sum() == 1.0)
    model.addConstr(analyzer.orness_coefficients @ w >= a_min)
    model.addConstr(analyzer.orness_coefficients @ w <= a_max)
    target = np.array([p[c] for c in analyzer.normalized_criteria_matrix.columns])
    model.addConstr(analyzer.shapley_matrix @ w - target <= delta)
    model.addConstr(target - analyzer.shapley_matrix @ w <= delta)
    model.setObjective(delta, GRB.MINIMIZE)
    model.optimize()
    return model.ObjVal


def main():
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    rows = []
    for scenario, p in build_default_importance_scenarios().items():
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), p)
        analyzer.build_transformed_alternative_matrix()
        target = np.array([p[c] for c in CRITERIA])
        for profile, a_min, a_max in PROFILES:
            row = {"Scenario": scenario, "Profile": profile,
                   "Delta_Star": min_shapley_deviation(analyzer, p, a_min, a_max)}
            for setting, label in SETTINGS:
                g = results[(results.Scenario == scenario) & (results.Profile == profile)
                            & (results.Setting == setting)]
                best = g[g.Rho >= g.Rho.max() - 1e-9]
                phi = best[[f"Shapley_{c}" for c in CRITERIA]].to_numpy()
                row[f"Winner_{label}"] = " / ".join(sorted(best.Alternative))
                row[f"Rho_{label}"] = g.Rho.max()
                row[f"Deviation_{label}"] = float(np.abs(phi - target).max())
            rows.append(row)
    df = pd.DataFrame(rows)
    (OUTPUT_CSV_DIR / "revision").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_DIR / "revision" / "shapley_deviation.csv", index=False)

    lines = []
    for _, r in df.iterrows():
        cells = [f"{r.Delta_Star:.3f}"] + [
            f"{tex_id(r[f'Winner_{label}'])} ({r[f'Deviation_{label}']:.3f})" for _, label in SETTINGS
        ]
        lines.append(f"{r.Scenario} & {PROFILE_NAMES[r.Profile]} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    header = (r"Scenario & Profile & $\delta^*$ & free & $\delta=0.10$ & $\delta=0.05$ & $\delta=0.01$")
    (OUTPUT_TEX_DIR / "revision").mkdir(parents=True, exist_ok=True)
    (OUTPUT_TEX_DIR / "revision" / "shapley_deviation_table.tex").write_text(
        f"\\begin{{tabular}}{{lllllll}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
