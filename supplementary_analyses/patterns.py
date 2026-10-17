"""
Patterns of stability on the X12 choice set (main setting): relation of the radius with the
entropy of the center and its divergence from W^lin, with the competition score, the time
limit and the distance to the nearest alternative, with each component of the center, and
dispersion of the candidates by time limit and solver.
Reads the results of wowa_stability_analysis.py; writes to output/csv/supplementary/.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_and_tables.plot_stability_results import (  # noqa: E402
    CRITERIA,
    SOLVER_TABLE_PATH,
)
from project_paths import (  # noqa: E402
    SUPPLEMENTARY_CSV_DIR,
    RAW_DECISION_MATRIX_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
)
from wowa_stability_analysis import (  # noqa: E402
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
)

OUT_DIR = SUPPLEMENTARY_CSV_DIR


def competition_scores() -> pd.Series:
    import re

    rows = re.findall(r"^(T\d\\_D\d+\\_S\d+) &.*& ([\d.]+) \\\\", SOLVER_TABLE_PATH.read_text(), re.M)
    return pd.Series({k.replace(r"\_", "_"): float(v) for k, v in rows})


def alternative_patterns() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    results = results[results["Setting"] == "Main"].copy()
    scores = competition_scores()
    rows = []
    for scenario, p in build_weighting_scenarios().items():
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), p)
        analyzer.build_unified_representation()
        grid = np.array(analyzer.grid, dtype=float)
        w_lin = np.diff(grid)
        x = analyzer.normalized_criteria_matrix.to_numpy(dtype=float)
        dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        distinct = dict(zip(analyzer.alternatives, dist.min(1)))
        for _, r in results[results["Scenario"] == scenario].iterrows():
            row = r[["Scenario", "Profile", "Alternative", "Rho", "Rho_P", "Status"]].to_dict()
            if r["Status"] != "infeasible" and isinstance(r["Center"], str):
                c = np.array([float(v) for v in r["Center"].split()])
                pos = c > 1e-12
                row.update({
                    "Entropy": r["Entropy"],
                    "KL_from_linear": float((c[pos] * np.log(c[pos] / w_lin[pos])).sum()),
                    "Shapley_Entropy": r["Shapley_Entropy"],
                    "Max_Component": c.max(),
                    "Orness": r["Optimized_Orness"],
                })
            team, dur, seed = r["Alternative"].split("_")
            row.update({"Team": team, "Time": int(dur[1:]), "Seed": int(seed[1:]),
                        "Score": scores[r["Alternative"]], "Distinctiveness": distinct[r["Alternative"]]})
            rows.append(row)
    alts = pd.DataFrame(rows)
    alts["Rank"] = alts.groupby(["Scenario", "Profile"])["Score"].rank(method="min").astype(int)

    summary = []
    for (scenario, profile), g in alts.groupby(["Scenario", "Profile"], sort=False):
        pos = g[g["Rho"] > 1e-9]
        row = {"Scenario": scenario, "Profile": profile, "N_positive": len(pos),
               "N_feasible": int((g["Status"] != "infeasible").sum())}
        for var in ("Entropy", "KL_from_linear", "Score", "Time", "Distinctiveness"):
            rho, pval = spearmanr(pos["Rho"], pos[var])
            row[f"Spearman_{var}"], row[f"p_{var}"] = rho, pval
        # Same correlations over all alternatives (radius 0 for never-optimal ones).
        for var in ("Score", "Time", "Distinctiveness"):
            row[f"Spearman_all_{var}"] = spearmanr(g["Rho"], g[var])[0]
        summary.append(row)

    raw = pd.read_csv(RAW_DECISION_MATRIX_CSV_PATH)
    analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH),
                                     build_weighting_scenarios()["Baseline"])
    x = analyzer.normalized_criteria_matrix[CRITERIA].to_numpy(dtype=float)
    times = raw["Alternative"].str.extract(r"_D(\d+)_")[0].astype(int).to_numpy()
    teams = raw["Alternative"].str[:2].to_numpy()
    dispersion = []
    for key, values in (("Time", times), ("Team", teams)):
        for v in sorted(set(values)):
            idx = np.where(values == v)[0]
            d = [np.linalg.norm(x[a] - x[b]) for a, b in combinations(idx, 2)]
            dispersion.append({"Group": key, "Value": v, "N": len(idx),
                               "Mean_Pairwise_Distance": np.mean(d) if d else np.nan})
    return alts, pd.DataFrame(summary), pd.DataFrame(dispersion)


def component_patterns() -> pd.DataFrame:
    """Spearman correlation between the radius and each component of the centre."""
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    results = results[(results["Setting"] == "Main") & (results["Rho"] > 1e-9)]
    rows = []
    for (scenario, profile), g in results.groupby(["Scenario", "Profile"], sort=False):
        c = np.array([[float(v) for v in s.split()] for s in g["Center"]])
        corr = [spearmanr(g["Rho"], c[:, k])[0] for k in range(c.shape[1])]
        rows.append({"Scenario": scenario, "Profile": profile, "K": c.shape[1],
                     "Spearman_First": corr[0], "Spearman_Last": corr[-1],
                     "Spearman_Min_Component": spearmanr(g["Rho"], c.min(1))[0],
                     "Spearman_Orness": spearmanr(g["Rho"], g["Optimized_Orness"])[0],
                     "Components_Above_0.5": int(sum(r > 0.5 for r in corr)),
                     "Components_Below_-0.5": int(sum(r < -0.5 for r in corr))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alts, summary, dispersion = alternative_patterns()
    alts.to_csv(OUT_DIR / "patterns_alternatives.csv", index=False)
    summary.to_csv(OUT_DIR / "patterns_summary.csv", index=False)
    dispersion.to_csv(OUT_DIR / "patterns_time_dispersion.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(), "\n", dispersion.round(3).to_string())
    components = component_patterns()
    components.to_csv(OUT_DIR / "patterns_components.csv", index=False)
    print(components.round(2).to_string())


if __name__ == "__main__":
    main()
