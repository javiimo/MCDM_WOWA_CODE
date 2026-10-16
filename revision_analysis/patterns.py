"""
Patterns of stability on the original X12 choice set (main setting) requested by
Reviewer 1 (B1) and structural sensitivity to the weighting vector p (R1.C2, R2.2).
Reads the results of wowa_stability_analysis.py; writes to output/csv/revision/.
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
    OUTPUT_CSV_DIR,
    RAW_DECISION_MATRIX_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
)
from wowa_stability_analysis import (  # noqa: E402
    PROFILES,
    WOWAStabilityAnalyzer,
    build_default_importance_scenarios,
    shapley_bounds_around_weights,
)

OUT_DIR = OUTPUT_CSV_DIR / "revision"


def competition_scores() -> pd.Series:
    import re

    rows = re.findall(r"^(T\d\\_D\d+\\_S\d+) &.*& ([\d.]+) \\\\", SOLVER_TABLE_PATH.read_text(), re.M)
    return pd.Series({k.replace(r"\_", "_"): float(v) for k, v in rows})


def alternative_patterns() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    results = results[results["Setting"] == "Main"].copy()
    scores = competition_scores()
    rows = []
    for scenario, p in build_default_importance_scenarios().items():
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), p)
        analyzer.build_transformed_alternative_matrix()
        grid = np.array(analyzer.unified_importance_grid, dtype=float)
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
                                     build_default_importance_scenarios()["Baseline"])
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


def p_sensitivity(n_draws=100, eps_values=(0.05, 0.15)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = []
    for scenario, p in build_default_importance_scenarios().items():
        bounds = shapley_bounds_around_weights(p)  # importance requirement stays on the original p
        names = list(p)
        base = np.array([p[c] for c in names])
        draws = [("original", 0.0, base)]
        for eps in eps_values:
            for _ in range(n_draws):
                q = base * np.exp(eps * rng.standard_normal(len(base)))
                draws.append((len(draws), eps, np.round(q / q.sum(), 6)))
        for draw, eps, q in draws:
            analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), dict(zip(names, q)))
            analyzer.build_transformed_alternative_matrix()
            k = len(analyzer.unified_importance_grid) - 1
            for profile, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, details=False)
                best = df.sort_values("Rho", ascending=False).iloc[0]
                rows.append({"Scenario": scenario, "Draw": draw, "Eps": eps, "K": k,
                             "Profile": profile, "Winner": best["Alternative"], "Rho": best["Rho"],
                             "Max_Abs_Change_p": float(np.abs(q - base).max())})
        print(scenario, "done")
    draws = pd.DataFrame(rows)
    original = draws[draws["Draw"] == "original"].set_index(["Scenario", "Profile"])["Winner"]
    draws["Original_Winner"] = [original[(s, pr)] for s, pr in zip(draws["Scenario"], draws["Profile"])]
    summary = []
    for (scenario, eps, profile), g in draws[draws["Draw"] != "original"].groupby(
            ["Scenario", "Eps", "Profile"], sort=False):
        top = g["Winner"].value_counts()
        summary.append({"Scenario": scenario, "Eps": eps, "Profile": profile,
                        "Median_K": g["K"].median(), "Median_Max_Change_p": g["Max_Abs_Change_p"].median(),
                        "Original_Winner": g["Original_Winner"].iloc[0],
                        "Share_Original": (g["Winner"] == g["Original_Winner"]).mean(),
                        "Most_Frequent": top.index[0], "Share_Most_Frequent": top.iloc[0] / len(g)})
    return draws, pd.DataFrame(summary)


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
    if (OUT_DIR / "p_sensitivity_summary.csv").exists():
        return  # the sampling below takes about half an hour
    draws, p_summary = p_sensitivity()
    draws.to_csv(OUT_DIR / "p_sensitivity.csv", index=False)
    p_summary.to_csv(OUT_DIR / "p_sensitivity_summary.csv", index=False)
    print(p_summary.round(3).to_string())


if __name__ == "__main__":
    main()
