"""
Extension of the stability-based selection to the alpha and instance runs.
Normalization and grid are fixed per instance; subsets are solved as choice sets.
"""

import tempfile

import numpy as np
import pandas as pd

from decision_matrix_builder import RAW_CRITERION_COLUMNS
from project_paths import (
    EXTENSION_DECISION_MATRIX_CSV_PATH,
    WOWA_EXTENSION_CSV_PATH,
    ensure_output_tree,
)
from wowa_stability_analysis import (
    PROFILES,
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
    shapley_bounds_around_weights,
)

INSTANCES = ("X03", "X12", "X13")


def load_runs(name):
    path = EXTENSION_DECISION_MATRIX_CSV_PATH.with_name(
        f"decision_matrix_raw_extension_{name}.csv"
    )
    runs = pd.read_csv(path)
    runs["Score_05"] = 0.5 * runs["mean_risk"] + 0.5 * runs["expected_excess"]
    runs["Key"] = runs[RAW_CRITERION_COLUMNS].round(10).astype(str).agg("|".join, axis=1)
    return runs


def main() -> None:
    ensure_output_tree()
    rows, draws, sizes = [], [], []
    rng = np.random.default_rng(0)
    for name in INSTANCES:
        if not EXTENSION_DECISION_MATRIX_CSV_PATH.with_name(
            f"decision_matrix_raw_extension_{name}.csv"
        ).exists():
            print(f"Skipping {name}: no decision matrix")
            continue
        runs = load_runs(name)
        unique = runs.groupby("Key", sort=False).agg(
            Alternative=("Alternative", "first"),
            Alphas=("alpha", lambda a: sorted(set(a))),
            Score_05=("Score_05", "min"),
            **{c: (c, "first") for c in RAW_CRITERION_COLUMNS},
        )
        alternatives = unique["Alternative"].tolist()
        score_rank = unique["Score_05"].rank(method="min").astype(int).to_numpy()
        sets = {"all": list(range(len(unique)))}
        for alpha in sorted(runs["alpha"].unique()):
            sets[f"alpha={alpha}"] = [i for i, a in enumerate(unique["Alphas"]) if alpha in a]
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            unique[["Alternative", *RAW_CRITERION_COLUMNS]].to_csv(tmp.name, index=False)
            for scenario, weights in build_weighting_scenarios().items():
                analyzer = WOWAStabilityAnalyzer(tmp.name, weights)
                bounds = shapley_bounds_around_weights(weights)
                for profile, a_min, a_max in PROFILES:
                    for set_name, members in sets.items():
                        df = analyzer.solve_stability_radius_lp(
                            a_min, a_max, bounds, "L2", members, details=set_name == "all"
                        ).sort_values("Rho", ascending=False)
                        if set_name == "all":
                            sizes.append({
                                "Instance": name, "Scenario": scenario, "Profile": profile,
                                "N": len(members), "K_minimal": analyzer.minimal_grid_size(),
                                "K_complete": len(analyzer.grid) - 1,
                                "Constraints": df["Num_Constraints"].max(),
                                "LP_Seconds": df["Solve_Time"].sum(),
                                "N_Infeasible": int((df["Status"] == "infeasible").sum()),
                                "LP_Seconds_Infeasible": df.loc[
                                    df["Status"] == "infeasible", "Solve_Time"
                                ].sum(),
                            })
                        win = df.iloc[0]
                        i_win = alternatives.index(win["Alternative"])
                        best_score = min(members, key=lambda i: score_rank[i])
                        rows.append({
                            "Instance": name, "Set": set_name, "N": len(members),
                            "Scenario": scenario, "Profile": profile,
                            "Winner": win["Alternative"], "Rho": win["Rho"],
                            "Relative_Gap": (win["Rho"] - df.iloc[1]["Rho"]) / win["Rho"],
                            "Winner_Score_Rank": int((score_rank[members] < score_rank[i_win]).sum() + 1),
                            "Best_Score_Positive": bool(
                                df.set_index("Alternative").loc[alternatives[best_score], "Rho"] > 1e-9
                            ),
                            "N_Positive": int((df["Rho"] > 1e-9).sum()),
                        })
                    if name == "X12" and profile == "Full_Spectrum":
                        size = len(sets["alpha=0.5"])
                        by_alpha = [sets[k] for k in sets if k.startswith("alpha")]
                        for draw in range(100):
                            pick = rng.permutation(np.concatenate(
                                [rng.choice(m, size // 3 + (k < size % 3), replace=False)
                                 for k, m in enumerate(by_alpha)]))
                            rho = analyzer.solve_stability_radius_lp(
                                a_min, a_max, bounds, "L2", pick, details=False
                            )["Rho"].max()
                            draws.append({"Scenario": scenario, "Draw": draw, "Rho_Star": rho})
    pd.DataFrame(rows).to_csv(WOWA_EXTENSION_CSV_PATH, index=False)
    pd.DataFrame(draws).to_csv(WOWA_EXTENSION_CSV_PATH.with_name("wowa_extension_mixed_draws.csv"), index=False)
    pd.DataFrame(sizes).to_csv(WOWA_EXTENSION_CSV_PATH.with_name("wowa_extension_lp_size.csv"), index=False)
    print(f"Saved {WOWA_EXTENSION_CSV_PATH}")


if __name__ == "__main__":
    main()
