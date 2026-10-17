"""
Sensitivity of the selection to the norm of the stability ball and to the weighting vector p.

1. Selection with the weights of Table 2 under the L2, L1, capacity and slope norms.
2. Refinement: radii on the complete grid and on finer grids with artificial points
   (only the slope norm, max_k |d_k| / (t_k - t_{k-1}), gives the same radii).
3. Perturbations of p: log-normal factors on every weight ("free") or on each group of
   equal weights, which keeps the equal weights and breaks the other ties ("tie_preserving").
   The importance requirements stay around the original p.
4. Neighbours on the 0.05 lattice: 0.05 moved from one criterion to another, with the
   importance requirements around the neighbour.

Writes to output/csv/supplementary/; the perturbations take about 30 minutes on 8 cores.
"""

import sys
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_and_tables.plot_stability_results import CRITERIA  # noqa: E402
from project_paths import RAW_DECISION_MATRIX_CSV_PATH, SUPPLEMENTARY_CSV_DIR  # noqa: E402
from wowa_stability_analysis import (  # noqa: E402
    PROFILES,
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
    shapley_bounds_around_weights,
)

NORMS = ("L2", "L1", "capacity", "slope")
ETAS = (0.001, 0.01, 0.05, 0.15)
N_DRAWS = 100
SCENARIOS = build_weighting_scenarios()
OUT_DIR = SUPPLEMENTARY_CSV_DIR


class RefinedAnalyzer(WOWAStabilityAnalyzer):
    """Complete grid with additional artificial grid points."""

    def __init__(self, path, weights, extra_points):
        self.extra_points = [Fraction(x) for x in extra_points]
        super().__init__(path, weights)

    def build_complete_weight_grid(self):
        super().build_complete_weight_grid()
        grid = sorted(set(self.grid_index) | {x for x in self.extra_points if 0 < x < 1})
        self.grid_index = {value: k for k, value in enumerate(grid)}
        self.grid = [float(value) for value in grid]


def analyzer_for(weights, extra_points=None):
    if extra_points is None:
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), weights)
    else:
        analyzer = RefinedAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), weights, extra_points)
    analyzer.build_unified_representation()
    return analyzer


def top_two(df):
    df = df.sort_values("Rho", ascending=False, ignore_index=True)
    return df.iloc[0], df.iloc[1]


def norm_selection():
    """Radii of every alternative under each norm, with the weights of Table 2."""
    rows = []
    for scenario, p in SCENARIOS.items():
        analyzer = analyzer_for(p)
        bounds = shapley_bounds_around_weights(p)
        for norm in NORMS:
            for profile, o_min, o_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(o_min, o_max, bounds, norm)
                df = df.sort_values("Rho", ascending=False, ignore_index=True)
                df["Rho_P"] = analyzer.reference_radius(o_min, o_max, bounds, norm)
                df["Max_Shapley_Deviation"] = np.abs(
                    df[[f"Shapley_{c}" for c in CRITERIA]].to_numpy() - [p[c] for c in CRITERIA]
                ).max(axis=1)
                df["Scenario"], df["Profile"], df["Norm"] = scenario, profile, norm
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def refinement():
    """Radii on the complete grid, with every cell split in two and with the 0.01 lattice added."""
    rows = []
    for scenario, p in SCENARIOS.items():
        base = analyzer_for(p)
        grid = sorted(base.grid_index)
        halves = [(a + b) / 2 for a, b in zip(grid[:-1], grid[1:])]
        analyzers = {
            "complete": base,
            "halves": analyzer_for(p, halves),
            "lattice_0.01": analyzer_for(p, [Fraction(j, 100) for j in range(1, 100)]),
        }
        bounds = shapley_bounds_around_weights(p)
        for norm in NORMS:
            for profile, o_min, o_max in PROFILES:
                for grid_name, analyzer in analyzers.items():
                    df = analyzer.solve_stability_radius_lp(o_min, o_max, bounds, norm, details=False)
                    rows.append(df[["Alternative", "Rho"]].assign(
                        Scenario=scenario, Profile=profile, Norm=norm, Grid=grid_name,
                        K=len(analyzer.grid) - 1))
    return pd.concat(rows, ignore_index=True)


def lattice_neighbours(p, step=Fraction(1, 20)):
    names = list(p)
    exact = {c: Fraction(str(round(p[c], 10))) for c in names}
    out = []
    for a in names:
        for b in names:
            if a != b and exact[a] - step > 0:
                q = dict(exact)
                q[a], q[b] = q[a] - step, q[b] + step
                out.append((f"{a} -> {b}", {c: float(v) for c, v in q.items()}))
    return out


def perturbed_weights(p, kind, eta, rng):
    names = list(p)
    base = np.array([p[c] for c in names])
    if kind == "free":
        factors = np.exp(eta * rng.standard_normal(len(base)))
    else:
        groups = {v: np.exp(eta * rng.standard_normal()) for v in sorted(set(base))}
        factors = np.array([groups[v] for v in base])
    q = base * factors
    return dict(zip(names, np.round(q / q.sum(), 8)))


def perturbation_tasks():
    rng = np.random.default_rng(0)
    tasks = []
    for scenario, p in SCENARIOS.items():
        bounds = shapley_bounds_around_weights(p)
        tasks.append((scenario, "original", 0, 0.0, p, bounds))
        for design in ("free", "tie_preserving"):
            for eta in ETAS:
                for draw in range(N_DRAWS):
                    tasks.append((scenario, design, draw, eta,
                                  perturbed_weights(p, design, eta, rng), bounds))
        for draw, (label, q) in enumerate(lattice_neighbours(p)):
            tasks.append((scenario, f"lattice: {label}", draw, 0.05, q,
                          shapley_bounds_around_weights(q)))
    return tasks


def solve_draw(task):
    """Winner and radii of every alternative, for every norm and profile, for one p."""
    gp.setParam("Threads", 1)
    scenario, design, draw, eta, weights, bounds = task
    analyzer = analyzer_for(weights)
    rows = []
    for norm in NORMS:
        for profile, o_min, o_max in PROFILES:
            df = analyzer.solve_stability_radius_lp(o_min, o_max, bounds, norm, details=False)
            first, second = top_two(df)
            rows.append({
                "Scenario": scenario, "Design": design, "Draw": draw, "Eta": eta,
                "K": len(analyzer.grid) - 1, "Norm": norm, "Profile": profile,
                "Winner": first["Alternative"], "Rho": first["Rho"],
                "Second": second["Alternative"], "Rho_Second": second["Rho"],
                "Radii": " ".join(f"{r:.6g}" for r in df["Rho"]),
                "Max_Abs_Change_p": max(abs(weights[c] - SCENARIOS[scenario][c]) for c in weights),
            })
    return rows


def summarize(selection, refined, draws):
    """Summary tables: selection per norm, refinement and perturbations of p."""
    rows = []
    for (s, p, norm), g in selection.groupby(["Scenario", "Profile", "Norm"], sort=False):
        first, second = g.iloc[0], g.iloc[1]
        rows.append({
            "Scenario": s, "Profile": p, "Norm": norm, "Winner": first.Alternative,
            "Tied_Winners": " / ".join(sorted(g[g.Rho >= first.Rho - 1e-7].Alternative)),
            "Rho": first.Rho, "Rho_Rel": first.Rho / first.Rho_P, "Second": second.Alternative,
            "Relative_Gap": (first.Rho - second.Rho) / first.Rho, "N_Positive": int((g.Rho > 1e-9).sum()),
            "Share_dominance": first.Share_dominance, "Share_orness": first.Share_orness,
            "Share_shapley": first.Share_shapley, "Share_simplex": first.Share_simplex,
            "Orness": first.Optimized_Orness, "Max_Shapley_Deviation": first.Max_Shapley_Deviation,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "norm_selection_summary.csv", index=False)

    rows = []
    base = refined[refined.Grid == "complete"].set_index(["Scenario", "Profile", "Norm", "Alternative"]).Rho
    for (norm, grid), g in refined[refined.Grid != "complete"].groupby(["Norm", "Grid"], sort=False):
        g = g.set_index(["Scenario", "Profile", "Norm", "Alternative"])
        rho0 = base.reindex(g.index)
        rho_star = rho0.groupby(level=[0, 1, 2]).transform("max")
        same = [gg.Rho.idxmax()[3] == rho0.loc[key].idxmax()
                for key, gg in g.groupby(level=[0, 1, 2])]
        rows.append({"Norm": norm, "Grid": grid, "K": f"{g.K.min()}-{g.K.max()}",
                     "Max_Relative_Change": ((g.Rho - rho0).abs() / rho_star).max(),
                     "Same_Winner": int(sum(same)), "Combinations": len(same)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "norm_refinement_summary.csv", index=False)

    original = draws[draws.Design == "original"].set_index(["Scenario", "Norm", "Profile"])
    keys = list(zip(draws.Scenario, draws.Norm, draws.Profile))
    radii0 = {k: np.array([float(v) for v in r.split()]) for k, r in original.Radii.items()}
    draws = draws.assign(
        Design_Group=draws.Design.where(~draws.Design.str.startswith("lattice"), "lattice"),
        Same=[w == original.Winner[k] for w, k in zip(draws.Winner, keys)],
        Radii_Change=[np.abs(np.array([float(v) for v in r.split()]) - radii0[k]).max() / radii0[k].max()
                      for r, k in zip(draws.Radii, keys)],
    )
    summary = (draws[draws.Design_Group != "original"]
               .groupby(["Design_Group", "Eta", "Norm", "Scenario", "Profile"], sort=False)
               .agg(Share_Same=("Same", "mean"), Median_Radii_Change=("Radii_Change", "median"),
                    Median_K=("K", "median"), Median_Max_Change_p=("Max_Abs_Change_p", "median"),
                    Most_Frequent=("Winner", lambda w: w.value_counts().index[0]),
                    Share_Most_Frequent=("Winner", lambda w: w.value_counts(normalize=True).iloc[0]))
               .reset_index())
    summary.to_csv(OUT_DIR / "weight_sensitivity_summary.csv", index=False)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = norm_selection()
    selection.to_csv(OUT_DIR / "norm_selection.csv", index=False)
    refined = refinement()
    refined.to_csv(OUT_DIR / "norm_refinement.csv", index=False)
    draws_path = OUT_DIR / "weight_sensitivity_draws.csv"
    if not draws_path.exists():
        with Pool(8) as pool:
            rows = [r for chunk in pool.imap_unordered(solve_draw, perturbation_tasks(), chunksize=4)
                    for r in chunk]
        pd.DataFrame(rows).to_csv(draws_path, index=False)
    summary = summarize(selection, refined, pd.read_csv(draws_path))
    pd.set_option("display.width", 200)
    print(pd.read_csv(OUT_DIR / "norm_refinement_summary.csv").to_string())
    print(summary.groupby(["Design_Group", "Eta", "Norm"]).Share_Same.mean().unstack().round(2))


if __name__ == "__main__":
    main()
