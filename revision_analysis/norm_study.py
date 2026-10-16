"""
Choice of the norm of the stability ball: L2, L1, capacity and slope.

1. Selection with the weights of Table 2 under each norm.
2. Refinement: radii on the complete grid and on finer grids with artificial points.
3. Continuity in p: log-normal perturbations of every weight ("free") or of each group
   of equal weights ("tie-preserving"), with the Shapley bounds kept around the original p.
4. Neighbours on the 0.05 lattice: 0.05 moved from one criterion to another, with the
   Shapley bounds around the neighbour.

Writes to output/csv/revision/norms/ and output/tex/revision/norms/.
"""

import sys
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_and_tables.plot_stability_results import CRITERIA, PROFILE_NAMES, tex_id  # noqa: E402
from project_paths import OUTPUT_CSV_DIR, OUTPUT_TEX_DIR, RAW_DECISION_MATRIX_CSV_PATH  # noqa: E402
from wowa_stability_analysis import (  # noqa: E402
    PROFILES,
    WOWAStabilityAnalyzer,
    build_default_importance_scenarios,
    shapley_bounds_around_weights,
)

NORMS = ("L2", "L1", "capacity", "slope")
NORM_NAMES = {"L2": "$L_2$", "L1": "$L_1$", "capacity": "Capacity", "slope": "Slope"}
ETAS = (0.001, 0.01, 0.05, 0.15)
N_DRAWS = 100
OUT_DIR = OUTPUT_CSV_DIR / "revision" / "norms"
TEX_DIR = OUTPUT_TEX_DIR / "revision" / "norms"
SCENARIOS = build_default_importance_scenarios()
COMBOS = [(s, p[0]) for s in SCENARIOS for p in PROFILES]


class RefinedAnalyzer(WOWAStabilityAnalyzer):
    """Complete grid with additional artificial grid points."""

    def __init__(self, path, weights, extra_points):
        self.extra_points = [Fraction(x) for x in extra_points]
        super().__init__(path, weights)

    def build_complete_weight_grid(self):
        super().build_complete_weight_grid()
        grid = sorted(set(self.grid_index) | {x for x in self.extra_points if 0 < x < 1})
        self.grid_index = {value: k for k, value in enumerate(grid)}
        self.unified_importance_grid = [float(value) for value in grid]


def analyzer_for(weights, extra_points=None):
    if extra_points is None:
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), weights)
    else:
        analyzer = RefinedAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), weights, extra_points)
    analyzer.build_transformed_alternative_matrix()
    return analyzer


def top_two(df):
    df = df.sort_values("Rho", ascending=False, ignore_index=True)
    return df.iloc[0], df.iloc[1]


def original_selection():
    rows = []
    for scenario, p in SCENARIOS.items():
        analyzer = analyzer_for(p)
        bounds = shapley_bounds_around_weights(p)
        for norm in NORMS:
            for profile, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, norm)
                rho_ref = analyzer.reference_radius(a_min, a_max, bounds, norm)
                df = df.sort_values("Rho", ascending=False, ignore_index=True)
                df["Rank"] = np.arange(1, len(df) + 1)
                df["Rho_P"], df["Rho_Rel"] = rho_ref, df["Rho"] / rho_ref
                df["Max_Shapley_Deviation"] = np.abs(
                    df[[f"Shapley_{c}" for c in CRITERIA]].to_numpy() - [p[c] for c in CRITERIA]
                ).max(axis=1)
                df["Scenario"], df["Profile"], df["Norm"] = scenario, profile, norm
                rows.append(df)
        print("original selection:", scenario)
    return pd.concat(rows, ignore_index=True)


def refinement():
    """Radii on the complete grid and on two refinements of it."""
    rows = []
    for scenario, p in SCENARIOS.items():
        base = analyzer_for(p)
        grid = sorted(base.grid_index)
        halves = [(a + b) / 2 for a, b in zip(grid[:-1], grid[1:])]
        lattice = [Fraction(j, 100) for j in range(1, 100)]
        analyzers = {"complete": base, "halves": analyzer_for(p, halves),
                     "lattice_0.01": analyzer_for(p, lattice)}
        bounds = shapley_bounds_around_weights(p)
        for norm in NORMS:
            for profile, a_min, a_max in PROFILES:
                for grid_name, analyzer in analyzers.items():
                    df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, norm, details=False)
                    rows.append(df[["Alternative", "Rho"]].assign(
                        Scenario=scenario, Profile=profile, Norm=norm, Grid=grid_name,
                        K=len(analyzer.unified_importance_grid) - 1))
        print("refinement:", scenario)
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


def solve_draw(task):
    """All norms and profiles for one weighting vector."""
    gp.setParam("Threads", 1)
    scenario, design, draw, eta, weights, bounds = task
    analyzer = analyzer_for(weights)
    k = len(analyzer.unified_importance_grid) - 1
    rows = []
    for norm in NORMS:
        for profile, a_min, a_max in PROFILES:
            df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, norm, details=False)
            first, second = top_two(df)
            rows.append({
                "Scenario": scenario, "Design": design, "Draw": draw, "Eta": eta, "K": k,
                "Norm": norm, "Profile": profile, "Winner": first["Alternative"],
                "Rho": first["Rho"], "Second": second["Alternative"], "Rho_Second": second["Rho"],
                "Radii": " ".join(f"{r:.6g}" for r in df["Rho"]),
                "Max_Abs_Change_p": max(abs(weights[c] - SCENARIOS[scenario][c]) for c in weights),
            })
    return rows


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


def write_table(name, header, rows, cols):
    body = "\n".join(rows)
    (TEX_DIR / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def summarize(selection, refined, draws):
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Selection with the original weights.
    rows, summary = [], []
    for s, p in COMBOS:
        cells = []
        for norm in NORMS:
            g = selection[(selection.Scenario == s) & (selection.Profile == p) & (selection.Norm == norm)]
            first, second = g.iloc[0], g.iloc[1]
            gap = (first.Rho - second.Rho) / first.Rho
            summary.append({
                "Scenario": s, "Profile": p, "Norm": norm, "Winner": first.Alternative,
                "Rho": first.Rho, "Rho_Rel": first.Rho_Rel, "Second": second.Alternative,
                "Relative_Gap": gap, "N_Positive": int((g.Rho > 1e-9).sum()),
                "Share_dominance": first.Share_dominance, "Share_orness": first.Share_orness,
                "Share_shapley": first.Share_shapley, "Share_simplex": first.Share_simplex,
                "Orness": first.Optimized_Orness, "Max_Shapley_Deviation": first.Max_Shapley_Deviation,
            })
            cells.append(f"{tex_id(first.Alternative)} ({100 * gap:.0f}; {first.Share_dominance:.2f})")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("norm_selection_table",
                "Scenario & Profile & " + " & ".join(NORM_NAMES[n] for n in NORMS), rows, "llllll")
    pd.DataFrame(summary).to_csv(OUT_DIR / "norm_selection_summary.csv", index=False)

    # 2. Refinement: largest relative change of a radius and change of the winner.
    ref_rows = []
    base = refined[refined.Grid == "complete"].set_index(["Scenario", "Profile", "Norm", "Alternative"]).Rho
    for (norm, grid), g in refined[refined.Grid != "complete"].groupby(["Norm", "Grid"], sort=False):
        g = g.set_index(["Scenario", "Profile", "Norm", "Alternative"])
        rho0 = base.reindex(g.index)
        rel = (g.Rho - rho0).abs() / rho0.groupby(level=[0, 1, 2]).transform("max")
        same = []
        for key, gg in g.groupby(level=[0, 1, 2]):
            same.append(gg.Rho.idxmax()[3] == rho0.loc[key].idxmax())
        ref_rows.append({"Norm": norm, "Grid": grid, "K": f"{g.K.min()}-{g.K.max()}",
                         "Max_Relative_Change": rel.max(), "Same_Winner": int(sum(same)),
                         "Combinations": len(same)})
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(OUT_DIR / "norm_refinement_summary.csv", index=False)
    write_table("norm_refinement_table",
                r"Norm & Grid & $K$ & Largest change of $\rho_i$ (\% of $\rho^*$) & Same selection",
                [f"{NORM_NAMES[r.Norm]} & {r.Grid.replace('_', ' ')} & {r.K} & {100 * r.Max_Relative_Change:.1f} & "
                 f"{r.Same_Winner} of {r.Combinations} \\\\" for r in ref.itertuples()], "llccc")

    # 3. Continuity and lattice neighbours: share of draws that keep the original winner.
    original = draws[draws.Design == "original"].set_index(["Scenario", "Norm", "Profile"]).Winner
    draws = draws.assign(Original_Winner=[original[k] for k in zip(draws.Scenario, draws.Norm, draws.Profile)])
    draws["Same"] = draws.Winner == draws.Original_Winner
    draws["Design_Group"] = draws.Design.where(~draws.Design.str.startswith("lattice"), "lattice")
    cont = (draws[draws.Design_Group != "original"]
            .groupby(["Design_Group", "Eta", "Norm", "Scenario", "Profile"], sort=False)
            .agg(Share_Same=("Same", "mean"), Median_K=("K", "median"),
                 Median_Max_Change_p=("Max_Abs_Change_p", "median"),
                 Most_Frequent=("Winner", lambda w: w.value_counts().index[0]),
                 Share_Most_Frequent=("Winner", lambda w: w.value_counts(normalize=True).iloc[0]))
            .reset_index())
    cont.to_csv(OUT_DIR / "norm_continuity_summary.csv", index=False)
    rows = []
    for design, eta in [("free", e) for e in ETAS] + [("tie_preserving", e) for e in ETAS] + [("lattice", 0.05)]:
        g = cont[(cont.Design_Group == design) & (cont.Eta == eta)]
        label = "0.05 lattice" if design == "lattice" else f"{design.replace('_', '-')}, $\\eta={eta}$"
        k = g.Median_K.median()
        dp = g.Median_Max_Change_p.median()
        cells = [f"{100 * g[g.Norm == n].Share_Same.mean():.0f}" for n in NORMS]
        rows.append(f"{label} & {k:.0f} & {dp:.4f} & " + " & ".join(cells) + " \\\\")
    write_table("norm_continuity_table",
                r"Perturbation of $\mathbf p$ & $K$ & $\max_j|p'_j-p_j|$ & "
                + " & ".join(NORM_NAMES[n] for n in NORMS), rows, "lcc" + "c" * len(NORMS))
    rows = []
    for s, p in COMBOS:
        cells = []
        for design, eta in (("free", 0.01), ("tie_preserving", 0.01), ("lattice", 0.05)):
            for n in ("L2", "slope"):
                r = cont[(cont.Design_Group == design) & (cont.Eta == eta) & (cont.Norm == n)
                         & (cont.Scenario == s) & (cont.Profile == p)].iloc[0]
                cells.append(f"{100 * r.Share_Same:.0f}")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("norm_continuity_detail_table",
                r"Scenario & Profile & free $L_2$ & free slope & tie-pres. $L_2$ & tie-pres. slope & lattice $L_2$ & lattice slope",
                rows, "llcccccc")
    return draws


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = original_selection()
    selection.to_csv(OUT_DIR / "norm_selection.csv", index=False)
    refined = refinement()
    refined.to_csv(OUT_DIR / "norm_refinement.csv", index=False)
    draws_path = OUT_DIR / "norm_draws.csv"
    if not draws_path.exists():
        tasks = perturbation_tasks()
        print(f"{len(tasks)} weighting vectors")
        with Pool(8) as pool:
            rows = [r for chunk in pool.imap_unordered(solve_draw, tasks, chunksize=4) for r in chunk]
        pd.DataFrame(rows).to_csv(draws_path, index=False)
    draws = summarize(selection, refined, pd.read_csv(draws_path))
    print(pd.read_csv(OUT_DIR / "norm_refinement_summary.csv").to_string())
    print(draws.groupby(["Design_Group", "Eta", "Norm"]).Same.mean().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
