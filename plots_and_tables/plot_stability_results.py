"""
Tables and figures of the stability-based selection on X12 and on the extended candidate
sets (Section 5): selection, fragility and Shapley importance at the centers, stability map,
representative quantifiers, choice-set size, extension and LP size.
Run after wowa_stability_analysis.py and wowa_extension_analysis.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (  # noqa: E402
    EXTENSION_DECISION_MATRIX_CSV_PATH,
    OUTPUT_PLOTS_AND_TABLES_DIR,
    OUTPUT_TEX_DIR,
    RAW_DECISION_MATRIX_CSV_PATH,
    WOWA_CHOICE_SET_SIZE_CSV_PATH,
    WOWA_EXTENSION_CSV_PATH,
    WOWA_EXTENSION_LP_SIZE_CSV_PATH,
    WOWA_GRID_SUMMARY_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
    WOWA_STABILITY_SCENARIOS_CSV_PATH,
    ensure_output_tree,
)
from wowa_stability_analysis import (  # noqa: E402
    MAIN_NORM,
    PROFILES,
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
    margin_coefficient,
)

SOLVER_TABLE_PATH = OUTPUT_TEX_DIR / "appendix_solver_results_table.tex"
CRITERIA = [
    "Reliability Stress",
    "Max Critical Overlap",
    "Risk Volatility",
    "Resource Slack",
    "Ski Season Risk",
    "Summer Eco-Stress",
    "Social Weekend Risk",
]
SHORT = dict(zip(CRITERIA, [f"$c_{j}$" for j in range(1, 8)]))
PROFILE_NAMES = {p[0]: p[0].replace("_", "-") for p in PROFILES}


def tex_id(name: str) -> str:
    return str(name).replace("_", r"\_")


def competition_ranks(alternatives) -> pd.Series:
    rows = re.findall(
        r"^(T\d\\_D\d+\\_S\d+) &.*& ([\d.]+) \\\\", SOLVER_TABLE_PATH.read_text(), re.M
    )
    scores = pd.Series({k.replace(r"\_", "_"): float(v) for k, v in rows})
    return scores.reindex(alternatives).rank(method="min").astype(int)


def write_table(name: str, header: str, rows: list[str], cols: str) -> None:
    body = "\n".join(rows)
    text = (
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    (OUTPUT_TEX_DIR / f"{name}.tex").write_text(text)


def selection_tables(winners: pd.DataFrame, results: pd.DataFrame) -> None:
    ranks = competition_ranks(results["Alternative"].unique())
    sel, frag, shap = [], [], []
    p = build_weighting_scenarios()
    for scenario, group in winners.groupby("Scenario", sort=False):
        shap.append(
            f"{scenario} & $\\mathbf p$ & "
            + " & ".join(f"{p[scenario][c]:.2f}" for c in CRITERIA)
            + " \\\\"
        )
        for _, w in group.iterrows():
            prof = PROFILE_NAMES[w["Profile"]]
            sel.append(
                f"{scenario} & {prof} & {tex_id(w['Alternative'])} & {w['Rho']:.4f} & "
                f"{tex_id(w['Second'])} & {w['Rho_Second']:.4f} & "
                f"{100 * w['Relative_Gap']:.1f} & {w['Optimized_Orness']:.3f} & "
                f"{ranks[w['Alternative']]} \\\\"
            )
            shifts = pd.Series({c: w.get(f"Tie_Shift_{c}", np.nan) for c in CRITERIA})
            if shifts.notna().any():
                gain, lose = shifts.idxmax(), shifts.idxmin()
                shift_txt = (
                    f"{w['Tie_Orness_Shift']:+.3f} & {SHORT[gain]} ({shifts[gain]:+.3f}), "
                    f"{SHORT[lose]} ({shifts[lose]:+.3f})"
                )
            else:
                shift_txt = "--- & ---"
            frag.append(
                f"{scenario} & {prof} & {tex_id(w['Alternative'])} & "
                f"{tex_id(w['Rival']) if isinstance(w['Rival'], str) else '---'} & "
                f"{w['Share_dominance']:.2f} & {w['Share_orness']:.2f} & "
                f"{w['Share_simplex']:.2f} & {shift_txt} \\\\"
            )
            shap.append(
                f" & {prof} & "
                + " & ".join(f"{w[f'Shapley_{c}']:.3f}" for c in CRITERIA)
                + " \\\\"
            )
        sel.append("\\midrule")
        frag.append("\\midrule")
        shap.append("\\midrule")
    for rows in (sel, frag, shap):
        rows.pop()
    write_table(
        "stability_selection_table",
        "Scenario & Profile & Most robust & $\\rho$ & "
        "Second & $\\rho_{2nd}$ & $\\Delta_\\rho$ (\\%) & Center orness & Rank",
        sel,
        "llllllccc",
    )
    write_table(
        "stability_fragility_table",
        "Scenario & Profile & Most robust & Competitor & $\\theta_{\\mathrm{dom}}$ & "
        "$\\theta_{\\mathrm{orn}}$ & $\\theta_{\\Delta}$ & "
        "$\\Delta$orness & Largest $\\Delta\\phi_j$ (gain, loss)",
        frag,
        "llllccccl",
    )
    write_table(
        "stability_shapley_table",
        "Scenario & Profile & " + " & ".join(SHORT[c] for c in CRITERIA),
        shap,
        "ll" + "c" * len(CRITERIA),
    )


def competition_comparison_table(results: pd.DataFrame) -> None:
    """Competition quality and positive-radius counts, from the generated run table and LP CSV."""
    rows = re.findall(r"^(T\d\\_D\d+\\_S\d+) &.*& ([\d.]+) \\\\", SOLVER_TABLE_PATH.read_text(), re.M)
    scores = pd.Series({k.replace(r"\_", "_"): float(v) for k, v in rows})
    main = results[results.Setting == "Main"]
    scores = scores.reindex(main.Alternative.unique()).sort_values()
    ranks = scores.rank(method="min").astype(int)
    winners = main.loc[main.groupby(["Scenario", "Profile"]).Rho.idxmax(), "Alternative"]
    names = list(dict.fromkeys(list(scores.head(3).index) + list(winners)))
    lines = []
    for name in names:
        g = main[main.Alternative == name]
        lines.append(f"{tex_id(name)} & {ranks[name]} & "
                     f"{100*(scores[name]/scores.min()-1):.2f} & {int((winners == name).sum())} & "
                     f"{max(g.Rho.max(), 0.0):.4f} " + r"\\")
    write_table("competition_comparison_table",
                r"Alternative & Rank & Score gap (\%) & Selected (of 8) & Max.\ $\rho$",
                lines, "lcccc")


def lp_size_table(grid: pd.DataFrame, extension: pd.DataFrame | None = None) -> None:
    """Grid size and LP times: X12 choice set of the paper and extended candidate sets."""
    rows = [
        f"X12 & {r.Scenario} & {r.N} & {r.K_complete} & "
        f"{int(r.Constraints)} & {1000 * r.Mean_LP_Seconds:.3f} & {1000 * r.Total_LP_Seconds:.1f} \\\\"
        for r in grid.itertuples()
    ]
    if extension is not None:
        rows.append("\\midrule")
        for (instance, scenario), g in extension.groupby(["Instance", "Scenario"], sort=False):
            n_lps = g.N.sum()
            rows.append(
                f"{instance} (extended) & {scenario} & {g.N.iloc[0]} & "
                f"{g.K_complete.iloc[0]} & {int(g.Constraints.max())} & "
                f"{1000 * g.LP_Seconds.sum() / n_lps:.3f} & {1000 * g.LP_Seconds.sum():.1f} \\\\"
            )
    write_table(
        "lp_size_table",
        "Instance & Scenario & $N$ & $K$ & Constraints & "
        "LP time (ms) & Total (ms)",
        rows,
        "llccccc",
    )


def plot_stability_map(results: pd.DataFrame) -> None:
    main = results[results.Setting == "Main"].copy()
    main["Config"] = main.Scenario.str[:4] + ". " + main.Profile.map(PROFILE_NAMES)
    mat = main.pivot_table(index="Alternative", columns="Config", values="Rho", sort=False)
    ranks = competition_ranks(mat.index)
    mat = mat.loc[(mat > 1e-9).any(axis=1)]
    mat = mat.loc[ranks.reindex(mat.index).sort_values().index]
    mat.index = [f"{a} (#{ranks[a]})" for a in mat.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(mat) + 1.8))
    shown = mat.where(mat > 1e-9)
    im = ax.imshow(shown.to_numpy(), cmap="Blues", vmin=0, vmax=None, aspect="auto")
    for (r, c), v in np.ndenumerate(shown.to_numpy()):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    fig.colorbar(im, ax=ax, label=r"$\rho_i$")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_AND_TABLES_DIR / "stability_map.png", dpi=300)
    plt.close(fig)


def plot_choice_set_size(size: pd.DataFrame) -> None:
    best = size.groupby(["Scenario", "Profile", "Permutation", "N"]).Rho.max().reset_index()
    full_n = best.N.max()
    full = best[best.N == full_n]
    best = pd.concat(
        [best[best.N < full_n]]
        + [full.assign(Permutation=k) for k in best.Permutation.unique()]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (scenario, g) in zip(axes, best.groupby("Scenario", sort=False)):
        for profile, gp in g.groupby("Profile", sort=False):
            q = gp.groupby("N").Rho.quantile([0.25, 0.5, 0.75]).unstack()
            style = "--" if profile == "Full_Spectrum" else "-"
            ax.plot(q.index, q[0.5], style, marker="o", label=PROFILE_NAMES[profile])
            ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.2)
        ax.set_title(scenario)
        ax.set_xlabel("Number of alternatives $N$")
    axes[0].set_ylabel(r"$\rho^*$")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_AND_TABLES_DIR / "choice_set_size.png", dpi=300)
    plt.close(fig)


def plot_quantifiers(winners: pd.DataFrame) -> None:
    """
    Representative quantifier Q_{W*} of each winner and the envelope of the quantifiers in
    its ball: Q(t_k) changes by at most rho * kappa(1_{h <= k}).
    """
    scenarios = build_weighting_scenarios()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (scenario, g) in zip(axes, winners.groupby("Scenario", sort=False)):
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), scenarios[scenario])
        analyzer.build_complete_weight_grid()
        grid = np.array(analyzer.grid)
        k_segments = len(grid) - 1
        for _, w in g[g.Profile != "Full_Spectrum"].iterrows():
            center = np.array([float(v) for v in w["Center"].split()])
            q = np.concatenate([[0.0], np.cumsum(center)])
            band = w["Rho"] * np.array(
                [margin_coefficient((np.arange(k_segments) < k).astype(float), MAIN_NORM,
                                    analyzer.cell_widths)
                 for k in range(k_segments + 1)]
            )
            line, = ax.plot(grid, q, marker=".",
                            label=f"{PROFILE_NAMES[w['Profile']]}: {w['Alternative']}")
            ax.fill_between(grid, q - band, q + band, color=line.get_color(), alpha=0.15)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="$Q(t)=t$")
        ax.set_title(scenario)
        ax.set_xlabel("$t$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$Q_{\mathbf{W}^*}(t)$")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_AND_TABLES_DIR / "stability_quantifiers.png", dpi=300)
    plt.close(fig)


def extension_table(ext: pd.DataFrame) -> None:
    """Per instance and candidate set: distinct winners over the eight configurations,
    configurations won by the best competition schedule (alpha = 0.5 objective) and
    Full-Spectrum radii; the mixed-alpha row gives median radii of matched-size draws."""
    def rho_star(g: pd.DataFrame) -> str:
        fs = g[g.Profile == "Full_Spectrum"].set_index("Scenario").Rho
        return f"{fs['Baseline']:.4f} / {fs['Sustainability']:.4f}"

    rows = []
    for (instance, set_name), g in ext.groupby(["Instance", "Set"], sort=False):
        n = g["N"].iloc[0]
        label = "all $\\alpha$" if set_name == "all" else f"$\\alpha={set_name.split('=')[1]}$"
        rows.append(
            f"{instance} & {label} & {n} & {g.Winner.nunique()} & "
            f"{int((g.Winner_Score_Rank == 1).sum())} & {rho_star(g)} \\\\"
        )
        if instance == "X12" and set_name == "alpha=0.75":
            draws = pd.read_csv(WOWA_EXTENSION_CSV_PATH.with_name("wowa_extension_mixed_draws.csv"))
            med = draws.groupby("Scenario").Rho_Star.median()
            rows.append(
                f"X12 & mixed $\\alpha$ & "
                f"{ext[(ext.Instance == 'X12') & (ext.Set == 'alpha=0.5')].N.iloc[0]} & --- & --- & "
                f"{med['Baseline']:.4f} / {med['Sustainability']:.4f} \\\\"
            )
    write_table(
        "extension_table",
        r"Instance & Candidates & $N$ & Winners & \shortstack{Best score\\selected (of 8)} & "
        r"$\rho^*$ Full-Spectrum (B / S)",
        rows,
        "llcccc",
    )


def plot_risk_excess() -> None:
    runs = pd.read_csv(EXTENSION_DECISION_MATRIX_CSV_PATH.with_name(
        "decision_matrix_raw_extension_X12.csv"))
    ext = pd.read_csv(WOWA_EXTENSION_CSV_PATH)
    winners = set(ext[(ext.Instance == "X12")].Winner)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"team1": "o", "team2": "s", "team3": "^"}
    for (alpha, solver), g in runs.groupby(["alpha", "solver"]):
        ax.scatter(g.mean_risk, g.expected_excess, marker=markers[solver], s=25,
                   color=plt.cm.viridis((alpha - 0.25) / 0.6),
                   label=f"$\\alpha={alpha}$, {solver.replace('team', 'T')}")
    w = runs[runs.Alternative.isin(winners)]
    ax.scatter(w.mean_risk, w.expected_excess, s=160, facecolors="none", edgecolors="red",
               label="Most robust (some set and configuration)")
    ax.set_xlabel(r"Mean risk $\overline{R}$")
    ax.set_ylabel(r"Expected excess $\overline{E}$")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_AND_TABLES_DIR / "risk_excess_x12.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_tree()
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    winners = pd.read_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH)
    selection_tables(winners, results)
    competition_comparison_table(results)
    lp_size_table(
        pd.read_csv(WOWA_GRID_SUMMARY_CSV_PATH),
        pd.read_csv(WOWA_EXTENSION_LP_SIZE_CSV_PATH) if WOWA_EXTENSION_LP_SIZE_CSV_PATH.exists() else None,
    )
    plot_stability_map(results)
    plot_choice_set_size(pd.read_csv(WOWA_CHOICE_SET_SIZE_CSV_PATH))
    plot_quantifiers(winners)
    extension_table(pd.read_csv(WOWA_EXTENSION_CSV_PATH))
    plot_risk_excess()
    print(f"Tables saved to {OUTPUT_TEX_DIR}; figures to {OUTPUT_PLOTS_AND_TABLES_DIR}")


if __name__ == "__main__":
    main()
