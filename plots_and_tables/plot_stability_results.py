"""
Tables and figures for the revised stability-based selection (complete grid LP).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    EXTENSION_DECISION_MATRIX_CSV_PATH,
    MDS_SELECTION_CSV_PATH,
    MDS_STABILITY_CSV_PATH,
    WOWA_EXTENSION_CSV_PATH,
    OUTPUT_PLOTS_AND_TABLES_DIR,
    OUTPUT_TEX_DIR,
    WOWA_CHOICE_SET_SIZE_CSV_PATH,
    WOWA_GRID_SUMMARY_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
    WOWA_STABILITY_SCENARIOS_CSV_PATH,
    ensure_output_tree,
)
from wowa_stability_analysis import (
    PROFILES,
    WOWAStabilityAnalyzer,
    build_default_importance_scenarios,
    margin_coefficient,
)
from project_paths import RAW_DECISION_MATRIX_CSV_PATH

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
    p = build_default_importance_scenarios()
    for scenario, group in winners.groupby("Scenario", sort=False):
        shap.append(
            f"{scenario} & $\\mathbf p$ & & "
            + " & ".join(f"{p[scenario][c]:.2f}" for c in CRITERIA)
            + " \\\\"
        )
        for _, w in group.iterrows():
            prof = PROFILE_NAMES[w["Profile"]]
            sel.append(
                f"{scenario} & {prof} & {tex_id(w['Alternative'])} & {w['Rho']:.4f} & "
                f"{w['Rho_Rel']:.2f} & {tex_id(w['Second'])} & {w['Rho_Second']:.4f} & "
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
                f"{w['Share_shapley']:.2f} & {w['Share_simplex']:.2f} & {shift_txt} \\\\"
            )
            shap.append(
                f" & {prof} & {w['Entropy']:.3f} & "
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
        "Scenario & Profile & Most stable & $\\rho$ & $\\rho/\\rho_{\\mathcal P}$ & "
        "Second & $\\rho_{2nd}$ & $\\Delta_\\rho$ (\\%) & Orness & Rank",
        sel,
        "llllcllccc",
    )
    write_table(
        "stability_fragility_table",
        "Scenario & Profile & Most stable & Rival & $\\theta_{\\mathrm{dom}}$ & "
        "$\\theta_{\\mathrm{orn}}$ & $\\theta_{\\mathrm{Sh}}$ & $\\theta_{\\Delta}$ & "
        "$\\Delta$orness & Largest $\\Delta\\phi_j$ (gain, loss)",
        frag,
        "llllcccccl",
    )
    write_table(
        "stability_shapley_table",
        "Scenario & Profile & $H(\\mathbf W^*)$ & " + " & ".join(SHORT[c] for c in CRITERIA),
        shap,
        "llc" + "c" * len(CRITERIA),
    )


def sensitivity_table(results: pd.DataFrame) -> None:
    rows = []
    for (scenario, profile), g in results.groupby(["Scenario", "Profile"], sort=False):
        main = g[g.Setting == "Main"].set_index("Alternative").Rho
        cells, taus = [], []
        for setting in ("Main", "Free_Shapley", "Shapley_0.05", "Shapley_0.01", "L1", "Capacity"):
            other = g[g.Setting == setting].set_index("Alternative").Rho
            best = other[other >= other.max() - 1e-9].index
            cells.append(" / ".join(tex_id(a) for a in best))
            if setting != "Main":
                taus.append(kendalltau(main, other.reindex(main.index))[0])
        rows.append(
            f"{scenario} & {PROFILE_NAMES[profile]} & " + " & ".join(cells)
            + f" & {min(taus):.3f} \\\\"
        )
    write_table(
        "stability_sensitivity_table",
        "Scenario & Profile & $\\varepsilon=0.1$ & free & $\\varepsilon=0.05$ & "
        "$\\varepsilon=0.01$ & $L_1$ & Capacity & min.\\ $\\tau$",
        rows,
        "llllllllc",
    )


def lp_size_table(grid: pd.DataFrame) -> None:
    rows = [
        f"{r.Scenario} & {r.M} & {r.N} & {r.K_minimal} & {r.K_complete} & "
        f"{int(r.Constraints)} & {1000 * r.Mean_LP_Seconds:.2f} & {1000 * r.Total_LP_Seconds:.1f} \\\\"
        for r in grid.itertuples()
    ]
    write_table(
        "lp_size_table",
        "Scenario & $M$ & $N$ & $K_{\\min}$ & $K$ & Constraints & "
        "LP time (ms) & Total (ms)",
        rows,
        "lccccccc",
    )


def plot_stability_map(results: pd.DataFrame) -> None:
    main = results[results.Setting == "Main"].copy()
    main["Config"] = main.Scenario.str[:4] + ". " + main.Profile.map(PROFILE_NAMES)
    mat = main.pivot_table(index="Alternative", columns="Config", values="Rho_Rel", sort=False)
    ranks = competition_ranks(mat.index)
    mat = mat.loc[(mat > 1e-9).any(axis=1)]
    mat = mat.loc[ranks.reindex(mat.index).sort_values().index]
    mat.index = [f"{a} (#{ranks[a]})" for a in mat.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(mat) + 1.8))
    shown = mat.where(mat > 1e-9)
    im = ax.imshow(shown.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(shown.to_numpy()):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    fig.colorbar(im, ax=ax, label=r"$\rho_i/\rho_{\mathcal{P}}$")
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
    scenarios = build_default_importance_scenarios()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (scenario, g) in zip(axes, winners.groupby("Scenario", sort=False)):
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), scenarios[scenario])
        analyzer.build_unified_importance_grid()
        grid = np.array(analyzer.unified_importance_grid)
        k_segments = len(grid) - 1
        for _, w in g[g.Profile != "Full_Spectrum"].iterrows():
            center = np.array([float(v) for v in w["Center"].split()])
            q = np.concatenate([[0.0], np.cumsum(center)])
            band = w["Rho"] * np.array(
                [margin_coefficient((np.arange(k_segments) < k).astype(float))
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
    rows = []
    for (instance, set_name), g in ext.groupby(["Instance", "Set"], sort=False):
        n = g["N"].iloc[0]
        label = "all $\\alpha$" if set_name == "all" else f"$\\alpha={set_name.split('=')[1]}$"
        rows.append(
            f"{instance} & {label} & {n} & {g.N_Positive.median():.0f} & "
            f"{g.Winner.nunique()} & {g.Winner_Score_Rank.median():.0f} & "
            f"{100 * g.Best_Score_Positive.mean():.0f} & {100 * g.Relative_Gap.median():.0f} \\\\"
        )
    write_table(
        "extension_table",
        "Instance & Candidates & $N$ & Optimal in $\\mathcal P$ & Winners & "
        "Winner rank & Best score optimal (\\%) & $\\Delta_\\rho$ (\\%)",
        rows,
        "llcccccc",
    )


def mds_table(stab: pd.DataFrame, sel: pd.DataFrame) -> None:
    def unchanged(variant):
        x = sel[sel.Variant == variant].set_index(["Scenario", "Profile"])
        ref = x[x.Seed == "reference"].Winner
        seeds = x[x.Seed != "reference"]
        return 100 * (seeds.Winner == ref.reindex(seeds.index)).mean()

    a, c = (stab[stab.Variant == v] for v in ("aligned", "contained"))
    ref_a = sel[(sel.Variant == "aligned") & (sel.Seed == "reference")].set_index(["Scenario", "Profile"]).Winner
    ref_c = sel[(sel.Variant == "contained") & (sel.Seed == "reference")].set_index(["Scenario", "Profile"]).Winner
    abl = sel[sel.Seed == "without_c5_c6"].set_index(["Scenario", "Profile"]).Winner
    both = lambda f, col, mult=1: f"{mult * f(a[col]):.{2 if mult == 1 else 0}f} & {mult * f(c[col]):.{2 if mult == 1 else 0}f}"
    med = lambda x: x.median()
    rows = [
        f"Stress (median) & \\multicolumn{{2}}{{c}}{{{a.Stress.median():.3f} [{a.Stress.min():.3f}, {a.Stress.max():.3f}]}} \\\\",
        f"Runs reaching the iteration limit & \\multicolumn{{2}}{{c}}{{{(a.Iterations >= 500).sum()} of {len(a)}}} \\\\",
        f"Procrustes disparity (0: same shape, 1: unrelated) & \\multicolumn{{2}}{{c}}{{{a.Disparity.median():.3f}}} \\\\",
        f"Displacement, median (km) & {a.Shift_Median_km.median():.0f} & {c.Shift_Median_km.median():.0f} \\\\",
        f"Interventions with changed $\\mu_{{\\mathrm{{ski}}}}$ (\\%) & {both(med, 'Ski_Changed', 100)} \\\\",
        f"Interventions with changed $\\mu_{{\\mathrm{{park}}}}$ (\\%) & {both(med, 'Park_Changed', 100)} \\\\",
        f"Spearman of $c_5$ with the reference & {both(med, 'Spearman_c5')} \\\\",
        f"Spearman of $c_6$ with the reference & {both(med, 'Spearman_c6')} \\\\",
        f"Most stable alternative unchanged (\\% of seeds $\\times$ combinations) & {unchanged('aligned'):.0f} & {unchanged('contained'):.0f} \\\\",
        f"Reference selection changed by containment (\\% of combinations) & \\multicolumn{{2}}{{c}}{{{100 * (ref_a != ref_c.reindex(ref_a.index)).mean():.0f}}} \\\\",
        f"Reference selection unchanged without $c_5,c_6$ (\\% of combinations) & \\multicolumn{{2}}{{c}}{{{100 * (abl == ref_a.reindex(abl.index)).mean():.0f}}} \\\\",
    ]
    write_table("mds_stability_table", "Measure (median over 20 seeds) & Aligned & Aligned and contained", rows, "lcc")


def map_frequency_table(sel: pd.DataFrame) -> None:
    """Selection across the plausible maps (aligned MDS seeds) of each combination."""
    x = sel[sel.Variant == "aligned"]
    ref = x[x.Seed == "reference"].set_index(["Scenario", "Profile"]).Winner
    seeds = x[x.Seed != "reference"]
    rows = []
    for (scenario, profile), g in seeds.groupby(["Scenario", "Profile"], sort=False):
        freq = g.Winner.value_counts(normalize=True)
        reference = ref[(scenario, profile)]
        rows.append(
            f"{scenario} & {PROFILE_NAMES[profile]} & {tex_id(reference)} & "
            f"{100 * freq.get(reference, 0):.0f} & {tex_id(freq.index[0])} & {100 * freq.iloc[0]:.0f} \\\\"
        )
    write_table(
        "map_frequency_table",
        "Scenario & Profile & Reference map & \\% of maps & Most frequent & \\% of maps",
        rows,
        "llllcc",
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
               label="Most stable (some set and configuration)")
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
    sensitivity_table(results)
    lp_size_table(pd.read_csv(WOWA_GRID_SUMMARY_CSV_PATH))
    plot_stability_map(results)
    plot_choice_set_size(pd.read_csv(WOWA_CHOICE_SET_SIZE_CSV_PATH))
    plot_quantifiers(winners)
    extension_table(pd.read_csv(WOWA_EXTENSION_CSV_PATH))
    plot_risk_excess()
    if MDS_SELECTION_CSV_PATH.exists():
        mds_table(pd.read_csv(MDS_STABILITY_CSV_PATH), pd.read_csv(MDS_SELECTION_CSV_PATH))
        map_frequency_table(pd.read_csv(MDS_SELECTION_CSV_PATH))
    print(f"Tables saved to {OUTPUT_TEX_DIR}; figures to {OUTPUT_PLOTS_AND_TABLES_DIR}")


if __name__ == "__main__":
    main()
