"""
Tables and figures of the sensitivity and supplementary analyses (Section 5 and Appendix B):
Shapley tolerances, importance of the criteria, patterns of stability, uncertainty of the
spatial reconstruction, and sensitivity to the weighting vector and to the norm.
Run after wowa_stability_analysis.py and the scripts in supplementary_analyses/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_and_tables.plot_stability_results import (  # noqa: E402
    CRITERIA,
    PROFILE_NAMES,
    tex_id,
    write_table,
)
from project_paths import (  # noqa: E402
    MDS_STABILITY_CSV_PATH,
    OUTPUT_PLOTS_AND_TABLES_DIR,
    OUTPUT_TEX_DIR,
    SUPPLEMENTARY_CSV_DIR,
    WOWA_SHAPLEY_DEVIATION_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
    ensure_output_tree,
)
from wowa_stability_analysis import PROFILES, build_weighting_scenarios  # noqa: E402

IN_DIR = SUPPLEMENTARY_CSV_DIR
SCENARIOS = ["Baseline", "Sustainability"]
COMBOS = [(s, p[0]) for s in SCENARIOS for p in PROFILES]


def winners_of(df):
    """Winner per scenario/profile (ties listed with ' / ')."""
    out = {}
    for (s, p), g in df.groupby(["Scenario", "Profile"]):
        best = g["Rho"].max()
        out[(s, p)] = " / ".join(sorted(g.loc[g["Rho"] >= best - 1e-9, "Alternative"]))
    return out


def sensitivity_table(results):
    settings = [("Main", r"$\varepsilon=0.1$"), ("Free_Shapley", "free"),
                ("Shapley_0.05", r"$\varepsilon=0.05$"), ("Shapley_0.01", r"$\varepsilon=0.01$")]
    win = {s: winners_of(results[results["Setting"] == s]) for s, _ in settings}
    rows = []
    for s, p in COMBOS:
        ref = results[(results["Setting"] == "Main") & (results["Scenario"] == s) & (results["Profile"] == p)]
        taus = []
        for setting in ("Free_Shapley", "Shapley_0.05"):
            other = results[(results["Setting"] == setting) & (results["Scenario"] == s) & (results["Profile"] == p)]
            m = ref.merge(other, on="Alternative")
            taus.append(kendalltau(m["Rho_x"], m["Rho_y"])[0])
        rho_p = {st: results[(results["Setting"] == st) & (results["Scenario"] == s) & (results["Profile"] == p)]["Rho_P"].iloc[0]
                 for st, _ in settings}
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(tex_id(win[st][(s, p)]) for st, _ in settings)
                    + f" & {min(taus):.3f} & {rho_p['Main']:.4f} & {rho_p['Shapley_0.01']:.4f} \\\\")
    write_table("stability_sensitivity_table",
                "Scenario & Profile & " + " & ".join(n for _, n in settings)
                + r" & min.\ $\tau$ & $\rho_{\mathcal P}$ ($0.1$) & $\rho_{\mathcal P}$ ($0.01$)", rows, "llllllccc")


def shapley_deviation_table(results):
    """
    delta* (smallest max_j |phi_j - p_j| in the profile) and, for each Shapley tolerance, the
    most stable alternative with max_j |phi_j - p_j| at the center of its ball.
    """
    delta_star = pd.read_csv(WOWA_SHAPLEY_DEVIATION_CSV_PATH).set_index(["Scenario", "Profile"]).Delta_Star
    weights = build_weighting_scenarios()
    settings = [("Free_Shapley", "free"), ("Main", r"$\varepsilon=0.1$"),
                ("Shapley_0.05", r"$\varepsilon=0.05$"), ("Shapley_0.01", r"$\varepsilon=0.01$")]
    rows = []
    for s, p in COMBOS:
        target = np.array([weights[s][c] for c in CRITERIA])
        cells = [f"{delta_star[(s, p)]:.3f}"]
        for setting, _ in settings:
            g = results[(results.Scenario == s) & (results.Profile == p) & (results.Setting == setting)]
            best = g[g.Rho >= g.Rho.max() - 1e-9]
            deviation = np.abs(best[[f"Shapley_{c}" for c in CRITERIA]].to_numpy() - target).max()
            cells.append(f"{tex_id(' / '.join(sorted(best.Alternative)))} ({deviation:.3f})")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("shapley_deviation_table",
                r"Scenario & Profile & $\delta^*$ & " + " & ".join(n for _, n in settings),
                rows, "llclllll")


def patterns_table():
    alts = pd.read_csv(IN_DIR / "patterns_alternatives.csv")
    rows = []
    for s, p in COMBOS:
        g = alts[(alts["Scenario"] == s) & (alts["Profile"] == p)]
        pos = g[g["Rho"] > 1e-9]
        cells = [str(len(pos)), f"{spearmanr(pos['Rho'], pos['Entropy'])[0]:.2f}",
                 f"{spearmanr(pos['Rho'], pos['KL_from_linear'])[0]:.2f}"]
        for var in ("Score", "Time", "Distinctiveness"):
            r, pv = spearmanr(g["Rho"], g[var])
            cells.append(f"{r:.2f}{'$^*$' if pv < 0.05 else ''}")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("patterns_table",
                r"Scenario & Profile & $N_{>0}$ & $H(\mathbf W^*)$ & $D(\mathbf W^*\Vert\mathbf W^{\mathrm{lin}})$ & Score $f$ & Time limit & Distance to nearest",
                rows, "llcccccc")


def map_tables():
    radii = pd.read_csv(IN_DIR / "mds_radii.csv", dtype={"Map": str})
    corr = pd.read_csv(IN_DIR / "mds_pairwise_spearman.csv", dtype={"Map_A": str, "Map_B": str})
    # Per-alternative frequency and radius across the 21 aligned maps.
    radii["Rho"] = radii["Rho"].clip(lower=0.0) + 0.0  # infeasible LPs and solver round-off
    aligned = radii[radii["Variant"] == "aligned"]
    freq = aligned.groupby(["Scenario", "Profile", "Alternative"]).agg(
        Wins=("Winner", "mean"), Mean=("Rho", "mean"), Min=("Rho", "min"), Max=("Rho", "max")).reset_index()
    freq.to_csv(IN_DIR / "mds_alternative_frequency.csv", index=False)
    top = freq[freq["Wins"] > 0].copy()
    ref = winners_of(aligned[aligned["Map"] == "reference"])
    rows = []
    for s, p in COMBOS:
        g = top[(top["Scenario"] == s) & (top["Profile"] == p)].sort_values("Wins", ascending=False)
        dagger = "$^\\dagger$"
        cells = [f"{tex_id(r.Alternative)}{dagger if r.Alternative in ref[(s, p)] else ''} ({100 * r.Wins:.0f}; "
                 f"{r.Mean:.3f} [{r.Min:.3f}, {r.Max:.3f}])" for r in g.head(3).itertuples()]
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + "; ".join(cells) + " \\\\")
    write_table("map_frequency_table", r"Scenario & Profile & Most stable alternative (\% of maps; mean $\rho$ [range])",
                rows, "llp{11cm}")

    # Orientation and ablation summaries.
    orient = radii[radii["Variant"] == "orientation"]
    win_or = {m: winners_of(g) for m, g in orient.groupby("Map")}
    ablation = winners_of(radii[radii["Variant"] == "without_c5_c6"])
    contained = radii[radii["Variant"] == "contained"]
    win_ct = {m: winners_of(g) for m, g in contained.groupby("Map")}
    rows = []
    for s, p in COMBOS:
        r0 = ref[(s, p)]
        or_counts = pd.Series([w[(s, p)] for w in win_or.values()]).value_counts()
        ct_counts = pd.Series([w[(s, p)] for w in win_ct.values()]).value_counts()
        rows.append(f"{s} & {PROFILE_NAMES[p]} & {tex_id(r0)} & "
                    f"{100 * np.mean([w[(s, p)] == r0 for w in win_or.values()]):.0f} & {tex_id(or_counts.index[0])} ({100 * or_counts.iloc[0] / or_counts.sum():.0f}) & "
                    f"{tex_id(ct_counts.index[0])} ({100 * ct_counts.iloc[0] / ct_counts.sum():.0f}) & {tex_id(ablation[(s, p)])} \\\\")
    write_table("map_variants_table",
                r"Scenario & Profile & Reference map & Same, orientations (\%) & Most frequent, orientations (\%) & Most frequent, contained maps (\%) & Without $c_5,c_6$",
                rows, "llllll l")

    summary = {}
    for variant in ("aligned", "contained", "orientation"):
        c = corr[corr["Variant"] == variant]
        with_ref = c[(c["Map_A"] == "reference") | (c["Map_B"] == "reference")]
        summary[variant] = {
            "pairs_c5": c["Spearman_Ski Season Risk"].describe(),
            "pairs_c6": c["Spearman_Summer Eco-Stress"].describe(),
            "ref_c5": with_ref["Spearman_Ski Season Risk"].describe(),
            "ref_c6": with_ref["Spearman_Summer Eco-Stress"].describe(),
        }
    out = pd.concat({(v, k): d for v, s in summary.items() for k, d in s.items()}, axis=1).T
    out.to_csv(IN_DIR / "mds_spearman_summary.csv")
    print(out.round(3).to_string())
    mds_table(summary)
    return freq


def mds_table(summary):
    """Appendix table on the reconstruction: reproducibility, stress and rank correlations."""
    st = pd.read_csv(MDS_STABILITY_CSV_PATH)
    stress = pd.read_csv(IN_DIR / "mds_stress.csv").set_index("Map")["Stress"]
    repeat, seed42 = pd.read_csv(IN_DIR / "mds_reproducibility.csv")["Disparity"]
    a, c = st[st["Variant"] == "aligned"], st[st["Variant"] == "contained"]
    ref = stress["reference"]
    seeds = stress.drop("reference")
    rank_ref = int((stress < ref).sum()) + 1
    at_limit = int((a["Iterations"] >= 500).sum())

    def med(variant, key):
        x = summary[variant][key]
        return f"{x['50%']:.2f} [{x['min']:.2f}, {x['max']:.2f}]"

    def both(value):
        return f"\\multicolumn{{2}}{{c}}{{{value}}}"

    rows = [
        f"Disparity of a repeated seed / of seed 42 with the reference & {both(f'{repeat:.0e} / {seed42:.0e}')} \\\\",
        f"Stress of the reference map (rank among the 21 maps) & {both(f'{ref:.4f} ({rank_ref})')} \\\\",
        f"Stress of the 20 seeds, median [range] & {both(f'{seeds.median():.4f} [{seeds.min():.4f}, {seeds.max():.4f}]')} \\\\",
        f"Runs reaching the iteration limit & {both(f'{at_limit} of 20')} \\\\",
        f"Procrustes disparity with the reference, median & {both(f'{a.Disparity.median():.3f}')} \\\\",
        f"Displacement, median (km) & {a.Shift_Median_km.median():.0f} & {c.Shift_Median_km.median():.0f} \\\\",
        f"Interventions with changed $\\mu_{{\\mathrm{{ski}}}}$ (\\%) & {100 * a.Ski_Changed.median():.0f} & {100 * c.Ski_Changed.median():.0f} \\\\",
        f"Interventions with changed $\\mu_{{\\mathrm{{park}}}}$ (\\%) & {100 * a.Park_Changed.median():.0f} & {100 * c.Park_Changed.median():.0f} \\\\",
        f"Spearman of $c_5$ between maps, median [range] & {med('aligned', 'pairs_c5')} & {med('contained', 'pairs_c5')} \\\\",
        f"Spearman of $c_6$ between maps, median [range] & {med('aligned', 'pairs_c6')} & {med('contained', 'pairs_c6')} \\\\",
        f"Spearman of $c_5$ / $c_6$ between orientations, median [range] & {both(med('orientation', 'pairs_c5') + ' / ' + med('orientation', 'pairs_c6'))} \\\\",
    ]
    write_table("mds_stability_table", "Measure & Aligned & Aligned and contained", rows, "lcc")


def map_figure(freq):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=False)
    for ax, (s, p) in zip(axes.ravel(), COMBOS):
        g = freq[(freq["Scenario"] == s) & (freq["Profile"] == p)].sort_values("Mean", ascending=False).head(8)[::-1]
        y = np.arange(len(g))
        ax.hlines(y, g["Min"], g["Max"], color="0.6", lw=2)
        ax.scatter(g["Mean"], y, c=g["Wins"], cmap="Blues", vmin=0, vmax=1, edgecolor="k", zorder=3, s=40)
        ax.set_yticks(y, [f"{a} ({100 * w:.0f}%)" for a, w in zip(g["Alternative"], g["Wins"])], fontsize=7)
        ax.set_title(f"{s} / {PROFILE_NAMES[p]}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel(r"stability radius $\rho$ across maps", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_AND_TABLES_DIR / "map_radius_ranges.png", dpi=200)


NORM_NAMES = {"L2": "$L_2$", "L1": "$L_1$", "capacity": "Capacity", "slope": "Slope"}
NORMS = list(NORM_NAMES)


def norm_tables():
    """Selection under each norm, invariance under refinement of the grid."""
    sel = pd.read_csv(IN_DIR / "norm_selection_summary.csv")
    rows = []
    for s, p in COMBOS:
        cells = []
        for norm in NORMS:
            r = sel[(sel.Scenario == s) & (sel.Profile == p) & (sel.Norm == norm)].iloc[0]
            cells.append(f"{tex_id(r.Tied_Winners)} ({100 * r.Relative_Gap:.0f}; {r.Share_dominance:.2f})")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("norm_selection_table",
                "Scenario & Profile & " + " & ".join(NORM_NAMES[n] for n in NORMS), rows, "llllll")

    ref = pd.read_csv(IN_DIR / "norm_refinement_summary.csv")
    rows = [f"{NORM_NAMES[r.Norm]} & {r.Grid.replace('_', ' ')} & {r.K} & {100 * r.Max_Relative_Change:.1f} & "
            f"{r.Same_Winner} of {r.Combinations} \\\\" for r in ref.itertuples()]
    write_table("norm_refinement_table",
                r"Norm & Grid & $K$ & Largest change of $\rho_i$ (\% of $\rho^*$) & Same selection",
                rows, "llccc")


def weight_sensitivity_tables():
    """Perturbations of p: share of draws that keep the selection, by design and norm."""
    df = pd.read_csv(IN_DIR / "weight_sensitivity_summary.csv")
    designs = [("free", e) for e in (0.001, 0.01, 0.05, 0.15)] + \
              [("tie_preserving", e) for e in (0.001, 0.01, 0.05, 0.15)] + [("lattice", 0.05)]
    rows = []
    for design, eta in designs:
        g = df[(df.Design_Group == design) & (df.Eta == eta)]
        label = "0.05 lattice" if design == "lattice" else f"{design.replace('_', '-')}, $\\eta={eta}$"
        cells = [f"{100 * g[g.Norm == n].Share_Same.mean():.0f} ({100 * g[g.Norm == n].Median_Radii_Change.median():.0f})"
                 for n in NORMS]
        rows.append(f"{label} & {g.Median_K.median():.0f} & {g.Median_Max_Change_p.median():.4f} & "
                    + " & ".join(cells) + " \\\\")
    write_table("norm_continuity_table",
                r"Perturbation of $\mathbf p$ & $K$ & $\max_j|p'_j-p_j|$ & "
                + " & ".join(NORM_NAMES[n] for n in NORMS), rows, "lcc" + "c" * len(NORMS))

    rows = []
    for s, p in COMBOS:
        cells = []
        for design, eta in (("free", 0.01), ("tie_preserving", 0.01), ("lattice", 0.05)):
            r = df[(df.Design_Group == design) & (df.Eta == eta) & (df.Norm == "L2")
                   & (df.Scenario == s) & (df.Profile == p)].iloc[0]
            cells.append(f"{100 * r.Share_Same:.0f} & {tex_id(r.Most_Frequent)} ({100 * r.Share_Most_Frequent:.0f})")
        rows.append(f"{s} & {PROFILE_NAMES[p]} & " + " & ".join(cells) + " \\\\")
    write_table("p_sensitivity_table",
                r"Scenario & Profile & \multicolumn{2}{c}{free, $\eta=0.01$} & "
                r"\multicolumn{2}{c}{tie-preserving, $\eta=0.01$} & \multicolumn{2}{c}{0.05 lattice} \\"
                "\n & & Same (\\%) & Most frequent (\\%) & Same (\\%) & Most frequent (\\%) & Same (\\%) & Most frequent (\\%)",
                rows, "llclclcl")


def main():
    ensure_output_tree()
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    sensitivity_table(results)
    shapley_deviation_table(results)
    patterns_table()
    if (IN_DIR / "mds_radii.csv").exists():
        map_figure(map_tables())
    if (IN_DIR / "weight_sensitivity_summary.csv").exists():
        norm_tables()
        weight_sensitivity_tables()
    print(f"Tables in {OUTPUT_TEX_DIR}; figures in {OUTPUT_PLOTS_AND_TABLES_DIR}")


if __name__ == "__main__":
    main()
