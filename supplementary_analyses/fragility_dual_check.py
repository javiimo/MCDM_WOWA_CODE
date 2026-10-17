"""
Uniqueness of the fragility shares theta_c (Proposition 6).

For the most stable alternative of each scenario and profile (main setting), the shares of
each type of constraint (dominance, orness, Shapley, simplex) are minimized and maximized
over the set of optimal dual solutions of its Linear Program, and the competitors that can
receive a positive share are listed. Equal bounds and a single competitor mean that the
fragility table does not depend on the dual solution returned by the solver.
Writes output/csv/supplementary/dual_shares_range.csv.
"""

import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import RAW_DECISION_MATRIX_CSV_PATH, SUPPLEMENTARY_CSV_DIR  # noqa: E402
from project_paths import WOWA_STABILITY_SCENARIOS_CSV_PATH  # noqa: E402
from wowa_stability_analysis import (  # noqa: E402
    PROFILES,
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
    margin_coefficient,
    shapley_bounds_around_weights,
)

KINDS = ("dominance", "orness", "shapley", "simplex")


def margin_rows(analyzer, i_cand, a_min, a_max, bounds):
    """Rows n_c, right-hand sides b_c, coefficients kappa_c and kinds of (LP_i)."""
    z = analyzer.z
    k_segments = z.shape[1]
    rows = [(np.eye(k_segments)[k], 0.0, "simplex") for k in range(k_segments)]
    o = analyzer.orness_coefficients
    if a_min > 0.0:
        rows.append((o, a_min, "orness"))
    if a_max < 1.0:
        rows.append((-o, -a_max, "orness"))
    for j, col in enumerate(analyzer.normalized_criteria_matrix.columns):
        low, high = bounds[col]
        if low > 0.0:
            rows.append((analyzer.shapley_matrix[j], low, "shapley"))
        if high < 1.0:
            rows.append((-analyzer.shapley_matrix[j], -high, "shapley"))
    for i_riv in range(len(z)):
        if i_riv != i_cand:
            rows.append((z[i_cand] - z[i_riv], 0.0, "dominance", analyzer.alternatives[i_riv]))
    n = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    kappa = np.array([margin_coefficient(r[0]) for r in rows])
    labels = [r[3] if len(r) > 3 else None for r in rows]
    return n, b, kappa, [r[2] for r in rows], labels


def share_range(n, b, kappa, kinds, labels, rho_star):
    """Min and max share of each kind, and rivals with a positive share, over the optimal duals."""
    model = gp.Model("Dual")
    model.setParam("OutputFlag", 0)
    lam = model.addMVar(len(b), lb=0.0)
    mu = model.addMVar(n.shape[1], lb=0.0)
    nu = model.addMVar(1, lb=-GRB.INFINITY)
    model.addConstr(n.T @ lam + mu - np.ones((n.shape[1], 1)) @ nu == 0)  # stationarity in W
    model.addConstr(kappa @ lam >= 1.0)  # stationarity in rho >= 0
    model.addConstr(nu.sum() - b @ lam <= rho_star + 1e-9)  # dual optimality
    out = {}
    for kind in KINDS:
        mask = np.array([k == kind for k in kinds], dtype=float)
        for sense, label in ((GRB.MINIMIZE, "Min"), (GRB.MAXIMIZE, "Max")):
            model.setObjective((mask * kappa) @ lam, sense)
            model.optimize()
            out[f"{label}_{kind}"] = model.ObjVal if model.status == GRB.OPTIMAL else np.nan
    rivals = []
    for c, label in enumerate(labels):
        if label is not None:
            model.setObjective(kappa[c] * lam[c], GRB.MAXIMIZE)
            model.optimize()
            if model.ObjVal > 1e-6:
                rivals.append(label)
    out["Possible_Rivals"] = " / ".join(rivals)
    return out


def main():
    winners = pd.read_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH)
    rows = []
    for scenario, p in build_weighting_scenarios().items():
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), p)
        analyzer.build_unified_representation()
        bounds = shapley_bounds_around_weights(p)
        for profile, a_min, a_max in PROFILES:
            w = winners[(winners.Scenario == scenario) & (winners.Profile == profile)].iloc[0]
            i_cand = analyzer.alternatives.index(w.Alternative)
            n, b, kappa, kinds, labels = margin_rows(analyzer, i_cand, a_min, a_max, bounds)
            row = {"Scenario": scenario, "Profile": profile, "Alternative": w.Alternative, "Rho": w.Rho}
            row.update({f"Reported_{k}": w[f"Share_{k}"] for k in KINDS})
            row["Reported_Rival"] = w.Rival
            row.update(share_range(n, b, kappa, kinds, labels, w.Rho))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(SUPPLEMENTARY_CSV_DIR / "dual_shares_range.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
