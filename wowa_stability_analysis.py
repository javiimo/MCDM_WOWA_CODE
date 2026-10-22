"""
Stability-based selection with the WOWA operator.

For a weighting vector p and a decision matrix X, the complete weight grid
Omega_p = {p(B) : B subset of C} = {0 = t_0 < t_1 < ... < t_K = 1} gives a common simplex
of unified weights W_k = Q(t_k) - Q(t_{k-1}) in which the score of every alternative is
W^T z_i (Theorem 1), its orness is o^T W (Proposition 3) and its Shapley importance is
G W (Proposition 2). The stability radius rho_i of alternative a_i is the radius of the
largest ball of admissible W in which a_i is the best alternative, computed with one
Linear Program (LP_i) per alternative (Section 3).

Running this file reproduces the main experiment of Section 5 on instance X12.
"""

import os
from fractions import Fraction
from itertools import combinations
from math import factorial, log

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from project_paths import (
    NORMALIZED_DECISION_MATRIX_CSV_PATH,
    RAW_DECISION_MATRIX_CSV_PATH,
    WOWA_CHOICE_SET_SIZE_CSV_PATH,
    WOWA_GRID_SUMMARY_CSV_PATH,
    WOWA_SHAPLEY_DEVIATION_CSV_PATH,
    WOWA_STABILITY_RESULTS_CSV_PATH,
    WOWA_STABILITY_SCENARIOS_CSV_PATH,
    ensure_output_tree,
)

# Criteria of Section 4 and whether larger raw values are better (benefit criterion).
CRITERIA_DIRECTIONS = {
    "Reliability Stress": False,
    "Resource Slack": True,
    "Ski Season Risk": False,
    "Social Weekend Risk": False,
    "Max Critical Overlap": False,
    "Risk Volatility": False,
    "Summer Eco-Stress": False,
}


# ---------------------------------------------------------------------------
# Margin coefficients of the norms
# ---------------------------------------------------------------------------


def project_to_simplex_hyperplane(vec):
    """Projection Pi v = v - mean(v) 1 onto the hyperplane orthogonal to 1."""
    arr = np.asarray(vec, dtype=float)
    return arr - np.mean(arr)


def _weighted_median(arr, widths):
    order = np.argsort(arr)
    cum = np.cumsum(widths[order])
    return arr[order][np.searchsorted(cum, 0.5 * cum[-1])]


def margin_coefficient(vec, norm="L2", widths=None):
    """
    Margin coefficient kappa(n) = max{n^T d : 1^T d = 0, ||d|| <= 1} (remark on the norm).

    L2: ||Pi n||_2.  L1: (max n - min n) / 2.  Capacity, ||d|| = max_k |d_1 + ... + d_k|:
    sum_k |n_{k+1} - n_k|.  Slope, ||d|| = max_k |d_k| / (t_k - t_{k-1}): sum_k dt_k |n_k - m|,
    with m a weighted median of n; it needs the cell widths dt_k.
    """
    arr = np.asarray(vec, dtype=float)
    if norm == "L2":
        return float(np.linalg.norm(project_to_simplex_hyperplane(arr)))
    if norm == "L1":
        return float((arr.max() - arr.min()) / 2.0)
    if norm == "capacity":
        return float(np.abs(np.diff(arr)).sum())
    if norm == "slope":
        widths = np.asarray(widths, dtype=float)
        return float(widths @ np.abs(arr - _weighted_median(arr, widths)))
    raise ValueError(f"Unknown norm {norm}")


def steepest_direction(vec, norm="L2", widths=None):
    """
    Unit displacement d (1^T d = 0, ||d|| = 1) that decreases n^T d the most, so that
    W + rho d lies on the boundary n^T W = b of an active constraint (Proposition 6).
    """
    arr = np.asarray(vec, dtype=float)
    if norm == "L2":
        d = -project_to_simplex_hyperplane(arr)
        return d / np.linalg.norm(d)
    if norm == "L1":
        d = np.zeros(len(arr))
        d[np.argmin(arr)] += 0.5
        d[np.argmax(arr)] -= 0.5
        return d
    if norm == "capacity":
        return np.diff(np.concatenate([[0.0], np.sign(np.diff(arr)), [0.0]]))
    if norm == "slope":
        widths = np.asarray(widths, dtype=float)
        median = _weighted_median(arr, widths)
        d = -np.sign(arr - median) * widths
        at_median = np.isclose(arr, median)
        d[at_median] = -d.sum() * widths[at_median] / widths[at_median].sum()
        return d
    raise ValueError(f"Unknown norm {norm}")


def normalized_entropy(vec):
    arr = np.asarray(vec, dtype=float)
    arr = arr[arr > 1e-12]
    if len(vec) <= 1:
        return 0.0
    return float(-(arr * np.log(arr)).sum() / log(len(vec)))


def shapley_bounds_around_weights(weights, tolerance=0.1):
    """Importance requirements p_j - tolerance <= phi_j <= p_j + tolerance, within [0, 1]."""
    return {
        col: (max(0.0, p - tolerance), min(1.0, p + tolerance))
        for col, p in weights.items()
    }


# ---------------------------------------------------------------------------
# Unified representation and stability Linear Programs
# ---------------------------------------------------------------------------


class WOWAStabilityAnalyzer:
    """
    Stability analysis of a choice set for a fixed weighting vector p.

    Main attributes, built by build_unified_representation():
        grid                 grid points t_0 < ... < t_K of Omega_p (floats)
        z                    N x K matrix with rows z_i (Theorem 1)
        orness_coefficients  vector o (Proposition 3)
        shapley_matrix       K-column matrix G (Proposition 2)
    """

    def __init__(self, raw_decision_matrix_path, weights):
        self.raw_decision_matrix_path = raw_decision_matrix_path
        self.weights = weights
        self.alternatives: list[str] | None = None
        self.normalized_decision_matrix: pd.DataFrame | None = None
        self.normalized_criteria_matrix: pd.DataFrame | None = None
        self.grid: list[float] | None = None
        self.grid_index: dict[Fraction, int] | None = None
        self.subset_weights: dict[tuple, Fraction] | None = None
        self.z: np.ndarray | None = None
        self.orness_coefficients: np.ndarray | None = None
        self.shapley_matrix: np.ndarray | None = None
        self.load_and_normalize_decision_matrix()

    def load_and_normalize_decision_matrix(self):
        """
        Section 4.4: benefit criteria are divided by their largest value, and cost criteria
        are mapped to 1 - (x - min) / max, so that 1 is the best observed value.
        """
        if not os.path.exists(self.raw_decision_matrix_path):
            raise FileNotFoundError(f"{self.raw_decision_matrix_path} not found.")
        df_raw = pd.read_csv(self.raw_decision_matrix_path)
        for col in CRITERIA_DIRECTIONS:
            if col not in df_raw.columns:
                raise ValueError(f"Column {col} missing in CSV")

        self.alternatives = df_raw["Alternative"].tolist()
        self.normalized_decision_matrix = df_raw.copy()
        for col, is_benefit in CRITERIA_DIRECTIONS.items():
            max_val, min_val = df_raw[col].max(), df_raw[col].min()
            if max_val == 0:
                self.normalized_decision_matrix[col] = 1.0
            elif is_benefit:
                self.normalized_decision_matrix[col] = df_raw[col] / max_val
            else:
                self.normalized_decision_matrix[col] = 1.0 - (df_raw[col] - min_val) / max_val
        self.normalized_criteria_matrix = self.normalized_decision_matrix.loc[
            :, [col for col in CRITERIA_DIRECTIONS if col in self.weights]
        ].copy()

    def exact_weights(self):
        """Weights p_j as exact fractions (10 decimals), rescaled to add up to one."""
        weights = [
            Fraction(str(round(self.weights[col], 10)))
            for col in self.normalized_criteria_matrix.columns
        ]
        total = sum(weights)
        return [w / total for w in weights]

    def build_complete_weight_grid(self):
        """Complete weight grid Omega_p of Definition 4, in exact arithmetic."""
        weights = self.exact_weights()
        n_criteria = len(weights)
        self.subset_weights = {
            subset: sum((weights[j] for j in subset), Fraction(0))
            for size in range(n_criteria + 1)
            for subset in combinations(range(n_criteria), size)
        }
        grid = sorted(set(self.subset_weights.values()))
        self.grid_index = {value: k for k, value in enumerate(grid)}
        self.grid = [float(value) for value in grid]

    def minimal_grid_size(self):
        """K of the grid of Definition 3, formed by the cumulative weights of the alternatives."""
        weights = self.exact_weights()
        points = {Fraction(0)}
        for row in self.normalized_criteria_matrix.to_numpy(dtype=float):
            cum = Fraction(0)
            for j in np.argsort(-row, kind="stable"):
                cum += weights[j]
                points.add(cum)
        return len(points) - 1

    @property
    def cell_widths(self):
        """Cell widths t_k - t_{k-1}."""
        return np.diff(np.asarray(self.grid, dtype=float))

    def build_wowa_coefficients(self):
        """Orness coefficients o (Proposition 3) and Shapley matrix G (Proposition 2)."""
        weights = self.exact_weights()
        n_criteria = len(weights)
        k_segments = len(self.grid_index) - 1
        orness = np.zeros(k_segments)
        shapley = np.zeros((n_criteria, k_segments))
        for subset, weight in self.subset_weights.items():
            size = len(subset)
            k_subset = self.grid_index[weight]
            if size < n_criteria:
                coef = factorial(n_criteria - size) * factorial(size) / factorial(n_criteria)
                orness[:k_subset] += coef / (n_criteria - 1)
                coef = factorial(size) * factorial(n_criteria - size - 1) / factorial(n_criteria)
                for j in set(range(n_criteria)) - set(subset):
                    k_union = self.grid_index[weight + weights[j]]
                    shapley[j, k_subset:k_union] += coef
        self.orness_coefficients = orness
        self.shapley_matrix = shapley

    def build_unified_representation(self):
        """
        Grid, coefficients o and G, and vectors z_i: on the cells of Omega_p between
        p(B_{i,j-1}) and p(B_ij), where B_ij holds the j best criteria of x_i, z_i takes the
        j-th largest value of x_i (Theorem 1).
        """
        self.build_complete_weight_grid()
        self.build_wowa_coefficients()
        vals = self.normalized_criteria_matrix.to_numpy(dtype=float)
        weights = self.exact_weights()
        self.z = np.zeros((len(vals), len(self.grid) - 1))
        for i, row in enumerate(vals):
            start, cum = 0, Fraction(0)
            for j in np.argsort(-row, kind="stable"):
                cum += weights[j]
                end = self.grid_index[cum]
                self.z[i, start:end] = row[j]
                start = end

    def check_wowa_identities(self, tol=1e-9):
        """
        Numerical checks of Theorem 1 (direct WOWA scores equal W^T z_i for two quantifiers),
        Proposition 2 (columns of G add up to one, G W^lin = p, Shapley value of a random
        capacity) and Proposition 3 (o_1 = 1, o^T W^lin = 1/2).
        """
        if self.z is None:
            self.build_unified_representation()
        grid = np.array(self.grid)
        weights = np.array([float(w) for w in self.exact_weights()])
        w_lin = np.diff(grid)
        orness, shapley = self.orness_coefficients, self.shapley_matrix
        assert np.allclose(shapley.sum(axis=0), 1.0, atol=tol)
        assert np.allclose(shapley @ w_lin, weights, atol=tol)
        assert abs(orness @ w_lin - 0.5) < tol and abs(orness[0] - 1) < tol
        vals = self.normalized_criteria_matrix.to_numpy(dtype=float)
        for exponent in (0.5, 2.0):
            w_q = np.diff(grid**exponent)
            for i, row in enumerate(vals):
                idx = np.argsort(-row)
                cum = np.concatenate([[0.0], np.cumsum(weights[idx])])
                direct = np.diff(cum**exponent) @ row[idx]
                assert abs(direct - w_q @ self.z[i]) < 1e-7
        rng = np.random.default_rng(0)
        w_rand = rng.dirichlet(np.ones(len(w_lin)))
        cum_w = np.concatenate([[0.0], np.cumsum(w_rand)])
        capacity = {s: cum_w[self.grid_index[v]] for s, v in self.subset_weights.items()}
        n = len(weights)
        for j in range(n):
            phi = sum(
                factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                * (capacity[tuple(sorted(s + (j,)))] - capacity[s])
                for s in capacity
                if j not in s
            )
            assert abs(phi - shapley[j] @ w_rand) < tol

    @staticmethod
    def _add_margin(model, rho, margins, expr, rhs, kind, label, kappa):
        constr = model.addConstr(expr - kappa * rho >= rhs, f"{kind}_{label}")
        margins.append((constr, kind, label, kappa))

    def _build_ball_model(self, orness_min, orness_max, shapley_bounds, norm):
        """
        Ball of radius rho inside the admissible set W (simplex, orness interval and
        importance requirements): the margin constraints of (LP_i) except the dominance ones.
        The constraints without margin are implied by rho >= 0 and are omitted.
        """
        if self.z is None:
            self.build_unified_representation()
        k_segments = self.z.shape[1]
        widths = self.cell_widths
        model = gp.Model("Stability")
        model.setParam("OutputFlag", 0)
        model.setParam("DualReductions", 0)
        w = model.addVars(k_segments, lb=0.0, name="w")
        rho = model.addVar(lb=0.0, name="rho")
        model.addConstr(w.sum() == 1.0, "SumWeights")
        margins = []
        eye = np.eye(k_segments)
        for k in range(k_segments):
            kappa = margin_coefficient(eye[k], norm, widths)
            self._add_margin(model, rho, margins, w[k], 0.0, "simplex", k, kappa)

        def add_bounds(coefs, low, high, kind, label):
            expr = gp.quicksum(coefs[k] * w[k] for k in range(k_segments))
            kappa = margin_coefficient(coefs, norm, widths)
            if low > 0.0:
                self._add_margin(model, rho, margins, expr, low, kind, f"{label}min", kappa)
            if high < 1.0:
                self._add_margin(model, rho, margins, -expr, -high, kind, f"{label}max", kappa)

        add_bounds(self.orness_coefficients, orness_min, orness_max, "orness", "")
        for j, col in enumerate(self.normalized_criteria_matrix.columns):
            if shapley_bounds and col in shapley_bounds:
                low, high = shapley_bounds[col]
                add_bounds(self.shapley_matrix[j], low, high, "shapley", f"{col}_")
        return model, w, rho, margins

    def reference_radius(self, orness_min, orness_max, shapley_bounds=None, norm="L2"):
        """Radius rho_P of the largest ball in the admissible set, without dominance constraints."""
        model, _, rho, _ = self._build_ball_model(orness_min, orness_max, shapley_bounds, norm)
        model.setObjective(rho, GRB.MAXIMIZE)
        model.optimize()
        return model.ObjVal if model.status == GRB.OPTIMAL else np.nan

    def min_shapley_deviation(self, orness_min, orness_max):
        """delta* = min over W in the orness interval of max_j |phi_j(W) - p_j|."""
        if self.z is None:
            self.build_unified_representation()
        model = gp.Model("ShapleyDeviation")
        model.setParam("OutputFlag", 0)
        w = model.addMVar(self.z.shape[1], lb=0.0)
        delta = model.addVar(lb=0.0)
        target = np.array([float(p) for p in self.exact_weights()])
        model.addConstr(w.sum() == 1.0)
        model.addConstr(self.orness_coefficients @ w >= orness_min)
        model.addConstr(self.orness_coefficients @ w <= orness_max)
        model.addConstr(self.shapley_matrix @ w - target <= delta)
        model.addConstr(target - self.shapley_matrix @ w <= delta)
        model.setObjective(delta, GRB.MINIMIZE)
        model.optimize()
        return model.ObjVal

    def is_uniform_winner(
        self, i_cand, orness_min, orness_max, shapley_bounds=None, choice_set=None
    ):
        """Whether alternative i_cand is optimal for every admissible W."""
        model, w, rho, _ = self._build_ball_model(orness_min, orness_max, shapley_bounds, "L2")
        rho.ub = 0.0
        for i_riv in range(len(self.z)) if choice_set is None else choice_set:
            if i_riv == i_cand:
                continue
            diff = self.z[i_cand] - self.z[i_riv]
            model.setObjective(gp.quicksum(diff[k] * w[k] for k in range(len(diff))))
            model.optimize()
            if model.status != GRB.OPTIMAL or model.ObjVal < -1e-9:
                return False
        return True

    def solve_stability_radius_lp(
        self,
        orness_min=0.0,
        orness_max=1.0,
        shapley_bounds=None,
        norm="L2",
        choice_set=None,
        details=True,
    ):
        """
        Solve (LP_i) for every alternative of the choice set and return the radii.

        With details=True, each row also has the center W* (its orness, Shapley importance and
        entropy), the fragility shares theta_c of each type of constraint, the competitor
        with the largest share and the changes of orness and importance in the direction of
        the tie with that competitor (Proposition 6).
        """
        if self.z is None:
            self.build_unified_representation()
        z = self.z
        orness, shapley = self.orness_coefficients, self.shapley_matrix
        criteria = list(self.normalized_criteria_matrix.columns)
        indices = range(len(z)) if choice_set is None else list(choice_set)
        results = []

        for i_cand in indices:
            model, w, rho, margins = self._build_ball_model(
                orness_min, orness_max, shapley_bounds, norm
            )
            for i_riv in indices:
                if i_riv == i_cand:
                    continue
                diff = z[i_cand] - z[i_riv]
                expr = gp.quicksum(diff[k] * w[k] for k in range(len(diff)))
                kappa = margin_coefficient(diff, norm, self.cell_widths)
                self._add_margin(
                    model, rho, margins, expr, 0.0, "dominance", self.alternatives[i_riv], kappa
                )
            model.setObjective(rho, GRB.MAXIMIZE)
            model.optimize()

            row = {
                "Alternative": self.alternatives[i_cand],
                "Num_Constraints": model.NumConstrs,
                "Solve_Time": model.Runtime,
            }
            if model.status != GRB.OPTIMAL:
                row.update({"Rho": 0.0, "Status": "infeasible", "Is_Stable": False})
                results.append(row)
                continue

            center = np.array([w[k].X for k in range(len(w))])
            rho_val = model.ObjVal
            row.update(
                {
                    "Rho": rho_val,
                    "Status": "positive" if rho_val > 1e-9 else "zero",
                    "Is_Stable": rho_val > 1e-5,
                    "Optimized_Orness": float(orness @ center),
                }
            )
            if not details:
                results.append(row)
                continue

            phi = shapley @ center
            row.update({f"Shapley_{c}": phi[j] for j, c in enumerate(criteria)})
            row.update(
                {
                    "Entropy": normalized_entropy(center),
                    "Shapley_Entropy": normalized_entropy(phi),
                    "Center": " ".join(f"{v:.6f}" for v in center),
                }
            )
            if rho_val > 1e-9:
                shares = {}
                rival, rival_share = None, 0.0
                for constr, kind, label, kappa in margins:
                    share = abs(constr.Pi) * kappa
                    shares[kind] = shares.get(kind, 0.0) + share
                    if kind == "dominance" and share > rival_share:
                        rival, rival_share = label, share
                row.update({f"Share_{k}": shares.get(k, 0.0) for k in
                            ("dominance", "orness", "shapley", "simplex")})
                row["Binding"] = max(shares, key=shares.get)
                row["Rival"] = rival
                if rival is not None:
                    n_vec = z[i_cand] - z[self.alternatives.index(rival)]
                    d = steepest_direction(n_vec, norm, self.cell_widths)
                    row["Tie_Orness_Shift"] = rho_val * float(orness @ d)
                    row.update(
                        {f"Tie_Shift_{c}": rho_val * v for c, v in zip(criteria, shapley @ d)}
                    )
            results.append(row)

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Experiment of Section 5
# ---------------------------------------------------------------------------


def build_weighting_scenarios():
    """Weighting vectors p of Table 2."""
    return {
        "Baseline": {
            "Reliability Stress": 0.10,
            "Max Critical Overlap": 0.25,
            "Risk Volatility": 0.10,
            "Resource Slack": 0.20,
            "Ski Season Risk": 0.10,
            "Summer Eco-Stress": 0.10,
            "Social Weekend Risk": 0.15,
        },
        "Sustainability": {
            "Social Weekend Risk": 0.15,
            "Summer Eco-Stress": 0.25,
            "Ski Season Risk": 0.10,
            "Reliability Stress": 0.20,
            "Resource Slack": 0.15,
            "Max Critical Overlap": 0.05,
            "Risk Volatility": 0.10,
        },
    }


# Attitudinal profiles: orness intervals.
PROFILES = [
    ("Risk_Averse", 0.00, 0.35),
    ("Neutral", 0.35, 0.65),
    ("Optimistic", 0.65, 1.00),
    ("Full_Spectrum", 0.0, 1.0),
]

MAIN_NORM = "L2"
SETTINGS = [
    # (setting name, norm, Shapley tolerance around p); the first one is the main setting.
    ("Main", MAIN_NORM, 0.1),
    ("Free_Shapley", MAIN_NORM, None),
    ("Shapley_0.05", MAIN_NORM, 0.05),
    ("Shapley_0.01", MAIN_NORM, 0.01),
]


def run_choice_set_size_experiment(
    analyzer, scenario_name, shapley_bounds, sizes=(5, 10, 15, 20, 25), n_perm=100
):
    """Nested random choice sets on a fixed grid and normalization (Proposition 5)."""
    rng = np.random.default_rng(0)
    n_alts = len(analyzer.alternatives)
    rows = []
    for perm in range(n_perm):
        order = rng.permutation(n_alts)
        for size in (*sizes, n_alts):
            if size == n_alts and perm > 0:
                continue
            for profile_name, o_min, o_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(
                    o_min, o_max, shapley_bounds, MAIN_NORM, order[:size], details=False
                )
                df["Scenario"], df["Profile"] = scenario_name, profile_name
                df["Permutation"], df["N"] = perm, size
                rows.append(df[["Scenario", "Profile", "Permutation", "N", "Alternative", "Rho"]])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_output_tree()
    raw_decision_matrix_path = RAW_DECISION_MATRIX_CSV_PATH
    if not raw_decision_matrix_path.exists():
        print(f"Missing input CSV: {raw_decision_matrix_path}")
        return

    print(f"Running WOWA stability analysis from {raw_decision_matrix_path}")
    all_results, winners, grid_rows, size_rows, deviation_rows = [], [], [], [], []
    for scenario_name, weights in build_weighting_scenarios().items():
        print(f"Scenario: {scenario_name}")
        start = pd.Timestamp.now()
        analyzer = WOWAStabilityAnalyzer(str(raw_decision_matrix_path), weights)
        analyzer.build_unified_representation()
        build_seconds = (pd.Timestamp.now() - start).total_seconds()
        analyzer.check_wowa_identities()
        analyzer.normalized_decision_matrix.to_csv(NORMALIZED_DECISION_MATRIX_CSV_PATH, index=False)

        for setting_name, norm, tolerance in SETTINGS:
            bounds = None if tolerance is None else shapley_bounds_around_weights(weights, tolerance)
            for profile_name, o_min, o_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(o_min, o_max, bounds, norm)
                df = df.sort_values("Rho", ascending=False, ignore_index=True)
                rho_ref = analyzer.reference_radius(o_min, o_max, bounds, norm)
                df["Rho_P"], df["Rho_Rel"] = rho_ref, df["Rho"] / rho_ref
                df["Scenario"], df["Profile"], df["Setting"] = scenario_name, profile_name, setting_name
                df["Orness_Interval"] = f"[{o_min}, {o_max}]"
                all_results.append(df)
                if setting_name == "Main":
                    win = df.iloc[0].copy()
                    win["Second"], win["Rho_Second"] = df.iloc[1]["Alternative"], df.iloc[1]["Rho"]
                    win["Relative_Gap"] = (win["Rho"] - win["Rho_Second"]) / win["Rho"]
                    win["Uniform_Winner"] = analyzer.is_uniform_winner(
                        analyzer.alternatives.index(win["Alternative"]), o_min, o_max, bounds
                    )
                    winners.append(win)
                    deviation_rows.append({
                        "Scenario": scenario_name, "Profile": profile_name,
                        "Delta_Star": analyzer.min_shapley_deviation(o_min, o_max),
                    })
                    print(f"  {profile_name}: {win['Alternative']} rho={win['Rho']:.4f}")

        main_df = pd.concat([d for d in all_results if d["Setting"].iloc[0] == "Main"
                             and d["Scenario"].iloc[0] == scenario_name])
        grid_rows.append({
            "Scenario": scenario_name,
            "M": len(weights),
            "N": len(analyzer.alternatives),
            "K_minimal": analyzer.minimal_grid_size(),
            "K_complete": len(analyzer.grid) - 1,
            "Build_Seconds": build_seconds,
            "Constraints": main_df["Num_Constraints"].max(),
            "Mean_LP_Seconds": main_df["Solve_Time"].mean(),
            "Total_LP_Seconds": main_df["Solve_Time"].sum(),
            "Mean_LP_Seconds_Feasible": main_df.loc[main_df.Status != "infeasible", "Solve_Time"].mean(),
            "Mean_LP_Seconds_Infeasible": main_df.loc[main_df.Status == "infeasible", "Solve_Time"].mean(),
        })
        size_rows.append(run_choice_set_size_experiment(
            analyzer, scenario_name, shapley_bounds_around_weights(weights)))

    pd.concat(all_results, ignore_index=True).to_csv(WOWA_STABILITY_RESULTS_CSV_PATH, index=False)
    pd.DataFrame(winners).to_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH, index=False)
    pd.DataFrame(grid_rows).to_csv(WOWA_GRID_SUMMARY_CSV_PATH, index=False)
    pd.DataFrame(deviation_rows).to_csv(WOWA_SHAPLEY_DEVIATION_CSV_PATH, index=False)
    pd.concat(size_rows, ignore_index=True).to_csv(WOWA_CHOICE_SET_SIZE_CSV_PATH, index=False)
    print(f"Saved results: {WOWA_STABILITY_RESULTS_CSV_PATH}")
    print(f"Saved scenario comparison: {WOWA_STABILITY_SCENARIOS_CSV_PATH}")


if __name__ == "__main__":
    main()
