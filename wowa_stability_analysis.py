"""
WOWA stability analysis implementation.
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
    WOWA_STABILITY_RESULTS_CSV_PATH,
    WOWA_STABILITY_SCENARIOS_CSV_PATH,
    ensure_output_tree,
)


def project_to_simplex_hyperplane(vec):
    """
    Project a vector onto the hyperplane orthogonal to the all-ones vector.
    """
    arr = np.asarray(vec, dtype=float)
    return arr - np.mean(arr)


def projected_simplex_norm(vec):
    """
    Euclidean norm of the projection onto the simplex hyperplane.
    """
    return float(np.linalg.norm(project_to_simplex_hyperplane(vec)))


def margin_coefficient(vec, norm="L2"):
    """
    Margin coefficient kappa(n) = max{n^T d : 1^T d = 0, ||d|| <= 1}.
    """
    arr = np.asarray(vec, dtype=float)
    if norm == "L2":
        return projected_simplex_norm(arr)
    if norm == "L1":
        return float((arr.max() - arr.min()) / 2.0)
    if norm == "capacity":
        return float(np.abs(np.diff(arr)).sum())
    raise ValueError(f"Unknown norm {norm}")


def normalized_entropy(vec):
    arr = np.asarray(vec, dtype=float)
    arr = arr[arr > 1e-12]
    if len(vec) <= 1:
        return 0.0
    return float(-(arr * np.log(arr)).sum() / log(len(vec)))


def shapley_bounds_around_weights(criterion_importances, tolerance=0.1):
    """
    Shapley importance bounds centred on the weighting vector, clipped to [0, 1].
    """
    return {
        col: (max(0.0, p - tolerance), min(1.0, p + tolerance))
        for col, p in criterion_importances.items()
    }


class WOWAStabilityAnalyzer:
    def __init__(
        self,
        raw_decision_matrix_path,
        criterion_importances,
        verbose: bool = False,
        complete_grid: bool = True,
    ):
        self.raw_decision_matrix_path = raw_decision_matrix_path
        self.criterion_importances = criterion_importances
        self.verbose = verbose
        self.normalized_decision_matrix: pd.DataFrame | None = None
        self.normalized_criteria_matrix: pd.DataFrame | None = None
        self.transformed_alternative_matrix: np.ndarray | None = None
        self.unified_importance_grid: list[float] | None = None
        self.orness_coefficients: np.ndarray | None = None
        self.alternatives: list[str] | None = None
        self.complete_grid = complete_grid
        self.shapley_matrix: np.ndarray | None = None

        self.load_and_normalize_decision_matrix()

    def load_and_normalize_decision_matrix(self):
        if not os.path.exists(self.raw_decision_matrix_path):
            raise FileNotFoundError(f"{self.raw_decision_matrix_path} not found.")

        df_raw = pd.read_csv(self.raw_decision_matrix_path)

        directions = {
            "Reliability Stress": False,
            "Resource Slack": True,
            "Ski Season Risk": False,
            "Social Weekend Risk": False,
            "Max Critical Overlap": False,
            "Risk Volatility": False,
            "Summer Eco-Stress": False,
        }

        for col in directions.keys():
            if col not in df_raw.columns:
                raise ValueError(f"Column {col} missing in CSV")

        self.alternatives = df_raw["Alternative"].tolist()
        self.normalized_decision_matrix = df_raw.copy()

        for col, is_max in directions.items():
            max_val = df_raw[col].max()
            min_val = df_raw[col].min()

            if max_val == 0:
                self.normalized_decision_matrix[col] = 1.0
                continue

            if is_max:
                self.normalized_decision_matrix[col] = df_raw[col] / max_val
            else:
                self.normalized_decision_matrix[col] = 1.0 - (
                    (df_raw[col] - min_val) / max_val
                )

        normalized_criteria_matrix = self.normalized_decision_matrix.loc[
            :, [col for col in directions if col in self.criterion_importances]
        ]
        self.normalized_criteria_matrix = pd.DataFrame(normalized_criteria_matrix)

    def build_unified_importance_grid(self):
        """
        Build the unified grid Omega from all alternatives' priority breakpoints.
        """
        if self.complete_grid:
            self.build_complete_weight_grid()
            return

        breakpoints = set([0.0, 1.0])

        if self.normalized_criteria_matrix is None:
            raise RuntimeError("Normalized criteria matrix is not available.")

        vals = self.normalized_criteria_matrix.to_numpy(dtype=float)
        n_alts, _ = vals.shape
        criterion_importances = np.array(
            [
                self.criterion_importances[col]
                for col in self.normalized_criteria_matrix.columns
            ]
        )

        for i in range(n_alts):
            row_values = vals[i]
            sort_idx = np.argsort(-row_values)
            cum_i = np.cumsum(criterion_importances[sort_idx])
            breakpoints.update(np.round(cum_i, 10))

        self.unified_importance_grid = sorted(list(breakpoints))

    def _exact_weights(self):
        if self.normalized_criteria_matrix is None:
            raise RuntimeError("Normalized criteria matrix is not available.")
        return [
            Fraction(str(round(self.criterion_importances[col], 10)))
            for col in self.normalized_criteria_matrix.columns
        ]

    def build_complete_weight_grid(self):
        """
        Build the complete grid of distinct subset weights p(B), with exact arithmetic.
        """
        weights = self._exact_weights()
        n_criteria = len(weights)
        self.subset_weights = {
            subset: sum((weights[j] for j in subset), Fraction(0))
            for size in range(n_criteria + 1)
            for subset in combinations(range(n_criteria), size)
        }
        grid = sorted(set(self.subset_weights.values()))
        self.grid_index = {value: k for k, value in enumerate(grid)}
        self.unified_importance_grid = [float(value) for value in grid]

    def build_wowa_coefficients(self):
        """
        Orness coefficients o (Marichal) and Shapley matrix G on the complete grid.
        """
        weights = self._exact_weights()
        n_criteria = len(weights)
        k_segments = len(self.grid_index) - 1
        orness = np.zeros(k_segments)
        shapley = np.zeros((n_criteria, k_segments))
        for subset, weight in self.subset_weights.items():
            size = len(subset)
            k_subset = self.grid_index[weight]
            if size < n_criteria:
                coef = factorial(n_criteria - size) * factorial(size) / factorial(
                    n_criteria
                )
                orness[:k_subset] += coef / (n_criteria - 1)
                coef = factorial(size) * factorial(n_criteria - size - 1) / factorial(
                    n_criteria
                )
                for j in set(range(n_criteria)) - set(subset):
                    k_union = self.grid_index[weight + weights[j]]
                    shapley[j, k_subset:k_union] += coef
        self.orness_coefficients = orness
        self.shapley_matrix = shapley

    def check_wowa_identities(self, tol=1e-9):
        """
        Numerical checks of the unified representation, Shapley and orness formulas.
        """
        if self.transformed_alternative_matrix is None:
            self.build_transformed_alternative_matrix()
        grid = np.array(self.unified_importance_grid)
        weights = np.array([float(w) for w in self._exact_weights()])
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
                assert abs(direct - w_q @ self.transformed_alternative_matrix[i]) < 1e-7
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

    def build_transformed_alternative_matrix(self):
        """
        Construct the transformed alternative matrix where rows are transformed alternatives.
        """
        if self.unified_importance_grid is None:
            self.build_unified_importance_grid()
        if (
            self.normalized_criteria_matrix is None
            or self.unified_importance_grid is None
        ):
            raise RuntimeError("WOWA transformed-space inputs are not available.")

        vals = self.normalized_criteria_matrix.to_numpy(dtype=float)
        n_alts, n_criteria = vals.shape
        grid = self.unified_importance_grid
        k_segments = len(grid) - 1
        criterion_importances = np.array(
            [
                self.criterion_importances[col]
                for col in self.normalized_criteria_matrix.columns
            ]
        )

        self.transformed_alternative_matrix = np.zeros((n_alts, k_segments))
        if self.complete_grid:
            self.build_wowa_coefficients()
        elif k_segments <= 1:
            self.orness_coefficients = np.array([0.0] * k_segments, dtype=float)
        else:
            self.orness_coefficients = np.array(
                [(k_segments - (k + 1)) / (k_segments - 1) for k in range(k_segments)],
                dtype=float,
            )

        if self.verbose:
            print(f"Grid constructed with {k_segments} segments.")

        transformed_matrix = self.transformed_alternative_matrix
        orness_coefficients = self.orness_coefficients

        for i in range(n_alts):
            row_vals = vals[i]
            sort_idx = np.argsort(-row_vals)
            sorted_vals = row_vals[sort_idx]
            sorted_i = criterion_importances[sort_idx]
            cum_i = np.cumsum(sorted_i)

            criterion_idx = 0
            for k in range(k_segments):
                t_end = grid[k + 1]
                while (
                    criterion_idx < n_criteria and cum_i[criterion_idx] < t_end - 1e-9
                ):
                    criterion_idx += 1

                if criterion_idx < n_criteria:
                    transformed_matrix[i, k] = sorted_vals[criterion_idx]
                else:
                    transformed_matrix[i, k] = sorted_vals[-1]

        self.transformed_alternative_matrix = transformed_matrix
        self.orness_coefficients = orness_coefficients

    @staticmethod
    def _add_margin(model, rho, margins, expr, rhs, kind, label, kappa):
        constr = model.addConstr(expr - kappa * rho >= rhs, f"{kind}_{label}")
        margins.append((constr, kind, label, kappa))

    def _build_ball_model(self, alpha_min, alpha_max, shapley_bounds, norm):
        """
        Ball of radius rho inside the admissible set (simplex, orness, Shapley bounds).
        Margin-free versions are implied by rho >= 0 and are therefore omitted.
        """
        if self.transformed_alternative_matrix is None:
            self.build_transformed_alternative_matrix()
        k_segments = self.transformed_alternative_matrix.shape[1]
        model = gp.Model("Stability")
        model.setParam("OutputFlag", 0)
        model.setParam("DualReductions", 0)
        w = model.addVars(k_segments, lb=0.0, name="w")
        rho = model.addVar(lb=0.0, name="rho")
        model.addConstr(w.sum() == 1.0, "SumWeights")
        margins = []
        eye = np.eye(k_segments)
        for k in range(k_segments):
            kappa = margin_coefficient(eye[k], norm)
            self._add_margin(model, rho, margins, w[k], 0.0, "simplex", k, kappa)

        def add_bounds(coefs, low, high, kind, label):
            expr = gp.quicksum(coefs[k] * w[k] for k in range(k_segments))
            kappa = margin_coefficient(coefs, norm)
            if low > 0.0:
                self._add_margin(model, rho, margins, expr, low, kind, f"{label}min", kappa)
            if high < 1.0:
                self._add_margin(model, rho, margins, -expr, -high, kind, f"{label}max", kappa)

        add_bounds(self.orness_coefficients, alpha_min, alpha_max, "orness", "")
        for j, col in enumerate(self.normalized_criteria_matrix.columns):
            if shapley_bounds and col in shapley_bounds:
                low, high = shapley_bounds[col]
                add_bounds(self.shapley_matrix[j], low, high, "shapley", f"{col}_")
        return model, w, rho, margins

    def reference_radius(self, alpha_min, alpha_max, shapley_bounds=None, norm="L2"):
        """
        Radius of the largest ball in the admissible set, without dominance constraints.
        """
        model, _, rho, _ = self._build_ball_model(alpha_min, alpha_max, shapley_bounds, norm)
        model.setObjective(rho, GRB.MAXIMIZE)
        model.optimize()
        return model.ObjVal if model.status == GRB.OPTIMAL else np.nan

    def is_uniform_winner(
        self, i_cand, alpha_min, alpha_max, shapley_bounds=None, choice_set=None
    ):
        """
        Check whether alternative i_cand is optimal for every admissible W.
        """
        z = self.transformed_alternative_matrix
        model, w, rho, _ = self._build_ball_model(alpha_min, alpha_max, shapley_bounds, "L2")
        rho.ub = 0.0
        for i_riv in range(len(z)) if choice_set is None else choice_set:
            if i_riv == i_cand:
                continue
            diff = z[i_cand] - z[i_riv]
            model.setObjective(gp.quicksum(diff[k] * w[k] for k in range(len(diff))))
            model.optimize()
            if model.status != GRB.OPTIMAL or model.ObjVal < -1e-9:
                return False
        return True

    def solve_stability_radius_lp(
        self,
        alpha_min=0.0,
        alpha_max=0.4,
        shapley_bounds=None,
        norm="L2",
        choice_set=None,
        details=True,
    ):
        """
        Solve the LP for each alternative and return stability radii.
        """
        if self.transformed_alternative_matrix is None:
            self.build_transformed_alternative_matrix()
        if self.transformed_alternative_matrix is None or self.alternatives is None:
            raise RuntimeError("Stability analysis data is not available.")

        z = self.transformed_alternative_matrix
        alternatives = self.alternatives
        orness, shapley = self.orness_coefficients, self.shapley_matrix
        criteria = list(self.normalized_criteria_matrix.columns)
        indices = range(len(z)) if choice_set is None else list(choice_set)
        results = []

        for i_cand in indices:
            model, w, rho, margins = self._build_ball_model(
                alpha_min, alpha_max, shapley_bounds, norm
            )
            for i_riv in indices:
                if i_riv == i_cand:
                    continue
                diff = z[i_cand] - z[i_riv]
                expr = gp.quicksum(diff[k] * w[k] for k in range(len(diff)))
                kappa = margin_coefficient(diff, norm)
                self._add_margin(
                    model, rho, margins, expr, 0.0, "dominance", alternatives[i_riv], kappa
                )
            model.setObjective(rho, GRB.MAXIMIZE)
            model.optimize()

            row = {"Alternative": alternatives[i_cand]}
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
                    "Num_Constraints": model.NumConstrs,
                    "Solve_Time": model.Runtime,
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
                if rival is not None and norm in ("L2", "L1"):
                    n_vec = z[i_cand] - z[alternatives.index(rival)]
                    if norm == "L2":
                        d = -project_to_simplex_hyperplane(n_vec)
                        d /= np.linalg.norm(d)
                    else:
                        d = np.zeros(len(n_vec))
                        d[np.argmin(n_vec)] += 0.5
                        d[np.argmax(n_vec)] -= 0.5
                    row["Tie_Orness_Shift"] = rho_val * float(orness @ d)
                    row.update(
                        {f"Tie_Shift_{c}": rho_val * v for c, v in zip(criteria, shapley @ d)}
                    )
            results.append(row)

        return pd.DataFrame(results)


def build_default_importance_scenarios():
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


PROFILES = [
    ("Risk_Averse", 0.00, 0.35),
    ("Neutral", 0.35, 0.65),
    ("Optimistic", 0.65, 1.00),
    ("Full_Spectrum", 0.0, 1.0),
]

# (setting name, norm, Shapley tolerance around p); the first one is the main setting.
SETTINGS = [
    ("Main", "L2", 0.1),
    ("Free_Shapley", "L2", None),
    ("Shapley_0.05", "L2", 0.05),
    ("Shapley_0.01", "L2", 0.01),
    ("L1", "L1", 0.1),
    ("Capacity", "capacity", 0.1),
]


def run_choice_set_size_experiment(
    analyzer, scenario_name, shapley_bounds, sizes=(5, 10, 15, 20, 25), n_perm=100
):
    """
    Nested random choice sets with fixed grid and normalization (main setting).
    """
    rng = np.random.default_rng(0)
    n_alts = len(analyzer.alternatives)
    rows = []
    for perm in range(n_perm):
        order = rng.permutation(n_alts)
        for size in (*sizes, n_alts):
            if size == n_alts and perm > 0:
                continue
            for profile_name, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(
                    a_min, a_max, shapley_bounds, "L2", order[:size], details=False
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

    scenarios = build_default_importance_scenarios()
    print(f"Running WOWA stability analysis from {raw_decision_matrix_path}")
    all_results, winners, grid_rows, size_rows = [], [], [], []

    for scenario_name, criterion_importances in scenarios.items():
        print(f"Scenario: {scenario_name}")
        start = pd.Timestamp.now()
        analyzer = WOWAStabilityAnalyzer(str(raw_decision_matrix_path), criterion_importances)
        analyzer.build_transformed_alternative_matrix()
        build_seconds = (pd.Timestamp.now() - start).total_seconds()
        analyzer.check_wowa_identities()
        analyzer.normalized_decision_matrix.to_csv(
            NORMALIZED_DECISION_MATRIX_CSV_PATH, index=False
        )
        bounds = shapley_bounds_around_weights(criterion_importances)
        minimal = WOWAStabilityAnalyzer(
            str(raw_decision_matrix_path), criterion_importances, complete_grid=False
        )
        minimal.build_unified_importance_grid()

        for setting_name, norm, tolerance in SETTINGS:
            setting_bounds = (
                None if tolerance is None
                else shapley_bounds_around_weights(criterion_importances, tolerance)
            )
            for profile_name, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(a_min, a_max, setting_bounds, norm)
                df = df.sort_values("Rho", ascending=False, ignore_index=True)
                rho_ref = analyzer.reference_radius(a_min, a_max, setting_bounds, norm)
                df["Rho_P"], df["Rho_Rel"] = rho_ref, df["Rho"] / rho_ref
                df["Scenario"], df["Profile"], df["Setting"] = (
                    scenario_name, profile_name, setting_name
                )
                df["Orness_Interval"] = f"[{a_min}, {a_max}]"
                all_results.append(df)
                if setting_name == "Main":
                    win = df.iloc[0].copy()
                    win["Second"], win["Rho_Second"] = df.iloc[1]["Alternative"], df.iloc[1]["Rho"]
                    win["Relative_Gap"] = (win["Rho"] - win["Rho_Second"]) / win["Rho"]
                    win["Uniform_Winner"] = analyzer.is_uniform_winner(
                        analyzer.alternatives.index(win["Alternative"]),
                        a_min, a_max, setting_bounds,
                    )
                    winners.append(win)
                    print(f"  {profile_name}: {win['Alternative']} rho={win['Rho']:.4f}")

        main_df = pd.concat([d for d in all_results if d["Setting"].iloc[0] == "Main"
                             and d["Scenario"].iloc[0] == scenario_name])
        grid_rows.append(
            {
                "Scenario": scenario_name,
                "M": len(criterion_importances),
                "N": len(analyzer.alternatives),
                "K_minimal": len(minimal.unified_importance_grid) - 1,
                "K_complete": len(analyzer.unified_importance_grid) - 1,
                "Build_Seconds": build_seconds,
                "Constraints": main_df["Num_Constraints"].max(),
                "Mean_LP_Seconds": main_df["Solve_Time"].mean(),
                "Total_LP_Seconds": main_df["Solve_Time"].sum(),
            }
        )
        size_rows.append(run_choice_set_size_experiment(analyzer, scenario_name, bounds))

    pd.concat(all_results, ignore_index=True).to_csv(WOWA_STABILITY_RESULTS_CSV_PATH, index=False)
    pd.DataFrame(winners).to_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH, index=False)
    pd.DataFrame(grid_rows).to_csv(WOWA_GRID_SUMMARY_CSV_PATH, index=False)
    pd.concat(size_rows, ignore_index=True).to_csv(WOWA_CHOICE_SET_SIZE_CSV_PATH, index=False)
    print(f"Saved results: {WOWA_STABILITY_RESULTS_CSV_PATH}")
    print(f"Saved scenario comparison: {WOWA_STABILITY_SCENARIOS_CSV_PATH}")


if __name__ == "__main__":
    main()
