"""
WOWA stability analysis implementation.
"""

import os
from typing import cast

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from project_paths import (
    NORMALIZED_DECISION_MATRIX_CSV_PATH,
    RAW_DECISION_MATRIX_CSV_PATH,
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


class WOWAStabilityAnalyzer:
    def __init__(
        self, raw_decision_matrix_path, criterion_importances, verbose: bool = False
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

        self.load_and_normalize_decision_matrix()

    def load_and_normalize_decision_matrix(self):
        if not os.path.exists(self.raw_decision_matrix_path):
            raise FileNotFoundError(f"{self.raw_decision_matrix_path} not found.")

        df_raw = pd.read_csv(self.raw_decision_matrix_path)

        directions = {
            "Reliability Stress": False,
            "Robustness Slack": True,
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
                self.normalized_decision_matrix[col] = 0.0 if is_max else 1.0
                continue

            if is_max:
                self.normalized_decision_matrix[col] = df_raw[col] / max_val
            else:
                self.normalized_decision_matrix[col] = 1.0 - (
                    (df_raw[col] - min_val) / max_val
                )

        normalized_criteria_matrix = self.normalized_decision_matrix.loc[
            :, list(directions.keys())
        ]
        self.normalized_criteria_matrix = pd.DataFrame(normalized_criteria_matrix)

    def build_unified_importance_grid(self):
        """
        Build the unified grid Omega from all alternatives' priority breakpoints.
        """
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
            breakpoints.update(cum_i)

        self.unified_importance_grid = sorted(list(breakpoints))

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
        if k_segments <= 1:
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

    def solve_stability_radius_lp(self, alpha_min=0.0, alpha_max=0.4):
        """
        Solve the LP for each alternative and return stability radii.
        """
        if self.transformed_alternative_matrix is None:
            self.build_transformed_alternative_matrix()
        if self.transformed_alternative_matrix is None or self.alternatives is None:
            raise RuntimeError("Stability analysis data is not available.")

        transformed_matrix = self.transformed_alternative_matrix
        alternatives = self.alternatives
        n_alts, k_segments = transformed_matrix.shape
        results = []
        face_margin = (
            projected_simplex_norm(np.eye(k_segments)[0]) if k_segments > 0 else 0.0
        )
        nu = (
            self.orness_coefficients
            if self.orness_coefficients is not None
            else np.zeros(k_segments)
        )
        nu_margin = projected_simplex_norm(nu) if k_segments > 0 else 0.0

        if self.verbose:
            print(f"Solving robustness optimization for {n_alts} alternatives...")

        for i_cand in range(n_alts):
            model = gp.Model("Stability")
            model.setParam("OutputFlag", 0)

            w = model.addVars(k_segments, lb=0.0, ub=1.0, name="w")
            rho = model.addVar(lb=0.0, name="rho")

            model.addConstr(
                gp.quicksum(w[k] for k in range(k_segments)) == 1.0, "SumWeights"
            )

            orness_expr = gp.quicksum(w[k] * nu[k] for k in range(k_segments))
            model.addConstr(orness_expr >= alpha_min, "OrnessMin")
            model.addConstr(orness_expr <= alpha_max, "OrnessMax")

            if nu_margin > 0.0:
                model.addConstr(
                    orness_expr - alpha_min >= rho * nu_margin,
                    "OrnessMinMargin",
                )
                model.addConstr(
                    alpha_max - orness_expr >= rho * nu_margin,
                    "OrnessMaxMargin",
                )

            for k in range(k_segments):
                model.addConstr(w[k] >= rho * face_margin, f"FaceMargin_{k}")

            vec_cand = transformed_matrix[i_cand]
            for i_riv in range(n_alts):
                if i_riv == i_cand:
                    continue
                vec_riv = transformed_matrix[i_riv]
                diff = vec_cand - vec_riv
                dist = projected_simplex_norm(diff)
                model.addConstr(
                    gp.quicksum(w[k] * diff[k] for k in range(k_segments))
                    >= rho * dist,
                    f"Dom_{i_riv}",
                )

            model.setObjective(rho, GRB.MAXIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                opt_w = [w[k].X for k in range(k_segments)]
                opt_orness = sum(opt_w[k] * nu[k] for k in range(k_segments))
                results.append(
                    {
                        "Alternative": alternatives[i_cand],
                        "Rho": model.ObjVal,
                        "Optimized_Orness": opt_orness,
                        "Is_Stable": model.ObjVal > 1e-5,
                    }
                )
            else:
                results.append(
                    {
                        "Alternative": alternatives[i_cand],
                        "Rho": 0.0,
                        "Optimized_Orness": 0.0,
                        "Is_Stable": False,
                    }
                )

        return pd.DataFrame(results)


def build_default_importance_scenarios():
    return {
        "Baseline": {
            "Reliability Stress": 0.10,
            "Max Critical Overlap": 0.25,
            "Risk Volatility": 0.10,
            "Robustness Slack": 0.20,
            "Ski Season Risk": 0.10,
            "Summer Eco-Stress": 0.10,
            "Social Weekend Risk": 0.15,
        },
        "Sustainability": {
            "Social Weekend Risk": 0.15,
            "Summer Eco-Stress": 0.25,
            "Ski Season Risk": 0.10,
            "Reliability Stress": 0.20,
            "Robustness Slack": 0.15,
            "Max Critical Overlap": 0.05,
            "Risk Volatility": 0.10,
        },
    }


def main() -> None:
    ensure_output_tree()

    raw_decision_matrix_path = RAW_DECISION_MATRIX_CSV_PATH
    if not raw_decision_matrix_path.exists():
        print(f"Missing input CSV: {raw_decision_matrix_path}")
        return

    scenarios = build_default_importance_scenarios()
    profiles = [
        ("Risk_Averse", 0.00, 0.35),
        ("Neutral", 0.35, 0.65),
        ("Optimistic", 0.65, 1.00),
        ("Full_Spectrum", 0.0, 1.0),
    ]

    print(f"Running WOWA stability analysis from {raw_decision_matrix_path}")
    final_summary_dfs = []
    normalized_matrix_written = False

    for scenario_name, criterion_importances in scenarios.items():
        total_imp = sum(criterion_importances.values())
        if abs(total_imp - 1.0) > 1e-6:
            criterion_importances = {
                key: value / total_imp for key, value in criterion_importances.items()
            }

        print(f"Scenario: {scenario_name}")
        try:
            stability_analyzer = WOWAStabilityAnalyzer(
                str(raw_decision_matrix_path),
                criterion_importances,
                verbose=False,
            )

            if not normalized_matrix_written:
                if stability_analyzer.normalized_decision_matrix is None:
                    raise RuntimeError("Normalized decision matrix is not available.")
                normalized_matrix = stability_analyzer.normalized_decision_matrix
                normalized_matrix.to_csv(
                    NORMALIZED_DECISION_MATRIX_CSV_PATH,
                    index=False,
                )
                normalized_matrix_written = True

            all_results = []
            for profile_name, a_min, a_max in profiles:
                df_res = stability_analyzer.solve_stability_radius_lp(
                    alpha_min=a_min,
                    alpha_max=a_max,
                )
                df_res["Profile"] = profile_name
                df_res["Orness_Interval"] = f"[{a_min}, {a_max}]"
                df_res["Scenario"] = scenario_name
                all_results.append(df_res.sort_values("Rho", ascending=False))

            scenario_df = pd.concat(all_results, ignore_index=True)
            winners = scenario_df.groupby("Profile").first().reset_index()
            profile_order = {p[0]: i for i, p in enumerate(profiles)}
            winners["order"] = winners["Profile"].map(
                lambda profile: profile_order[str(profile)]
            )
            winners = winners.sort_values("order").drop("order", axis=1)

            print(
                winners[
                    ["Profile", "Alternative", "Rho", "Optimized_Orness"]
                ].to_markdown(index=False, floatfmt=".4f")
            )
            final_summary_dfs.append(winners)

        except gp.GurobiError as e:
            print(f"Gurobi error in {scenario_name}: {e}")
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")

    if final_summary_dfs:
        global_df = pd.concat(final_summary_dfs, ignore_index=True)
        global_df.to_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH, index=False)
        print(f"Saved scenario comparison: {WOWA_STABILITY_SCENARIOS_CSV_PATH}")
        print(f"Saved normalized matrix: {NORMALIZED_DECISION_MATRIX_CSV_PATH}")


if __name__ == "__main__":
    main()
