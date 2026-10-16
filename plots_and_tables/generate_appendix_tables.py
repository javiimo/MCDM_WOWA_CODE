"""
Generate appendix LaTeX tables into output/tex using current repository data.
"""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import RTE_ChallengeROADEF2020_checker as checker
from project_paths import (
    INSTANCE_PATH,
    NORMALIZED_DECISION_MATRIX_CSV_PATH,
    OUTPUT_TEX_DIR,
    X12_DIR,
    ensure_output_tree,
)


DECISION_MATRIX_CSV = NORMALIZED_DECISION_MATRIX_CSV_PATH
SOLUTIONS_BASE = X12_DIR


def _escape_latex(value: str) -> str:
    return value.replace("_", r"\_")


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _reset_solution_starts(instance: dict) -> None:
    for intervention in instance[checker.INTERVENTIONS_STR].values():
        intervention.pop(checker.START_STR, None)


def _solution_path(team: int, duration: int, seed: int) -> Path:
    filename = f"sol{duration}_t{team}_X_12_s{seed}.txt"
    return SOLUTIONS_BASE / f"Team{team}" / filename


def _evaluate_solution(instance: dict, path: Path) -> tuple[float, float, float] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    _reset_solution_starts(instance)
    sink = StringIO()
    try:
        with redirect_stdout(sink):
            checker.read_solution_from_txt(instance, str(path))
            checker.check_all_constraints(instance)
            mean_risk, quantile = checker.compute_objective(instance)
    except Exception:
        return None

    obj_1 = float(np.mean(mean_risk))
    excess = np.maximum(
        np.asarray(quantile, dtype=float) - np.asarray(mean_risk, dtype=float), 0.0
    )
    obj_2 = float(np.mean(excess))
    alpha = float(instance[checker.ALPHA_STR])
    total = alpha * obj_1 + (1.0 - alpha) * obj_2
    return obj_1, obj_2, total


def _decision_matrix_rows(df: pd.DataFrame) -> Iterable[str]:
    for _, row in df.iterrows():
        cells = [
            _escape_latex(str(row["Alternative"])),
            f"{row['Reliability Stress']:.3f}",
            f"{row['Resource Slack']:.3f}",
            f"{row['Ski Season Risk']:.3f}",
            f"{row['Social Weekend Risk']:.3f}",
            f"{row['Max Critical Overlap']:.3f}",
            f"{row['Risk Volatility']:.3f}",
            f"{row['Summer Eco-Stress']:.3f}",
        ]
        yield " & ".join(cells) + r" \\"


def build_normalized_decision_matrix_table() -> str:
    if not DECISION_MATRIX_CSV.exists():
        raise FileNotFoundError(f"Missing {DECISION_MATRIX_CSV}")

    df = pd.read_csv(DECISION_MATRIX_CSV)
    expected_cols = [
        "Alternative",
        "Reliability Stress",
        "Resource Slack",
        "Ski Season Risk",
        "Social Weekend Risk",
        "Max Critical Overlap",
        "Risk Volatility",
        "Summer Eco-Stress",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {DECISION_MATRIX_CSV}: {missing}")
    df = df[expected_cols].copy()

    body = "\n".join(_decision_matrix_rows(df))
    return (
        r"\begin{table}[H]"
        "\n"
        r"\centering"
        "\n"
        r"\scriptsize"
        "\n"
        r"\renewcommand{\arraystretch}{0.90}"
        "\n"
        r"\caption{Decision matrix for WOWA (normalized).}"
        "\n"
        r"\label{tab:dm_matrix}"
        "\n"
        r"\begin{adjustbox}{center}"
        "\n"
        r"\begin{tabular}{lrrrrrrr}"
        "\n"
        r"\toprule"
        "\n"
        r"Alternative & \shortstack{Reliability\\Stress} & \shortstack{Resource\\Slack} & \shortstack{Ski Season\\Risk} & \shortstack{Social\\Weekend Risk} & \shortstack{Max Critical\\Overlap} & \shortstack{Risk\\Volatility} & Summer Eco-Stress \\"
        "\n"
        r"\midrule"
        "\n"
        f"{body}\n"
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{adjustbox}"
        "\n"
        r"\end{table}"
        "\n"
    )


def _solver_rows(instance: dict) -> list[str]:
    durations = [180, 300, 500, 700, 900]
    seeds = [42, 33, 73]
    teams = [1, 2, 3]

    rows: list[str] = []
    for duration in durations:
        for seed in seeds:
            for team in teams:
                alt_id = _escape_latex(f"T{team}_D{duration}_S{seed}")
                solver_name = f"Team {team}"
                metrics = _evaluate_solution(
                    instance, _solution_path(team, duration, seed)
                )
                if metrics is None:
                    mean_risk = excess = score = "---"
                else:
                    mean_risk = _format_float(metrics[0])
                    excess = _format_float(metrics[1])
                    score = _format_float(metrics[2])

                rows.append(
                    f"{alt_id} & {solver_name} & {duration} & {seed} & {mean_risk} & {excess} & {score} "
                    + r"\\"
                )
            rows.append(r"\midrule")

    if rows and rows[-1] == r"\midrule":
        rows.pop()
    return rows


def build_solver_results_table() -> str:
    if not INSTANCE_PATH.exists():
        raise FileNotFoundError(f"Missing {INSTANCE_PATH}")

    sink = StringIO()
    with redirect_stdout(sink):
        instance = checker.read_json(str(INSTANCE_PATH))

    rows = "\n".join(_solver_rows(instance))
    return (
        r"{\centering"
        "\n"
        r"\scriptsize"
        "\n"
        r"\renewcommand{\arraystretch}{0.75}"
        "\n"
        r"\captionof{table}{Solver execution results for the solution pool generation. Team~1 \citep{buljubasic2020}, Team~2 \citep{parreno2020}, and Team~3 \citep{langiu2020}. Dashes (---) indicate that no feasible solution was found.}"
        "\n"
        r"\label{tab:solver_results}"
        "\n"
        r"\begin{adjustbox}{center}"
        "\n"
        r"\begin{tabular}{llccrrr}"
        "\n"
        r"\toprule"
        "\n"
        r"\textbf{ID} & \textbf{Solver} & \textbf{Time (s)} & \textbf{Seed} & \textbf{Mean Risk} & \textbf{Excess} & \textbf{Final Score} \\"
        "\n"
        r"\midrule"
        "\n"
        f"{rows}\n"
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{adjustbox}"
        "\n"
        r"}"
        "\n"
    )


def main() -> None:
    ensure_output_tree()
    OUTPUT_TEX_DIR.mkdir(parents=True, exist_ok=True)

    dm_table_path = OUTPUT_TEX_DIR / "appendix_decision_matrix_table.tex"
    solver_table_path = OUTPUT_TEX_DIR / "appendix_solver_results_table.tex"
    combined_path = OUTPUT_TEX_DIR / "appendix_tables_generated.tex"

    decision_matrix_table = build_normalized_decision_matrix_table()
    solver_results_table = build_solver_results_table()

    dm_table_path.write_text(decision_matrix_table, encoding="utf-8")
    solver_table_path.write_text(solver_results_table, encoding="utf-8")

    combined_content = (
        r"\section{Decision Matrix}"
        "\n"
        r"\label{app:decision_matrix}"
        "\n\n"
        f"{decision_matrix_table}\n"
        r"\newpage"
        "\n"
        r"\section{Solver Execution Results}"
        "\n"
        r"\label{app:solver_results}"
        "\n"
        f"{solver_results_table}\n"
    )
    combined_path.write_text(combined_content, encoding="utf-8")

    print(f"Written {dm_table_path}")
    print(f"Written {solver_table_path}")
    print(f"Written {combined_path}")


if __name__ == "__main__":
    main()
