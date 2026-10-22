"""
Generate a portrait-oriented LaTeX table for the normalized WOWA matrix.

- Uses the actual column names from the WOWA normalized decision matrix.
- Rounds numeric values to three decimals.
- Keeps the table in portrait orientation with no colouring.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import NORMALIZED_DECISION_MATRIX_CSV_PATH

CSV_PATH = Path(NORMALIZED_DECISION_MATRIX_CSV_PATH)


def load_normalized_decision_matrix(path: Path) -> pd.DataFrame:
    """Load the normalized decision matrix with safe formatting."""
    if not path.exists():
        raise SystemExit(f"CSV file {path.name} not found.")

    df = pd.read_csv(path)
    df = df.copy()
    df["Alternative"] = (
        df["Alternative"].astype(str).str.replace("_", r"\_", regex=False)
    )

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].apply(
        lambda col: col.map(lambda value: f"{value:.3f}")
    )

    # Keep the explicit column order from the CSV
    desired_order = [
        "Alternative", "Reliability Stress", "Max Critical Overlap", "Risk Volatility",
        "Resource Slack", "Ski Season Risk", "Summer Eco-Stress", "Social Weekend Risk",
    ]
    df = df[desired_order]
    return df


def build_decision_matrix_tabular(df: pd.DataFrame) -> str:
    """Return a LaTeX tabular for the given dataframe without extra deps."""
    col_format = "l" + "r" * (len(df.columns) - 1)
    header_cells = ["Alternative", *[f"$c_{j}$" for j in range(1, 8)]]
    header = " & ".join(header_cells) + r" \\"
    body_lines = [" & ".join(map(str, row)) + r" \\" for row in df.values]

    return "\n".join(
        [
            rf"\begin{{tabular}}{{{col_format}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def build_decision_matrix_document(table: str) -> str:
    """Wrap the LaTeX tabular into a compile-ready portrait document."""
    preamble = r"""\documentclass{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{caption}

\begin{document}
\section{Decision Matrix}
\label{app:decision_matrix}

\begin{table}[ht]
\centering
\small
"""
    ending = r"""\caption{Decision matrix for WOWA (normalized).}
\label{tab:dm_matrix}
\end{table}
\end{document}
"""
    return preamble + table + "\n" + ending


if __name__ == "__main__":
    df = load_normalized_decision_matrix(CSV_PATH)
    table = build_decision_matrix_tabular(df)
    print(build_decision_matrix_document(table))
