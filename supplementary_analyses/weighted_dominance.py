"""
Comparisons fixed by the weighting vector (Section 3): a_l dominates a_i for every quantifier
with weighting vector p if and only if z_l >= z_i componentwise on the complete grid, and then
a_i cannot have a positive stability radius. Counts the alternatives that are not dominated in
this sense, checks that dominated alternatives have rho <= 0 in every setting, and lists the
winners of the min and max operators, which belong to every WOWA family.
Reads the results of wowa_stability_analysis.py; writes to output/csv/supplementary/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import (  # noqa: E402
    RAW_DECISION_MATRIX_CSV_PATH,
    SUPPLEMENTARY_CSV_DIR,
    WOWA_STABILITY_RESULTS_CSV_PATH,
)
from wowa_stability_analysis import WOWAStabilityAnalyzer, build_weighting_scenarios  # noqa: E402

TOL = 1e-12


def dominated_alternatives(z: np.ndarray) -> np.ndarray:
    """Boolean mask: some other row is >= componentwise and differs somewhere."""
    geq = (z[None, :, :] >= z[:, None, :] - TOL).all(axis=2)  # geq[i, l]: z_l >= z_i
    gt = (z[None, :, :] > z[:, None, :] + TOL).any(axis=2)
    dom = geq & gt
    np.fill_diagonal(dom, False)
    return dom.any(axis=1)


def main() -> None:
    SUPPLEMENTARY_CSV_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(WOWA_STABILITY_RESULTS_CSV_PATH)
    rows = []
    for scenario, p in build_weighting_scenarios().items():
        analyzer = WOWAStabilityAnalyzer(str(RAW_DECISION_MATRIX_CSV_PATH), p)
        analyzer.build_unified_representation()
        z = analyzer.z
        dominated = dominated_alternatives(z)
        names = np.array(analyzer.alternatives)
        res = results[results.Scenario == scenario]
        rho_dominated = res[res.Alternative.isin(names[dominated])].Rho
        assert (rho_dominated.fillna(-np.inf) <= 1e-9).all(), "dominated alternative with rho > 0"
        rows.append(
            {
                "Scenario": scenario,
                "N": len(names),
                "Not_Dominated": int((~dominated).sum()),
                "Dominated": int(dominated.sum()),
                "Min_Winner": " / ".join(names[z[:, -1] >= z[:, -1].max() - TOL]),
                "Max_Winner": " / ".join(names[z[:, 0] >= z[:, 0].max() - TOL]),
                "Not_Dominated_List": " ".join(names[~dominated]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SUPPLEMENTARY_CSV_DIR / "weighted_dominance.csv", index=False)
    print(out.drop(columns="Not_Dominated_List").to_string(index=False))


if __name__ == "__main__":
    main()
