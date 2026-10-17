"""
Checks of the spatial reconstruction of X12 (Appendix B): reproducibility of the MDS runs,
stress of every map, pairwise rank correlations of c5 and c6 across maps, radius of every
alternative on every map, effect of the orientation of the map and ablation of c5 and c6.
Uses the MDS runs cached by mds_stability_analysis.py; writes to output/csv/supplementary/.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_matrix_builder import collect_solution_paths  # noqa: E402
from maintenance_scheduling_model import (  # noqa: E402
    MaintenanceSchedule,
    load_maintenance_instance_from_json,
)
from mds_stability_analysis import SEEDS, SPATIAL, align, containment  # noqa: E402
from project_paths import (  # noqa: E402
    DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
    OUTPUT_DIR,
    OUTPUT_NPY_DIR,
    RAW_DECISION_MATRIX_CSV_PATH,
    REGIONAL_NATURAL_PARKS_CSV_PATH,
    SUPPLEMENTARY_CSV_DIR,
    X12_DIR,
)
from spatial_memberships import (  # noqa: E402
    compute_park_proximity_membership,
    compute_ski_area_proximity_membership,
    load_france_regions,
    load_regional_natural_parks,
    load_ski_stations,
    scale_embedding_to_france,
)
from spatial_reconstruction import dissimilarity_and_weights  # noqa: E402
from wowa_stability_analysis import (  # noqa: E402
    PROFILES,
    WOWAStabilityAnalyzer,
    build_weighting_scenarios,
    shapley_bounds_around_weights,
)

OUT_DIR = SUPPLEMENTARY_CSV_DIR
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")


def run_r(dist, weights, body):
    """Run an R snippet with delta (dist object) and w loaded; returns the temp dir content."""
    with tempfile.TemporaryDirectory() as tmp:
        np.savetxt(f"{tmp}/delta.csv", dist, delimiter=",")
        np.savetxt(f"{tmp}/weight.csv", weights, delimiter=",")
        code = f"""
        suppressMessages(library(smacof))
        delta <- as.dist(as.matrix(read.csv("{tmp}/delta.csv", header = FALSE)))
        w <- as.matrix(read.csv("{tmp}/weight.csv", header = FALSE))
        tmp <- "{tmp}"
        {body}
        """
        subprocess.run([RSCRIPT, "-e", code], check=True, capture_output=True)
        return {p.name: p.read_text() for p in Path(tmp).glob("out_*")}


def reproducibility(dist, weights, reference, runs):
    """Repeat seed 1 and the original seed 42, and evaluate the stress of every map."""
    rows = []
    configs = {"reference": reference} | {s: r["points"] for s, r in runs.items()}
    with tempfile.TemporaryDirectory() as conf_dir:
        for k, v in configs.items():
            np.savetxt(f"{conf_dir}/conf_{k}.csv", v, delimiter=",")
        names = ",".join(f'"{k}"' for k in configs)
        out = run_r(dist, weights, f"""
        for (s in c(1, 42)) {{
          set.seed(s)
          res <- smacofSym(delta = delta, weightmat = w, ndim = 2, type = "ordinal",
                           init = "random", itmax = 500, modulus = 1)
          write.table(res$conf, sprintf("%s/out_conf_%d.csv", tmp, s), sep = ",",
                      row.names = FALSE, col.names = FALSE)
        }}
        for (k in c({names})) {{
          x <- as.matrix(read.csv(sprintf("{conf_dir}/conf_%s.csv", k), header = FALSE))
          write(stress0(delta, init = x, type = "ordinal", weightmat = w)$stress,
                sprintf("%s/out_stress_%s.txt", tmp, k))
        }}
        """)
    repeat_1 = np.loadtxt(out["out_conf_1.csv"].splitlines(), delimiter=",")
    seed_42 = np.loadtxt(out["out_conf_42.csv"].splitlines(), delimiter=",")
    rows.append({"Check": "seed 1 repeated vs cached seed 1",
                 "Disparity": procrustes(runs[1]["points"], repeat_1)[2],
                 "Max_Abs_Diff": np.abs(runs[1]["points"] - repeat_1).max()})
    rows.append({"Check": "seed 42 (original) vs reference map",
                 "Disparity": procrustes(reference, seed_42)[2],
                 "Max_Abs_Diff": np.abs(reference - seed_42).max()})
    stress = {k: float(out[f"out_stress_{k}.txt"]) for k in configs}
    return pd.DataFrame(rows), stress


def load_schedules(instance, names):
    schedules = {}
    for path in collect_solution_paths(X12_DIR):
        duration, team, seed = re.match(r"sol(\d+)_t(\d+)_X_12_s(\d+)", Path(path).name).groups()
        name = f"T{team}_D{duration}_S{seed}"
        if name in names:
            schedules[name] = MaintenanceSchedule(path)
            schedules[name].compute_active_interventions(instance)
    return schedules


def spatial_matrix(raw, schedules, instance, mu_ski, mu_park):
    matrix = raw.copy()
    for name, schedule in schedules.items():
        values = schedule.compute_raw_criteria(instance, mu_ski, mu_park)
        matrix.loc[name, SPATIAL] = [values[c] for c in SPATIAL]
    return matrix


def all_radii(raw, scenarios=None):
    """Radius of every alternative in every scenario and profile (main setting)."""
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        raw.to_csv(tmp.name)
        for name, weights in (scenarios or build_weighting_scenarios()).items():
            analyzer = WOWAStabilityAnalyzer(tmp.name, weights)
            bounds = shapley_bounds_around_weights(weights)
            for profile, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, details=False)
                best = df["Rho"].max()
                for _, r in df.iterrows():
                    rows.append({"Scenario": name, "Profile": profile, "Alternative": r["Alternative"],
                                 "Rho": r["Rho"], "Winner": bool(r["Rho"] == best and best > 1e-9)})
    return pd.DataFrame(rows)


def orientations(reference):
    """Rotations by multiples of 45 degrees, with and without reflection, about the centroid."""
    centre = reference.mean(0)
    out = {}
    for reflect in (False, True):
        base = (reference - centre) * ([-1, 1] if reflect else [1, 1])
        for angle in range(0, 360, 45):
            t = np.radians(angle)
            rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            out[f"{'R' if reflect else ''}{angle}"] = base @ rot.T + centre
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "X_12_mean_risk.json").open("r", encoding="utf-8") as f:
        instance = load_maintenance_instance_from_json(json.load(f))
    raw = pd.read_csv(RAW_DECISION_MATRIX_CSV_PATH, index_col=0)
    schedules = load_schedules(instance, set(raw.index))
    keys, dist, weights = dissimilarity_and_weights(instance)
    reference = np.load(DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH)
    runs = {s: np.load(OUTPUT_NPY_DIR / f"mds_X12_seed{s}.npz") for s in SEEDS}

    checks, stress = reproducibility(dist, weights, reference, runs)
    checks.to_csv(OUT_DIR / "mds_reproducibility.csv", index=False)
    pd.DataFrame({"Map": list(stress), "Stress": list(stress.values())}).to_csv(
        OUT_DIR / "mds_stress.csv", index=False)
    print(checks.to_string(), "\n", pd.Series(stress).describe())

    france = load_france_regions()
    parks, ski = load_regional_natural_parks(REGIONAL_NATURAL_PARKS_CSV_PATH), load_ski_stations()
    aligned = {"reference": reference} | {s: align(r["points"], reference) for s, r in runs.items()}
    variants = {"aligned": {k: scale_embedding_to_france(v, france) for k, v in aligned.items()}}
    variants["contained"] = containment(variants["aligned"], france)[0]
    variants["orientation"] = {k: scale_embedding_to_france(v, france)
                               for k, v in orientations(reference).items()}

    radii, correlations = [], []
    for variant, configs in variants.items():
        matrices = {}
        for k, v in configs.items():
            mu_ski = compute_ski_area_proximity_membership(v, keys, ski, already_scaled=True)
            mu_park = compute_park_proximity_membership(v, keys, parks, already_scaled=True)
            matrices[k] = spatial_matrix(raw, schedules, instance, mu_ski, mu_park)
            radii.append(all_radii(matrices[k]).assign(Variant=variant, Map=str(k)))
            print(variant, k)
        for a, b in combinations(matrices, 2):
            correlations.append({
                "Variant": variant, "Map_A": str(a), "Map_B": str(b),
                **{f"Spearman_{c}": spearmanr(matrices[a][c], matrices[b][c])[0] for c in SPATIAL},
            })

    ablation = {
        name: {c: w / (1 - sum(p[s] for s in SPATIAL)) for c, w in p.items() if c not in SPATIAL}
        for name, p in build_weighting_scenarios().items()
    }
    radii.append(all_radii(raw, ablation).assign(Variant="without_c5_c6", Map="none"))
    pd.concat(radii, ignore_index=True).to_csv(OUT_DIR / "mds_radii.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT_DIR / "mds_pairwise_spearman.csv", index=False)
    print(f"Saved results in {OUT_DIR}")


if __name__ == "__main__":
    main()
