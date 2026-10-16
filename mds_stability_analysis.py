"""
Stability of the spatial reconstruction across MDS initializations (Procrustes
alignment) and its effect on c5, c6 and on the stability-based selection.
"""

import json
import re
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from decision_matrix_builder import RAW_CRITERION_COLUMNS, collect_solution_paths
from maintenance_scheduling_model import MaintenanceSchedule, load_maintenance_instance_from_json
from project_paths import (
    DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
    MDS_SELECTION_CSV_PATH,
    MDS_STABILITY_CSV_PATH,
    OUTPUT_DIR,
    OUTPUT_NPY_DIR,
    RAW_DECISION_MATRIX_CSV_PATH,
    REGIONAL_NATURAL_PARKS_CSV_PATH,
    X12_DIR,
    ensure_output_tree,
)
from spatial_memberships import (
    _to_geodf,
    compute_park_proximity_membership,
    compute_ski_area_proximity_membership,
    load_france_regions,
    load_regional_natural_parks,
    load_ski_stations,
    scale_embedding_to_france,
)
from spatial_reconstruction import compute_mds_embeddings_rscript, dissimilarity_and_weights
from wowa_stability_analysis import (
    PROFILES,
    WOWAStabilityAnalyzer,
    build_default_importance_scenarios,
    shapley_bounds_around_weights,
)

SEEDS = list(range(1, 21))
SPATIAL = ["Ski Season Risk", "Summer Eco-Stress"]
METRIC = "EPSG:3857"  # same projection as the membership distances


def align(points, reference):
    """Orthogonal Procrustes with translation and isotropic scaling onto the reference."""
    a, b = reference - reference.mean(0), points - points.mean(0)
    rotation, scale = orthogonal_procrustes(b, a)
    return b @ rotation * (scale / (b**2).sum()) + reference.mean(0)


def to_metres(points_deg, france):
    return _to_geodf(points_deg, france.crs).to_crs(METRIC)


def containment(configs_deg, france, margin_m=1000.0):
    """
    Common similarity map c + s (x - c) + t (0 < s <= 1, c the mean of the reference)
    in EPSG:3857 that places every point of every configuration inside France with
    the given margin and minimizes the RMS displacement; returns transformed configs.
    """
    import shapely
    from scipy.optimize import minimize

    poly = france.to_crs(METRIC).union_all().buffer(-margin_m)
    metric = {k: np.column_stack([g.x, g.y]) for k, g in
              ((k, to_metres(v, france).geometry) for k, v in configs_deg.items())}
    points = np.vstack(list(metric.values()))
    centre = metric["reference"].mean(0)

    def moved(params):
        params = np.asarray(params, dtype=float)
        return centre + params[0] * (points - centre) + params[1:] * 1000.0

    def feasible(params):
        q = moved(params)
        return params[0] <= 1.0 and shapely.contains_xy(poly, q[:, 0], q[:, 1]).all()

    def rms(params):
        return np.sqrt(((moved(params) - points) ** 2).sum(1).mean())

    # Feasibility is not monotone in the scale (non-convex border): scan it downwards.
    best = None
    for tx in np.arange(-20, 20.1, 2.0):
        for ty in np.arange(-20, 20.1, 2.0):
            for scale in np.arange(1.0, 0.8, -0.0025):
                if feasible([scale, tx, ty]):
                    if best is None or rms([scale, tx, ty]) < rms(best):
                        best = np.array([scale, tx, ty])
                    break
    result = minimize(lambda x: rms(x) if feasible(x) else 1e12, best, method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-3})
    params = result.x if feasible(result.x) and rms(result.x) < rms(best) else best
    out = {}
    for k, q in metric.items():
        q = centre + params[0] * (q - centre) + params[1:] * 1000.0
        geo = gpd.GeoSeries(gpd.points_from_xy(q[:, 0], q[:, 1]), crs=METRIC).to_crs(france.crs)
        out[k] = np.column_stack([geo.x, geo.y])
    return out, params[0], params[1:], rms(params) / 1000.0


def winners(raw, scenarios=None):
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        raw.to_csv(tmp.name)
        for name, weights in (scenarios or build_default_importance_scenarios()).items():
            analyzer = WOWAStabilityAnalyzer(tmp.name, weights)
            bounds = shapley_bounds_around_weights(weights)
            for profile, a_min, a_max in PROFILES:
                df = analyzer.solve_stability_radius_lp(a_min, a_max, bounds, details=False)
                best = df.sort_values("Rho", ascending=False).iloc[0]
                rows.append({"Scenario": name, "Profile": profile,
                             "Winner": best["Alternative"], "Rho": best["Rho"]})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_output_tree()
    with (OUTPUT_DIR / "X_12_mean_risk.json").open("r", encoding="utf-8") as f:
        instance = load_maintenance_instance_from_json(json.load(f))
    raw = pd.read_csv(RAW_DECISION_MATRIX_CSV_PATH, index_col=0)
    schedules = {}
    for path in collect_solution_paths(X12_DIR):
        duration, team, seed = re.match(r"sol(\d+)_t(\d+)_X_12_s(\d+)", Path(path).name).groups()
        name = f"T{team}_D{duration}_S{seed}"
        if name in raw.index:
            schedules[name] = MaintenanceSchedule(path)
            schedules[name].compute_active_interventions(instance)

    keys, dist, weights = dissimilarity_and_weights(instance)
    reference = np.load(DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH)
    cache = {seed: OUTPUT_NPY_DIR / f"mds_X12_seed{seed}.npz" for seed in SEEDS}
    if not all(path.exists() for path in cache.values()):
        for seed, (points, stress, n_iter) in zip(
            SEEDS, compute_mds_embeddings_rscript(dist, weights, SEEDS)
        ):
            np.savez(cache[seed], points=points, stress=stress, n_iter=n_iter)
    runs = {seed: np.load(path) for seed, path in cache.items()}

    france = load_france_regions()
    parks, ski = load_regional_natural_parks(REGIONAL_NATURAL_PARKS_CSV_PATH), load_ski_stations()
    aligned = {"reference": reference} | {s: align(r["points"], reference) for s, r in runs.items()}
    variants = {"aligned": {k: scale_embedding_to_france(v, france) for k, v in aligned.items()}}
    variants["contained"], scale, shift, rms_km = containment(variants["aligned"], france)
    print(f"Containment: scale {scale:.4f}, shift {shift} km, RMS displacement {rms_km:.1f} km")

    stability, selection = [], []
    for variant, configs in variants.items():
        ref_m = to_metres(configs["reference"], france)
        memberships = {
            k: (compute_ski_area_proximity_membership(v, keys, ski, already_scaled=True),
                compute_park_proximity_membership(v, keys, parks, already_scaled=True))
            for k, v in configs.items()
        }
        matrices = {}
        for k, (mu_ski, mu_park) in memberships.items():
            matrices[k] = raw.copy()
            for name, schedule in schedules.items():
                values = schedule.compute_raw_criteria(instance, mu_ski, mu_park)
                matrices[k].loc[name, SPATIAL] = [values[c] for c in SPATIAL]
            selection.append(winners(matrices[k]).assign(Seed=k, Variant=variant))
        ref_ski, ref_park = memberships["reference"]
        for seed in SEEDS:
            mu_ski, mu_park = memberships[seed]
            shift_km = to_metres(configs[seed], france).distance(ref_m).to_numpy() / 1000
            stability.append({
                "Variant": variant,
                "Seed": seed,
                "Stress": float(runs[seed]["stress"]),
                "Iterations": int(runs[seed]["n_iter"]),
                "Disparity": procrustes(reference, runs[seed]["points"])[2],
                "Shift_Median_km": np.median(shift_km),
                "Shift_P90_km": np.percentile(shift_km, 90),
                "Ski_Change_Mean": (mu_ski - ref_ski).abs().mean(),
                "Park_Change_Mean": (mu_park - ref_park).abs().mean(),
                "Ski_Changed": ((mu_ski - ref_ski).abs() > 1e-9).mean(),
                "Park_Changed": ((mu_park - ref_park).abs() > 1e-9).mean(),
                "Spearman_c5": spearmanr(matrices[seed][SPATIAL[0]], matrices["reference"][SPATIAL[0]])[0],
                "Spearman_c6": spearmanr(matrices[seed][SPATIAL[1]], matrices["reference"][SPATIAL[1]])[0],
                "Containment_Scale": scale,
                "Containment_RMS_km": rms_km,
            })

    ablation = {
        name: {c: w / (1 - sum(p[s] for s in SPATIAL)) for c, w in p.items() if c not in SPATIAL}
        for name, p in build_default_importance_scenarios().items()
    }
    selection.append(winners(raw, ablation).assign(Seed="without_c5_c6", Variant="none"))
    pd.DataFrame(stability).to_csv(MDS_STABILITY_CSV_PATH, index=False)
    pd.concat(selection, ignore_index=True).to_csv(MDS_SELECTION_CSV_PATH, index=False)
    print(f"Saved {MDS_STABILITY_CSV_PATH} and {MDS_SELECTION_CSV_PATH}")


if __name__ == "__main__":
    main()
