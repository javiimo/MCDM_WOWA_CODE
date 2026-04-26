from __future__ import annotations

import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memberships import (
    load_france_regions,
    load_regional_natural_parks,
    load_ski_stations,
    scale_embedding_to_france,
    build_embedding_geodataframe,
    FRANCE_REGIONS_GEOJSON_PATH,
    REGIONAL_NATURAL_PARKS_CSV_PATH,
)
from project_paths import (
    DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH,
    OUTPUT_NPY_DIR,
    PLOTS_AND_TABLES_SPATIAL_RECONSTRUCTION_MAP_PATH,
    ensure_output_tree,
)


def plot_spatial_reconstruction_map():
    ensure_output_tree()
    point_candidates = sorted(
        p for p in OUTPUT_NPY_DIR.glob("points_*.npy") if "_keys_" not in p.name
    )
    points_path = (
        point_candidates[-1]
        if point_candidates
        else DEFAULT_SPATIAL_RECONSTRUCTION_POINTS_PATH
    )

    try:
        raw_points = np.load(points_path)
    except FileNotFoundError:
        print(f"Error: points file not found at {points_path}.")
        return

    france_gdf = load_france_regions(FRANCE_REGIONS_GEOJSON_PATH)
    parks_gdf = load_regional_natural_parks(REGIONAL_NATURAL_PARKS_CSV_PATH)
    ski_gdf = load_ski_stations()
    if ski_gdf.crs != france_gdf.crs:
        ski_gdf = ski_gdf.to_crs(france_gdf.crs)

    pts_scaled = scale_embedding_to_france(raw_points, france_gdf)
    pts_gdf = build_embedding_geodataframe(pts_scaled, france_gdf)

    fig, ax = plt.subplots(figsize=(12, 12))
    france_gdf.plot(
        ax=ax, facecolor="white", edgecolor="black", linewidth=0.5, zorder=1
    )
    parks_gdf.plot(
        ax=ax, facecolor="forestgreen", edgecolor="none", alpha=0.4, zorder=2
    )
    ski_gdf.plot(ax=ax, color="blue", marker="^", markersize=20, alpha=0.7, zorder=3)
    pts_gdf.plot(
        ax=ax, color="red", markersize=30, edgecolor="white", linewidth=0.5, zorder=4
    )

    legend_elements = [
        Patch(
            facecolor="forestgreen",
            edgecolor="none",
            alpha=0.4,
            label="Regional Natural Parks",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Ski Areas",
            markerfacecolor="blue",
            markersize=8,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Interventions",
            markerfacecolor="red",
            markersize=10,
            markeredgecolor="white",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fontsize="x-large",
        title_fontsize="xx-large",
    )
    ax.set_title(
        "Spatial Reconstruction of Interventions, Regional Natural Parks, and Ski Areas",
        fontsize=24,
        pad=20,
    )
    ax.set_axis_off()
    plt.tight_layout()
    output_path = Path(PLOTS_AND_TABLES_SPATIAL_RECONSTRUCTION_MAP_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    plot_spatial_reconstruction_map()
