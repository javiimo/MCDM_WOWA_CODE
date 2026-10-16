from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from pathlib import Path as FsPath
from typing import cast

REPO_ROOT = FsPath(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    NORMALIZED_DECISION_MATRIX_CSV_PATH,
    PLOTS_AND_TABLES_WINNERS_RADAR_PATH,
    WOWA_STABILITY_SCENARIOS_CSV_PATH,
    ensure_output_tree,
)


def radar_factory(num_vars, frame="circle"):
    """
    Helper function used to create the radar chart axes.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super(RadarAxes, self)._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=MplPath.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def load_profiles_for_radar() -> tuple[list[str], list[list[float]], list[str]]:
    df = pd.read_csv(NORMALIZED_DECISION_MATRIX_CSV_PATH)

    labels = [
        "Reliability\nStress",
        "Resource\nSlack",
        "Ski Season\nRisk",
        "Social\nWeekend Risk",
        "Max Critical\nOverlap",
        "Risk\nVolatility",
        "Summer Eco-Stress",
    ]

    winners = pd.read_csv(WOWA_STABILITY_SCENARIOS_CSV_PATH)["Alternative"].unique()
    alternatives = [(alternative, alternative) for alternative in winners]

    data: list[list[float]] = []
    legend_labels: list[str] = []
    for alternative_id, legend_label in alternatives:
        row = df.loc[df["Alternative"] == alternative_id]
        if row.empty:
            raise ValueError(
                f"Alternative {alternative_id} not found in the normalized matrix."
            )

        values = row.iloc[0][
            [
                "Reliability Stress",
                "Resource Slack",
                "Ski Season Risk",
                "Social Weekend Risk",
                "Max Critical Overlap",
                "Risk Volatility",
                "Summer Eco-Stress",
            ]
        ].astype(float)
        data.append(values.tolist())
        legend_labels.append(legend_label)

    return labels, data, legend_labels


def plot_robust_alternatives_radar():
    ensure_output_tree()
    labels, data, legend_labels = load_profiles_for_radar()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    N = len(labels)
    theta = radar_factory(N, frame="polygon")

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(projection="radar"))
    radar_ax = cast(PolarAxes, ax)
    fig.subplots_adjust(top=0.85, bottom=0.05)

    radar_ax.set_rgrids([0.8, 0.9, 1.0], angle=180, fontsize=16, color="grey")
    ax.set_ylim(0.7, 1.02)  # all values lie above 0.75

    for d, color, lbl in zip(data, colors, legend_labels):
        ax.plot(theta, d, color=color, linewidth=2.5, label=lbl)
        ax.fill(theta, d, facecolor=color, alpha=0.15)

    radar_ax.set_thetagrids(np.degrees(theta), labels)
    ax.tick_params(axis="x", labelsize=16, pad=20)

    for i, (label, angle) in enumerate(zip(ax.get_xticklabels(), theta)):
        if angle in (0, np.pi):
            label.set_horizontalalignment("center")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
        if i == 1:
            label.set_horizontalalignment("center")
        elif i == 2:
            label.set_horizontalalignment("right")
        elif i == 5:
            label.set_horizontalalignment("left")
        elif i == 6:
            label.set_horizontalalignment("left")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=False,
        ncol=3,
        fontsize=16,
    )


    output_path = FsPath(PLOTS_AND_TABLES_WINNERS_RADAR_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    plot_robust_alternatives_radar()
