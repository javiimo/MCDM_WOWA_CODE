# Since the orness function is linear with respect to the weights, the level curves of orness are hyperplane intersections with this simplex.

import numpy as np
import matplotlib.pyplot as plt


def plot_orness_level_curves_n3():
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 1. DEFINE GEOMETRY (Equilateral Triangle) ---
    # Top Vertex: w=[1,0,0] (Orness 1.0)
    v_top = np.array([0.0, np.sqrt(3) / 2])

    # Bottom Left Vertex: w=[0,1,0] (Orness 0.5)
    v_left = np.array([-0.5, 0.0])

    # Bottom Right Vertex: w=[0,0,1] (Orness 0.0)
    v_right = np.array([0.5, 0.0])

    vertices = np.array([v_top, v_left, v_right])
    vertex_weights = ["[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"]
    vertex_orness = [1.0, 0.5, 0.0]

    # Edges defined by indices of connected vertices
    edges = [(0, 1), (1, 2), (0, 2)]

    # --- 2. PLOT BOUNDARY ---
    # Draw the triangle perimeter
    boundary = np.vstack([vertices, vertices[0]])
    ax.plot(boundary[:, 0], boundary[:, 1], color="#444444", linewidth=2, zorder=1)

    # --- 3. COMPUTE AND DRAW LEVEL CURVES ---
    # We include 0.5 explicitly
    orness_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Colormap
    cmap = plt.cm.viridis

    for level in orness_levels:
        intersections = []

        # Check every edge
        for idx_a, idx_b in edges:
            orn_a = vertex_orness[idx_a]
            orn_b = vertex_orness[idx_b]

            # Use small epsilon for float comparison
            eps = 1e-6
            mn, mx = min(orn_a, orn_b), max(orn_a, orn_b)

            if (mn - eps) <= level <= (mx + eps):
                # If the edge has constant orness equal to level (unlikely here)
                if abs(orn_a - orn_b) < eps:
                    intersections.append(vertices[idx_a])
                    intersections.append(vertices[idx_b])
                else:
                    # Linear Interpolation
                    t = (level - orn_a) / (orn_b - orn_a)
                    # Point coordinate
                    pt = vertices[idx_a] + t * (vertices[idx_b] - vertices[idx_a])
                    intersections.append(pt)

        # Deduplicate points (crucial for Orness 0.5 which hits a vertex shared by 2 edges)
        unique_points = []
        for p in intersections:
            if not any(np.allclose(p, up, atol=1e-5) for up in unique_points):
                unique_points.append(p)
        unique_points = np.array(unique_points)

        # Draw line if we have exactly 2 endpoints
        if len(unique_points) == 2:
            color = cmap(level)

            # Plot Line
            ax.plot(
                unique_points[:, 0],
                unique_points[:, 1],
                linestyle="-",
                color=color,
                linewidth=2,
                alpha=0.9,
                zorder=2,
            )

            # --- LABEL PLACEMENT ---
            mid = np.mean(unique_points, axis=0)

            # Calculate angle for text rotation
            dx = unique_points[1, 0] - unique_points[0, 0]
            dy = unique_points[1, 1] - unique_points[0, 1]
            angle = np.degrees(np.arctan2(dy, dx))

            # Keep text upright
            if angle < -90 or angle > 90:
                angle += 180

            # Determine text color (darker for readability if needed, or match line)
            text_color = color

            # Draw Label (Just the number)
            # Use a white box background so the line doesn't strike through the number
            ax.text(
                mid[0],
                mid[1],
                f"{level}",
                color=text_color,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=angle,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

    # --- 4. DRAW VERTICES AND LABELS ---
    # Draw points
    ax.scatter(vertices[:, 0], vertices[:, 1], s=120, color="black", zorder=10)

    # Text Style
    font_style = {"fontsize": 13, "fontweight": "bold", "color": "#222222"}

    # Top
    ax.text(
        v_top[0],
        v_top[1] + 0.06,
        f"w={vertex_weights[0]}\n(Orness 1.0)",
        ha="center",
        va="bottom",
        **font_style,
    )

    # Left (Orness 0.5)
    ax.text(
        v_left[0] - 0.06,
        v_left[1] - 0.05,
        f"w={vertex_weights[1]}\n(Orness 0.5)",
        ha="right",
        va="top",
        **font_style,
    )

    # Right (Orness 0.0)
    ax.text(
        v_right[0] + 0.06,
        v_right[1] - 0.05,
        f"w={vertex_weights[2]}\n(Orness 0.0)",
        ha="left",
        va="top",
        **font_style,
    )

    # --- 5. CLEANUP ---
    ax.set_aspect("equal")
    ax.axis("off")

    # Expand limits slightly to prevent clipping of labels
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_orness_level_curves_n3()
