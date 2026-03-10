import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- Plot Configuration ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["figure.autolayout"] = True


# --- Quantifier Function ---
def quadratic_quantifier(r):
    return r**2


# --- Data Generation ---
n = 5
r_values = np.linspace(0, 1, 300)
q_values = quadratic_quantifier(r_values)

# Colores únicos para las líneas/puntos de cada criterio
criterion_colors = plt.cm.tab10(np.linspace(0, 0.5, n))

# Colores para las REGIONES DE FONDO (Morados, de claro a un poco más oscuro)
# Usamos un rango intermedio del mapa de color 'Purples' para que no sea ni blanco ni negro
purple_shades = plt.cm.Purples(np.linspace(0.3, 0.6, n))

# 1. Standard OWA (Equidistant)
prop_owa = np.linspace(0, 1, n + 1)

# 2. WOWA (Variable Importance)
# Pattern: 5%, 30%, 10%, 50%, 5%
importances = np.array([0.05, 0.30, 0.10, 0.50, 0.05])
prop_wowa = np.concatenate(([0], np.cumsum(importances)))

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

# Shared settings
arrow_args = dict(arrowstyle="<|-|>", lw=2, color="black")
dx = 0.04
gap = 0.0025  # 0.25% de separación para el hueco blanco


def plot_criteria(ax, props, is_wowa):
    # Iterar por cada criterio
    for j in range(1, n + 1):
        x0, x1 = props[j - 1], props[j]
        y0, y1 = quadratic_quantifier(x0), quadratic_quantifier(x1)
        mid_y = (y0 + y1) / 2
        mid_x = (x0 + x1) / 2

        c_color = criterion_colors[j - 1]  # Color de los puntos/líneas
        bg_color = purple_shades[j - 1]  # Color de fondo de la región

        # 1. REGIÓN DE IMPORTANCIA (Fondo Morado)
        # Dibujamos un rectángulo vertical con un pequeño gap blanco
        ax.axvspan(x0 + gap, x1 - gap, color=bg_color, alpha=0.6, zorder=0, linewidth=0)

        # Texto del porcentaje ARRIBA (Justo encima de y=1)
        pct_val = int((x1 - x0) * 100 + 0.1)
        ax.text(
            mid_x,
            1.01,
            f"{pct_val}%",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color="purple",
            zorder=1,
        )

        # 2. Curva base (encima del fondo)
        ax.plot(
            r_values, q_values, linewidth=3, color="gray", alpha=0.4, label="_nolegend_"
        )

        # 3. Líneas Verticales y Puntos
        ax.plot(
            [x1, x1],
            [0, y1],
            linestyle="--",
            linewidth=2,
            color=c_color,
            alpha=0.8,
            zorder=2,
        )
        ax.plot(x1, y1, "o", markersize=10, color=c_color, zorder=5)
        ax.plot([0, x1], [y1, y1], linestyle=":", linewidth=1, color="gray", alpha=0.3)

        # 4. Flecha de Peso (Weight)
        ax.annotate(
            "", xy=(x1 + dx, y0), xytext=(x1 + dx, y1), arrowprops=arrow_args, zorder=6
        )

        # Etiqueta del Peso
        label_text = f"$w_{{{j}}}$"
        # if not is_wowa:
        #      label_text = f'$w_{{{j}}}=Q({j}/{n})-Q({j-1}/{n})$'

        ax.text(
            x1 + dx * 1.5,
            mid_y,
            label_text,
            va="center",
            fontsize=20,
            fontweight="bold",
            color="black",
            zorder=6,
        )  # Increased fontsize


# --- Ejecutar Plots ---

# Izquierda: OWA
plot_criteria(ax1, prop_owa, is_wowa=False)
ax1.set_title(r"OWA ($w_i$ from uniform importance partition)", fontsize=24, pad=15)
ax1.set_xlabel("Accumulated Importance", fontsize=18)
ax1.set_ylabel("Quantifier Value, $Q(r)$", fontsize=18)
ax1.set_xlim(-0.02, 1.15)  # Match right plot limit

# Derecha: wowa
plot_criteria(ax2, prop_wowa, is_wowa=True)
ax2.set_title(r"WOWA ($w_i$ from variable importance partition)", fontsize=24, pad=15)
ax2.set_xlabel("Accumulated Importance", fontsize=18)
ax2.set_xlim(-0.02, 1.15)  # Límite cambiado a 1.15

# Ajustes comunes
for ax in [ax1, ax2]:
    ax.set_ylim(-0.02, 1.15)  # Aumentado para dejar sitio a los porcentajes arriba
    ax.grid(
        False
    )  # Quitamos grid por completo para limpiar las líneas moradas del fondo

# --- Leyenda Personalizada ---
legend_elements = [
    Line2D([0], [0], color="gray", lw=3, alpha=0.5, label=r"Attitude $Q(x)=x^2$"),
    Patch(
        facecolor=purple_shades[2],
        edgecolor="none",
        alpha=0.6,
        label="Importance (Width %)",
    ),
]

ax1.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.85),
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    fontsize=18,
)  # Increased fontsize

plt.suptitle(
    r"Weight Generation via Quantifier: OWA vs WOWA", fontsize=26, y=0.98
)  # Increased fontsize
plt.tight_layout()
plt.show()
