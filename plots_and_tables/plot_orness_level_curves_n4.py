import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations


def plot_orness_level_curves_n4():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # --- 1. DEFINICIÓN DE COORDENADAS PARA LA PERSPECTIVA DESEADA ---
    # Objetivo:
    # - Arriba (Z): w=[1,0,0,0] (Orness 1.0)
    # - Derecha (X+): w=[0,0,0,1] (Orness 0.0)
    # - Frente (Y-): w=[0,0,1,0] (Orness 0.33)
    # - Izquierda/Fondo: w=[0,1,0,0] (Orness 0.67) - (Calculado geométricamente)

    # Coordenadas (X, Y, Z) construyendo un tetraedro regular
    # Base en el plano XY, Vértice Top en Z

    # Vértice TOP
    v_top = np.array([0.0, 0.0, 1.5])

    # Vértices de la BASE (Triángulo equilátero centrado en 0,0)
    # Frente (Y negativo)
    v_front = np.array([0.0, -1.0, 0.0])

    # Derecha (X positivo, Y positivo)
    v_right = np.array([0.866, 0.5, 0.0])  # cos(30), sin(30)

    # Izquierda (X negativo, Y positivo)
    v_left = np.array([-0.866, 0.5, 0.0])

    vertices_coords = np.array([v_top, v_right, v_front, v_left])

    # Pesos correspondientes según tu petición
    # v_top   -> w=[1,0,0,0] (Orness 1.0)
    # v_right -> w=[0,0,0,1] (Orness 0.0) -> "A la derecha"
    # v_front -> w=[0,0,1,0] (Orness 0.33) -> "Al frente"
    # v_left  -> w=[0,1,0,0] (Orness 0.67)

    # Valores de Orness para cálculo
    vertex_orness = [1.0, 0.0, 1 / 3, 2 / 3]

    # Etiquetas de texto para los vértices
    labels_text = [
        "w=[1, 0, 0, 0]\n(Orness 1.0)",  # Top
        "w=[0, 0, 0, 1]\n(Orness 0.0)",  # Right
        "w=[0, 0, 1, 0]\n(Orness 0.33)",  # Front
        "w=[0, 1, 0, 0]\n(Orness 0.67)",  # Left
    ]

    # --- 2. DIBUJAR EL TETRAEDRO (WIREFRAME) ---
    vertex_indices = [0, 1, 2, 3]
    edges = list(combinations(vertex_indices, 2))

    # Dibujar aristas
    for e in edges:
        p1 = vertices_coords[e[0]]
        p2 = vertices_coords[e[1]]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="black",
            linewidth=1.5,
            alpha=0.7,
        )

    # --- 3. CALCULAR Y PINTAR CURVAS DE NIVEL (SLICES) ---
    # Niveles de orness a visualizar
    levels = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    cmap = plt.cm.viridis  # Mapa de colores

    for level in levels:
        points = []
        # Buscar intersecciones en las aristas
        for e in edges:
            idx_a, idx_b = e[0], e[1]
            orn_a, orn_b = vertex_orness[idx_a], vertex_orness[idx_b]

            # Tolerancia para comparaciones float
            if min(orn_a, orn_b) <= level <= max(orn_a, orn_b):
                if abs(orn_a - orn_b) > 1e-5:
                    t = (level - orn_a) / (orn_b - orn_a)
                    pt = vertices_coords[idx_a] + t * (
                        vertices_coords[idx_b] - vertices_coords[idx_a]
                    )
                    points.append(pt)

        points = np.array(points)
        if len(points) >= 3:
            # Ordenar puntos angularmente para formar polígono correcto
            center = np.mean(points, axis=0)
            vecs = points - center
            # Ordenamos por ángulo en el plano XY (suficiente para estas rodajas)
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            sort_order = np.argsort(angles)
            points = points[sort_order]

            # Crear polígono
            poly = Poly3DCollection([points], alpha=0.5)
            poly.set_facecolor(cmap(level))  # Color basado en el nivel
            poly.set_edgecolor("white")
            poly.set_linewidth(0.5)
            ax.add_collection3d(poly)

    # --- 4. ETIQUETAS Y ESTILO ---

    # Dibujar vértices (puntos discretos)
    ax.scatter(
        vertices_coords[:, 0],
        vertices_coords[:, 1],
        vertices_coords[:, 2],
        color="black",
        s=40,
    )

    # Colocar texto con "Offset" (desplazamiento) para que no pise la figura
    offsets = [
        (0, 0, 0.15),  # Top: Subir Z
        (0.2, 0.1, 0),  # Right: Mover a la derecha X
        (0, -0.2, -0.1),  # Front: Mover hacia el frente Y (y un poco abajo)
        (-0.2, 0.1, 0),  # Left: Mover a la izquierda X
    ]

    alignments = [
        ("center", "bottom"),  # Top
        ("left", "center"),  # Right
        ("center", "top"),  # Front
        ("right", "center"),  # Left
    ]

    for i, txt in enumerate(labels_text):
        x, y, z = vertices_coords[i]
        off_x, off_y, off_z = offsets[i]
        ha, va = alignments[i]

        ax.text(
            x + off_x,
            y + off_y,
            z + off_z,
            txt,
            fontsize=11,  # Letra ligeramente más grande
            fontweight="bold",
            color="#333333",
            horizontalalignment=ha,
            verticalalignment=va,
        )

    # Configuración de la cámara
    ax.set_axis_off()  # Quitar caja y ejes

    # Ajuste de vista:
    # Elev=20 para ver volumen
    # Azim=-90 pone el eje Y negativo (donde está nuestro vértice Frontal) mirando al espectador
    ax.view_init(elev=20, azim=-90)

    # Ajustar límites para que quepa el texto desplazado
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_orness_level_curves_n4()
