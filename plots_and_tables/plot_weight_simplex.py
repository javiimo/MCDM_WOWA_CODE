import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_weight_simplex():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Definir vértices del simplex
    vertices = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # 2. Definir la cara (el triángulo)
    faces = [vertices]

    # 3. Dibujar la superficie del triángulo
    poly3d = Poly3DCollection(faces, alpha=0.6, linewidths=1, edgecolors='k')
    poly3d.set_facecolor('orange')
    ax.add_collection3d(poly3d)

    # 4. Resaltar los vértices (Rojos)
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 
               color='red', s=50, label='Vertices')

    # 5. Dibujar los ejes manualmente (líneas discontinuas desde el origen)
    # Esto da contexto sin cerrar la figura como si fuera una pirámide sólida
    length = 1.2
    ax.plot([0, length], [0, 0], [0, 0], color='black', linestyle='--', linewidth=0.8)
    ax.plot([0, 0], [0, length], [0, 0], color='black', linestyle='--', linewidth=0.8)
    ax.plot([0, 0], [0, 0], [0, length], color='black', linestyle='--', linewidth=0.8)

    # Marcar el Origen claramente en primer plano
    ax.scatter([0], [0], [0], color='black', s=60, marker='o', label='Origin')

    # 6. Etiquetas
    offset = 0.05
    for v in vertices:
        ax.text(v[0], v[1], v[2]+offset, f"({v[0]},{v[1]},{v[2]})", color='black', fontsize=9)
    
    ax.text(0, 0, -0.1, "Origin", color='black', weight='bold')

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_zlabel('$w_3$')
    ax.set_title('Weight Simplex\n$w_i \geq 0, \sum w_i = 1$')
    
    # Límites de los ejes
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_zlim(0, 1.2)
    
    # --- CONFIGURACIÓN DE CÁMARA ---
    # Esta es la vista exacta de tu imagen
    ax.view_init(elev=25, azim=-115)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_weight_simplex()