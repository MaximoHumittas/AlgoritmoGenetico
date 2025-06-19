import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import json

# Cargar datos guardados por el algoritmo
with open("./Anexos/CMGO.json", "r", encoding="utf-8") as f:
    data = json.load(f)
mejor_coalicion = set(data["mejor_coalicion"])
votes = data["votes"]

# 1. Prepara las coordenadas de los miembros de la coalición
coords_cgm = np.array([[v["x"], v["y"]] for i, v in enumerate(votes) if i in mejor_coalicion])

# 2. Calcula el polígono convexo (Convex Hull)
hull = ConvexHull(coords_cgm)
hull_pts = coords_cgm[hull.vertices]

# 3. Calcula el centroide de la CGM
centroide = coords_cgm.mean(axis=0)

# 4. Calcula el radio máximo desde el centroide a los puntos del polígono convexo
dists = np.linalg.norm(hull_pts - centroide, axis=1)
r = dists.max()
alpha = 1  # Usar alpha=1 para el círculo "tal cual"; puedes aumentarlo si quieres abarcar más

# 5. Graficar todo
fig, ax = plt.subplots(figsize=(8, 8))

# --- Puntos del congreso, coloreados por partido ---
for i, v in enumerate(votes):
    color = 'blue' if v["party_short_name"] == "Democrat" else 'red'
    marker = 'o' if i in mejor_coalicion else 'x'
    ax.scatter(v["x"], v["y"], c=color, marker=marker, alpha=0.7, label=None)

# --- Polígono convexo (zona morada con transparencia) ---
from matplotlib.patches import Polygon
polygon = Polygon(hull_pts, closed=True, facecolor='purple', alpha=0.2, edgecolor='purple', lw=2)
ax.add_patch(polygon)

# --- Círculo negro gigante ---
circle = plt.Circle(centroide, r*alpha, fill=False, color='black', lw=2)
ax.add_patch(circle)

# --- Centroide ---
ax.scatter(*centroide, c='yellow', marker='*', s=200, label='Centroide')

# --- Opcional: leyenda personalizada ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Demócrata (coalición)', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Republicano (coalición)', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='x', color='blue', label='Demócrata (resto)', markersize=10),
    Line2D([0], [0], marker='x', color='red', label='Republicano (resto)', markersize=10),
    Line2D([0], [0], marker='*', color='yellow', label='Centroide', markersize=15),
    Line2D([0], [0], color='purple', lw=4, label='Polígono convexo (CGM)'),
    Line2D([0], [0], color='black', lw=2, label='Área búsqueda local (círculo)'),
]
ax.legend(handles=legend_elements, loc='upper left')

ax.set_title("Espacio ideológico: CGM, polígono convexo y radio de búsqueda local")
ax.set_xlabel("DW-NOMINATE X")
ax.set_ylabel("DW-NOMINATE Y")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
