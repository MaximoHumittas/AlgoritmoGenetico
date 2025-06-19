import json
import matplotlib.pyplot as plt

# Cargar datos
with open("./Anexos/datos.json", encoding="utf-16") as f:
    data = json.load(f)

# Accede a la lista de votos de la primera (o única) rollcall
votes = data["rollcalls"][0]["votes"]

# Listas para las coordenadas
x_dem, y_dem = [], []
x_rep, y_rep = [], []

# Separar por partido
for v in votes:
    if v["party_code"] == 100:  # Demócrata
        x_dem.append(v["x"])
        y_dem.append(v["y"])
    elif v["party_code"] == 200:  # Republicano
        x_rep.append(v["x"])
        y_rep.append(v["y"])

# Graficar
plt.figure(figsize=(10, 8))
plt.style.use("ggplot")
plt.scatter(x_dem, y_dem, c="blue", label="Demócrata", alpha=0.8)
plt.scatter(x_rep, y_rep, c="red", label="Republicano", alpha=0.8)

plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.title("Representantes por partido")
plt.legend(title="Partido")
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(True)
plt.show()
