import json
import numpy as np
import random
from time import time
from numpy.linalg import norm
from statistics import mean, stdev
from concurrent.futures import ProcessPoolExecutor, as_completed

# =================== CONFIGURACIÓN ===================
TAM_POBLACION = 38
TAM_COALICION = 216
GENERACIONES_MAX = 15000
TORNEO_K = 5
PROB_CRUCE = 0.8
PROB_MUTACION = 0.1700019
ELITISMO = True
ESTANCAMIENTO_MAX = 200
NUM_RUNS = 10

# ==================== LECTURA DE DATOS ====================
filename = "./Anexos/datos.json"
with open(filename, 'r', encoding='utf-16') as f:
    data = json.load(f)

votes = data['rollcalls'][0]['votes']
N = len(votes)
COORDS = np.array([(v['x'], v['y']) for v in votes], dtype=float)

# Matriz de distancias euclidianas
diff_x = COORDS[:, 0][:, None] - COORDS[:, 0]
diff_y = COORDS[:, 1][:, None] - COORDS[:, 1]
dist_matrix = np.sqrt(diff_x**2 + diff_y**2)
todos_indices = list(range(N))

FITNESS_GOAL = 9686.93831

# =================== GUARDADO DE RESULTADOS ===================
def guardar_resultados(rdol, ruta="./Anexos/CMGO.json"):
    try:
        with open(ruta, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data = {}
    if "datos" not in data or not isinstance(data["datos"], list):
        data["datos"] = []
    data["datos"].append(rdol)
    with open(ruta, "w") as f:
        json.dump(data, f, indent=4)
    print("Resultados guardados correctamente.")

# =================== FUNCIÓN FITNESS ===================
def calcular_fitness(indices_coalicion):
    idx = np.array(indices_coalicion, dtype=int)
    submat = dist_matrix[np.ix_(idx, idx)]
    total = submat.sum() / 2.0
    return total

# =================== POBLACIÓN INICIAL ===================
def generar_poblacion_inicial():
    return [random.sample(todos_indices, TAM_COALICION) for _ in range(TAM_POBLACION)]

# =================== SELECCIÓN POR TORNEO ===================
def seleccionar_padre_torneo(aptitudes, k=TORNEO_K):
    participantes = random.sample(range(len(aptitudes)), k)
    ganador = min(participantes, key=lambda i: aptitudes[i])
    return ganador

# =================== CRUZA ===================
def cruzar(parent1, parent2):
    set_hijo = set()
    num_tomar = TAM_COALICION // 2
    seleccionados_p1 = random.sample(parent1, num_tomar)
    set_hijo.update(seleccionados_p1)
    for gen in parent2:
        if len(set_hijo) >= TAM_COALICION:
            break
        set_hijo.add(gen)
    if len(set_hijo) < TAM_COALICION:
        faltantes = TAM_COALICION - len(set_hijo)
        disponibles = [idx for idx in range(N) if idx not in set_hijo]
        set_hijo.update(random.sample(disponibles, faltantes))
    return list(set_hijo)

# =================== MUTACIÓN ===================
def mutar(individuo):
    if random.random() < PROB_MUTACION:
        coalicion_set = set(individuo)
        dentro = random.choice(individuo)
        fuera = random.choice([idx for idx in range(N) if idx not in coalicion_set])
        nuevo = individuo.copy()
        nuevo.remove(dentro)
        nuevo.append(fuera)
        return nuevo
    return individuo

# =================== ALGORITMO GENÉTICO (UNA RUN) ===================
def algoritmo_genetico(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    poblacion = generar_poblacion_inicial()
    aptitudes = [calcular_fitness(ind) for ind in poblacion]
    mejor_aptitud = min(aptitudes)
    mejor_coalicion = poblacion[aptitudes.index(mejor_aptitud)]
    historial_fitness = [mejor_aptitud]

    generacion, sin_mejora = 0, 0
    t0 = time()
    total_iteraciones = 0

    while generacion < GENERACIONES_MAX and sin_mejora < ESTANCAMIENTO_MAX:
        generacion += 1
        nueva_poblacion = []
        if ELITISMO:
            nueva_poblacion.append(mejor_coalicion.copy())
        while len(nueva_poblacion) < TAM_POBLACION:
            idx_p1 = seleccionar_padre_torneo(aptitudes)
            idx_p2 = seleccionar_padre_torneo(aptitudes)
            padre1, padre2 = poblacion[idx_p1], poblacion[idx_p2]
            hijo = cruzar(padre1, padre2) if random.random() < PROB_CRUCE else (padre1 if aptitudes[idx_p1] <= aptitudes[idx_p2] else padre2)
            hijo = mutar(hijo)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion
        aptitudes = [calcular_fitness(ind) for ind in poblacion]
        total_iteraciones += TAM_POBLACION
        gen_mejor_aptitud = min(aptitudes)
        gen_mejor_coalicion = poblacion[aptitudes.index(gen_mejor_aptitud)]
        historial_fitness.append(gen_mejor_aptitud)
        if gen_mejor_aptitud < mejor_aptitud:
            mejor_aptitud = gen_mejor_aptitud
            mejor_coalicion = gen_mejor_coalicion.copy()
            sin_mejora = 0
        else:
            sin_mejora += 1
    tiempo = time() - t0
    return mejor_aptitud, tiempo, total_iteraciones, mejor_coalicion, historial_fitness

# =================== MAIN ===================
def main():
    fitness_list, tiempos, iteraciones, coaliciones, historiales = [], [], [], [], []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(algoritmo_genetico, i) for i in range(NUM_RUNS)]
        for f in as_completed(futures):
            fit, t, iters, coal, hist = f.result()
            fitness_list.append(fit)
            tiempos.append(t)
            iteraciones.append(iters)
            coaliciones.append(coal)
            historiales.append(hist)

    # Estadísticas generales
    fitness_arr = np.array(fitness_list)
    precision_arr = (FITNESS_GOAL / fitness_arr) * 100

    print("\n\n===== RESUMEN =====")
    print(f"Resultado esperado: {FITNESS_GOAL}")
    print(f"Fitness promedio: {fitness_arr.mean():.5f}")
    print(f"Fitness mínimo (mejor run): {fitness_arr.min():.5f}")
    print(f"Desviación estándar fitness: {fitness_arr.std():.5f}")
    print(f"Precisión promedio: {precision_arr.mean():.2f} %")
    print(f"Desviación estándar precisión: {precision_arr.std():.2f} %")
    print(f"Promedio de iteraciones: {np.mean(iteraciones):.2f}")
    print(f"Desviación estándar iteraciones: {np.std(iteraciones):.2f}")
    print(f"Promedio tiempo (s): {np.mean(tiempos):.2f}")
    print(f"Desviación estándar tiempo: {np.std(tiempos):.2f}")

    # Guardar la mejor coalición
    mejor_run_idx = int(np.argmin(fitness_list))
    mejor_coalicion = coaliciones[mejor_run_idx]
    resultado = {
        "mejor_coalicion": mejor_coalicion,
        "votes": votes
    }
    with open("./Anexos/CMGO.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
