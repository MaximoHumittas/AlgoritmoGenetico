import random as ra, math as ma, time as ti, requests as req, pandas as pd, numpy as np, statistics as st, matplotlib.pyplot as plt, scipy.spatial as sp

def download_file(url, local_filename):
    with req.get(url, stream=True) as r:
        r.raise_for_status()
        with open(f'.\Anexos\{local_filename}', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return f

members = pd.read_csv(".\Anexos\HSall_members.csv", encoding="utf-8")

votes = pd.read_csv(".\Anexos\RH0941234.csv", encoding="utf-8")

# cast_code == 0  → no era miembro aún
# cast_code == 1  → votó a favor
# cast_code == 2  → votó en contra
# cast_code == 3  → se abstuvo
# cast_code == 9  → estaba ausente / no votó

# Nos quedamos sólo con quienes realmente participaron
votacion_filtrada = votes[~votes["cast_code"].isin([0, 9])]
icpsr_presentes = votacion_filtrada["icpsr"].unique()

# El congreso 94 corresponde al período 1975-1976
# Filtrando datos de los miembros del Congreso 94

CONGRESS   = 94          # El prefijo RH094… indica 94.º Congreso
ROLLNUMBER = 1234        # Número de roll-call dentro del Congreso

members = members[members["icpsr"].isin(icpsr_presentes)]

members = members[members["congress"] == CONGRESS]

votacion_filtrada = votacion_filtrada[votacion_filtrada["icpsr"].isin(members["icpsr"])]

votacion_filtrada = votacion_filtrada[["icpsr", "cast_code"]]

combinado = pd.merge(votacion_filtrada, members, on="icpsr", how="inner")

coords = list(zip(combinado["nominate_dim1"], combinado["nominate_dim2"]))

print(coords)  # Muestra las primeras 5 coordenadas para verificar
print(f"Total de representantes: {len(coords)}")

# ALGORITMO GENETICO PARA LA COALICIÓN DE REPRESENTANTES
# Este código implementa un algoritmo genético para encontrar una coalición de representantes
# que minimice la distancia euclidiana entre sus miembros, respetando la restricción de tamaño de la coalición.
# El objetivo es seleccionar 216 representantes de un total de 431, de modo que la suma de las distancias
# euclidianas entre todos los pares de miembros de la coalición sea mínima.
# Este código sigue los pasos del algoritmo genético:
# 1. Generar una población inicial aleatoria de cromosomas (soluciones candidatas).
# 2. Evaluar cada cromosoma calculando su fitness (suma de distancias).
# 3. Seleccionar padres mediante un método de selección (ruleta por ranking).
# 4. Realizar cruza de un punto para generar nuevos hijos.
# 5. Mutar los hijos con una probabilidad dada (intercambio de bits).
# 6. Verificar que los hijos cumplen la restricción de tamaño de la coalición.
# 7. Reemplazar la población con los nuevos hijos y repetir hasta alcanzar un criterio de parada.

N_REPRESENTANTES = len(coords)  # número total de representantes
Q = N_REPRESENTANTES//2 + 1     # tamaño de la coalición (mayoría absoluta)
TAM_POBLACION = 38              # tamaño de la población (número de cromosomas)
PROB_MUTACION = 0.17            # probabilidad de mutación por cromosoma (aprox 17%)
P_SELECCION = 0.141             # parámetro de selección para ruleta por ranking

# Supongamos que 'coords' es una lista de tamaño 431 donde cada elemento coords[i] = (X_i, Y_i)
# representando las coordenadas DW-Nominate del representante i-ésimo en la votación.
# Estas coordenadas deben ser cargadas antes de ejecutar el GA (p.ej., desde un archivo de datos).
# coords = [...]  # Lista de tuplas (x,y) de longitud 431

def generar_poblacion_inicial(n, q, tam_pob):
    poblacion = []
    for _ in range(tam_pob):
        indices_unos = ra.sample(range(n), q)
        cromosoma = [0] * n
        for idx in indices_unos:
            cromosoma[idx] = 1
        poblacion.append(cromosoma)
    return poblacion

def calcular_fitness(cromosoma, coords):
    indices = [i for i, bit in enumerate(cromosoma) if bit == 1]
    total = 0.0
    for i in range(len(indices) - 1):
        for j in range(i + 1, len(indices)):
            a = indices[i]
            b = indices[j]
            dx = coords[a][0] - coords[b][0]
            dy = coords[a][1] - coords[b][1]
            dist = ma.sqrt(dx*dx + dy*dy)
            total += dist
    return total

def seleccionar_padres(poblacion, fitness_vals, P=P_SELECCION):
    indices_ordenados = sorted(range(len(poblacion)), key=lambda i: fitness_vals[i])
    poblacion_ordenada = [poblacion[i] for i in indices_ordenados]
    pesos = []
    total_peso = 0.0
    for rank in range(len(poblacion_ordenada)):
        peso = P * ((1 - P) ** rank)
        pesos.append(peso)
        total_peso += peso
    pesos = [w/total_peso for w in pesos]
    acum = []
    acumulado = 0.0
    for w in pesos:
        acumulado += w
        acum.append(acumulado)

    def elegir_individuo():
        r = ra.random()
        for idx, valor_acum in enumerate(acum):
            if valor_acum >= r:
                return poblacion_ordenada[idx]
        return poblacion_ordenada[-1]
    padre1 = elegir_individuo()
    padre2 = elegir_individuo()
    while padre2 is padre1:
        padre2 = elegir_individuo()
    return padre1, padre2

def cruzar(padre1, padre2):
    n = len(padre1)
    punto_corte = ra.randint(1, n-1)
    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
    return hijo1, hijo2

def mutar(cromosoma, prob_mut=PROB_MUTACION):
    if ra.random() < prob_mut:
        indices_uno = [i for i, bit in enumerate(cromosoma) if bit == 1]
        indices_cero = [i for i, bit in enumerate(cromosoma) if bit == 0]

        # Verificación robusta
        if len(indices_uno) > 0 and len(indices_cero) > 0:
            idx1 = ra.choice(indices_uno)
            idx0 = ra.choice(indices_cero)
            cromosoma[idx1], cromosoma[idx0] = 0, 1
        else:
            # Fuerza una nueva mutación válida al regenerar uno de los grupos
            # (esto rara vez debería pasar si las restricciones se cumplen)
            cromosoma = verificar_restriccion(cromosoma)
    return cromosoma

def verificar_restriccion(cromosoma, q=Q):
    ones = sum(cromosoma)
    if ones > q:
        indices_uno = [i for i, bit in enumerate(cromosoma) if bit == 1]
        excedentes = ones - q
        apagar = ra.sample(indices_uno, excedentes)
        for idx in apagar:
            cromosoma[idx] = 0
    elif ones < q:
        indices_cero = [i for i, bit in enumerate(cromosoma) if bit == 0]
        faltantes = q - ones
        encender = ra.sample(indices_cero, faltantes)
        for idx in encender:
            cromosoma[idx] = 1
    return cromosoma

def algoritmo_genetico(coords, max_iter=10000):
    poblacion = generar_poblacion_inicial(N_REPRESENTANTES, Q, TAM_POBLACION)

    fitness_vals = [calcular_fitness(crom, coords) for crom in poblacion]

    best_fit = min(fitness_vals)

    best_solution = poblacion[fitness_vals.index(best_fit)]
    iteraciones = 0

    historial_fitness = [best_fit]

    print("Mejor fitness inicial:", best_fit)

    while iteraciones < max_iter:
        iteraciones += 1
        nueva_poblacion = []
        while len(nueva_poblacion) < N_REPRESENTANTES:
            padre1, padre2 = seleccionar_padres(poblacion, fitness_vals)

            hijo1, hijo2 = cruzar(padre1, padre2)

            hijo1 = mutar(hijo1, PROB_MUTACION)
            hijo2 = mutar(hijo2, PROB_MUTACION)

            hijo1 = verificar_restriccion(hijo1, Q)
            hijo2 = verificar_restriccion(hijo2, Q)

            nueva_poblacion.append(hijo1)

            if len(nueva_poblacion) < N_REPRESENTANTES:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

        fitness_vals = [calcular_fitness(crom, coords) for crom in poblacion]

        current_best = min(fitness_vals)

        historial_fitness.append(current_best)

        if current_best < best_fit:
            best_fit = current_best
            best_solution = poblacion[fitness_vals.index(current_best)]
    
        if best_fit <= 9686.93831:
            break

    return best_solution, best_fit, iteraciones, historial_fitness

best_solution, best_fit, iteraciones, historial_fitness = algoritmo_genetico(coords)
print("Mejor fitness:", best_fit)

N_RUNS = 3
FITNESS_GOAL = 9686.93831

fitness_list = []
iters_list = []
time_list = []
fitness_historial_total = []

for run in range(N_RUNS):
    t0 = ti.time()
    best_solution, best_fit, iteraciones, historial_fitness = algoritmo_genetico(coords)
    t1 = ti.time()

    fitness_list.append(best_fit)
    iters_list.append(iteraciones)
    time_list.append(t1 - t0)
    fitness_historial_total.append(historial_fitness)


# Estadisticas

fitness_mean = st.mean(fitness_list)
fitness_std = st.stdev(fitness_list)

iters_mean = st.mean(iters_list)
iters_std = st.stdev(iters_list)

time_mean = st.mean(time_list)
time_std = st.stdev(time_list)

fitness_historial_total = [item for sublist in fitness_historial_total for item in sublist]
fitness_historial_mean = st.mean(fitness_historial_total)

mean_fitness_historial = (1- abs(FITNESS_GOAL - fitness_historial_mean) / FITNESS_GOAL) * 100

print("Historial de fitness promedio:", fitness_historial_mean)

print("\n" + "="*30 + "\nREPORTE DE RESULTADOS\n" + "="*30)

print(f"Mejor fitness encontrado: {best_fit:.5f}")
print(f"Mejor fitness promedio: {fitness_mean:.5f}")

print(f"Resultado esperado: {FITNESS_GOAL:.5f}")
print(f"Precisión: {mean_fitness_historial:.2f}%")
print(f"Desviación estándar del fitness: {fitness_std:.5f}")
print(f"Promedio iteraciones: {iters_mean:.2f}")
print(f"Desviación estándar de iteraciones: {iters_std:.2f}")
print(f"Tiempo promedio (s): {time_mean:.4f}")
print(f"Desviación estándar del tiempo: {time_std:.4f}")

print("="*30 + "\nFIN DEL REPORTE\n" + "="*30)




# Visualización de los resultados

def partido_a_color(party_code):

    return 'blue' if party_code == 100 else 'red'  #

colores = combinado["party_code"].apply(partido_a_color).values

combinado["CGM"] = best_solution 

x = combinado["nominate_dim1"].values
y = combinado["nominate_dim2"].values

es_CGM = combinado["CGM"] == 1
no_CGM = ~es_CGM

plt.figure(figsize=(8,8))
plt.style.use('dark_background')
plt.scatter(x[no_CGM], y[no_CGM], c=colores[no_CGM], marker='x', label='No pertenece')
plt.scatter(x[es_CGM], y[es_CGM], c=colores[es_CGM], marker='o', edgecolor='k', s=40, label='Pertenece')

coords_CGM = np.column_stack((x[es_CGM], y[es_CGM]))
hull = sp.ConvexHull(coords_CGM)
for simplex in hull.simplices:
    plt.plot(coords_CGM[simplex, 0], coords_CGM[simplex, 1], 'm-')

plt.fill(coords_CGM[hull.vertices,0], coords_CGM[hull.vertices,1], 'm', alpha=0.15)