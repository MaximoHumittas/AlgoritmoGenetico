
import json
from time import time
from numpy.linalg import norm
from pandas import read_csv,merge
from statistics import mean,stdev
from numpy.random import choice as np_choice
from random import random,randint,sample,choice
from numpy import array, arange,where, flatnonzero, argsort, cumsum, searchsorted, isclose

# ------------------------------------------------
# ---------- Guardado de la CMGO (datos) ---------
# ------------------------------------------------
def guardar_resultados(rdol):
    dirr = "./Anexos/CMGO.json"

    # Leer y cargar datos existentes
    with open(dirr, "r") as f:
        data = json.load(f)

    # Asegurar estructura esperada
    if "datos" not in data or not isinstance(data["datos"], list):
        data["datos"] = []

    # Agregar nuevos resultados
    data["datos"].append(rdol)

    # Reescribir el archivo con los nuevos datos
    with open(dirr, "w") as f:
        json.dump(data, f, indent=4)

    print("Resultados guardados correctamente.")


# ------------------------------------------------
# ---------- Configuración del Programa ----------
# ------------------------------------------------
CONGRESS = 94; ROLLNUMBER = 1234

def procesemiento_de_los_datos():
    members = read_csv("./Anexos/HSall_members.csv", encoding="utf-8")
    votes   = read_csv("./Anexos/RH0941234.csv", encoding="utf-8")
    """
    * cast_code == 0  → no era miembro aún
    * cast_code == 1  → votó a favor
    * cast_code == 2  → votó en contra
    * cast_code == 3  → se abstuvo
    * cast_code == 9  → estaba ausente / no votó
    """
    # solo votos validos (1,2,3)
    votacion_filtrada = votes[~votes["cast_code"].isin([0, 9])]
    # identificador unico de las personas
    icpsr_presentes   = votacion_filtrada["icpsr"].unique()
    members = members[members["icpsr"].isin(icpsr_presentes)]
    # solo los que perteneceb al congreso 94
    members = members[members["congress"] == CONGRESS]
    # solo los votos de los miembro del congreso 94
    votacion_filtrada = votacion_filtrada[votacion_filtrada["icpsr"].isin(members["icpsr"])]
    # reduce el DataFrame a las columnas ("icpsr", "cast_code")
    votacion_filtrada = votacion_filtrada[["icpsr", "cast_code"]]
    # une ambos DataFrames y solo se conservan las columnas que tengan en commun
    combinado = merge(votacion_filtrada, members, on="icpsr", how="inner")
    # cordenadas (x,y) de los congresistas
    coords = list(zip(combinado["nominate_dim1"], combinado["nominate_dim2"]))
    return coords

# cordenadas de los congresistas
COORDS  = procesemiento_de_los_datos()
# número total de representantes: 376
NUM_REPRESENTANTES = len(COORDS)
# tamaño de la coalición (mayoría absoluta)
Q = (NUM_REPRESENTANTES //2) + 1
# tamaño de la población (número de cromosomas)
TAM_POBLACION = 38
# probabilidad de mutación por cromosoma (aprox 17%)
PROB_MUTACION = 0.17
# parámetro de selección para ruleta por ranking
PRM_SELECCION = 0.141
# fitness optimo
FITNESS_GOAL = 9686.93831
# varianza permitida para el fitness optimo
VARIANZA = 50

# ------------------------------------------------
# ---------- Población Inicial -------------------
# ------------------------------------------------
def generar_poblacion_inicial():
    poblacion = []
    for _ in range(TAM_POBLACION):
        # 38 valores del cromosoma que serán activados
        indices_unos = sample(range(NUM_REPRESENTANTES), Q)
        # cromosoma con 376 valores inactivos
        cromosoma = [0] * NUM_REPRESENTANTES
        for idx in indices_unos:
            # activa los 38 cromosomas previamente seleccionados
            cromosoma[idx] = 1
        # agrega el cromosoma a la población inicial
        poblacion.append(cromosoma)
    return poblacion

# ------------------------------------------------
# ---------- Calculo del Fitness -----------------
# ------------------------------------------------
def calcular_fitness(cromosoma):
    # encuentra los indices donde el valor sea distinto de cero
    indices = flatnonzero(cromosoma)
    # guarda las tuplas (xi,yi) de los indices anteriores
    puntos = array(COORDS)[indices]
    # valor inicial del fitness
    total = 0.0
    for i in range(len(puntos) - 1):
        pi = puntos[i]
        # Solo comparamos contra puntos siguientes (j > i)
        distancias = norm(puntos[i+1:] - pi, axis=1)
        # suma todas las distancias obtenidas
        total += distancias.sum()
    return total

# ------------------------------------------------
# ---------- Selección de los Padres -------------
# ------------------------------------------------
def seleccionar_padres(poblacion, fitness_vals):
    # Ordenar índices de menor a mayor fitness (mejores primeros)
    indices_ordenados = argsort(fitness_vals)
    poblacion_ordenada = [poblacion[i] for i in indices_ordenados]

    # Calcular pesos por ranking exponencial: P * (1 - P)^rango
    ranks = arange(len(poblacion_ordenada))
    pesos = PRM_SELECCION * ((1 - PRM_SELECCION) ** ranks)
    pesos /= pesos.sum()  # normalizar

    # Acumulado de probabilidades para ruleta
    acumulados = cumsum(pesos)

    def elegir_individuo():
        r = random()
        return poblacion_ordenada[searchsorted(acumulados, r, side="right")]

    # Selecciona dos padres distintos
    padre1 = elegir_individuo()
    padre2 = elegir_individuo()
    while padre2 is padre1:
        padre2 = elegir_individuo()
    return padre1, padre2

# ------------------------------------------------
# ---------- Cruze de los Cromosomas -------------
# ------------------------------------------------
def cruzar(padre1, padre2):
    n = len(padre1)
    punto_corte = randint(1, n-1)
    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
    return hijo1, hijo2

# ------------------------------------------------
# ---------- Mutación de los Cromosomas ----------
# ------------------------------------------------
def mutar(cromosoma):
    if random() < PROB_MUTACION:
        # se guardan los indices donde el valor sea (0,1) según corresponda
        indices_uno  = [i for i, bit in enumerate(cromosoma) if bit == 1]
        indices_cero = [i for i, bit in enumerate(cromosoma) if bit == 0]

        if len(indices_uno) > 0 and len(indices_cero) > 0:
            # selecciona aleatoriamente un valor de las listas
            idx1 = choice(indices_uno)
            idx0 = choice(indices_cero)
            # intercambia los valores
            cromosoma[idx1], cromosoma[idx0] = 0, 1
        # en caso que no se pueda mutar se recontruye el comosoma
        else:
            cromosoma = verificar_restriccion(cromosoma)
    return cromosoma

def verificar_restriccion(cromosoma):
    cromosoma = array(cromosoma)
    ones_count = sum(cromosoma)

    if ones_count > Q:
        indices_uno = where(cromosoma == 1)[0]
        apagar = np_choice(indices_uno, size=ones_count - Q, replace=False)
        cromosoma[apagar] = 0

    elif ones_count < Q:
        indices_cero = where(cromosoma == 0)[0]
        encender = np_choice(indices_cero, size=Q - ones_count, replace=False)
        cromosoma[encender] = 1

    return cromosoma.tolist()

# ------------------------------------------------
# ---------- Algoritmo Genetico ------------------
# ------------------------------------------------
def algoritmo_genetico():
    # población inicial aleatoria
    poblacion = generar_poblacion_inicial()
    #
    fitness_vals = [calcular_fitness(crom) for crom in poblacion]
    #
    best_fit = min(fitness_vals)
    #
    best_solution = poblacion[fitness_vals.index(best_fit)]
    #
    historial_fitness = [best_fit]

    print("Mejor fitness inicial:", best_fit)

    iters = 0
    max_iter = 100
    while iters < max_iter:
        iters += 1
        nueva_poblacion = []
        while len(nueva_poblacion) < NUM_REPRESENTANTES:
            padre1, padre2 = seleccionar_padres(poblacion, fitness_vals)

            hijo1, hijo2 = cruzar(padre1, padre2)

            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)

            hijo1 = verificar_restriccion(hijo1)
            hijo2 = verificar_restriccion(hijo2)

            nueva_poblacion.append(hijo1)

            if len(nueva_poblacion) < NUM_REPRESENTANTES:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

        fitness_vals = [calcular_fitness(crom) for crom in poblacion]

        current_best = min(fitness_vals)

        historial_fitness.append(current_best)

        if current_best < best_fit:
            best_fit = current_best
            best_solution = poblacion[fitness_vals.index(current_best)]

        if (abs(FITNESS_GOAL - current_best) <= VARIANZA):
            break

    return (best_solution,best_fit,iters,historial_fitness)

# ------------------------------------------------
# ---------- Ejecución Principal (main) ----------
# ------------------------------------------------
def main():
    num_runs = 1
    fitness_list = []
    iters_list = []
    time_list = []
    fitness_historial_total = []

    for run in range(num_runs):
        t0 = time()
        best_solution, best_fit, iteraciones, historial_fitness = algoritmo_genetico()
        t1 = time()

        fitness_list.append(best_fit)
        iters_list.append(iteraciones)
        time_list.append(t1 - t0)
        fitness_historial_total.append(historial_fitness)

        rdol = {"best_fit":best_fit, "cant_iter":iteraciones, 
                "exe_time":t1-t0, "hstry":historial_fitness}
        guardar_resultados(rdol)
    return

if __name__ == "__main__":
    main()