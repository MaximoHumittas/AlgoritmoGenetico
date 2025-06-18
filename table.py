import json
from statistics import mean, stdev

FITNESS_GOAL = 9686.93831  # Valor de referencia

def cargar_datos_json(ruta):
    with open(ruta, "r") as f:
        data = json.load(f)
    return data.get("datos", [])

def extraer_listas(datos):
    fitness_list = [x["best_fit"] for x in datos if "best_fit" in x]
    iters_list = [x["cant_iter"] for x in datos if "cant_iter" in x]
    time_list = [x["exe_time"] for x in datos if "exe_time" in x]
    fitness_historial_total = [
        val for x in datos if "hstry" in x for val in x["hstry"]
    ]
    return fitness_list, iters_list, time_list, fitness_historial_total

def calcular_estadisticas(fitness_list, iters_list, time_list, historial_total):
    fitness_mean = mean(fitness_list)
    fitness_std = stdev(fitness_list) if len(fitness_list) > 1 else 0

    iters_mean = mean(iters_list)
    iters_std = stdev(iters_list) if len(iters_list) > 1 else 0

    time_mean = mean(time_list)
    time_std = stdev(time_list) if len(time_list) > 1 else 0

    fitness_historial_mean = mean(historial_total) if historial_total else 0
    mean_fitness_historial = (
        (1 - abs(FITNESS_GOAL - fitness_historial_mean) / FITNESS_GOAL) * 100
        if FITNESS_GOAL != 0 else 0
    )

    return {
        "best_fitness": min(fitness_list),
        "fitness_mean": fitness_mean,
        "fitness_std": fitness_std,
        "iters_mean": iters_mean,
        "iters_std": iters_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "fitness_historial_mean": fitness_historial_mean,
        "mean_fitness_historial": mean_fitness_historial
    }

def imprimir_estadisticas(stats):
    print("=" * 30)
    print(f"Mejor fitness encontrado: {stats['best_fitness']:.5f}")
    print(f"Mejor fitness promedio: {stats['fitness_mean']:.5f}")
    print(f"Resultado esperado: {FITNESS_GOAL:.5f}")
    print(f"Precisión: {stats['mean_fitness_historial']:.14f}%")
    print(f"Desviación estándar del fitness: {stats['fitness_std']:.5f}")
    print(f"Promedio iteraciones: {stats['iters_mean']:.2f}")
    print(f"Desviación estándar de iteraciones: {stats['iters_std']:.2f}")
    print(f"Tiempo promedio (s): {stats['time_mean']:.4f}")
    print(f"Desviación estándar del tiempo: {stats['time_std']:.4f}")
    print("=" * 30)

def comparar_con_ultimo(stats, ultimo):
    print("\nComparativa con último registro:")
    print(f"{'Métrica':<35}{'Promedio':>15}{'Último':>15}")
    print("-" * 65)
    print(f"{'Fitness final':<35}{stats['fitness_mean']:>15.5f}{ultimo.get('best_fit', 0):>15.5f}")
    print(f"{'Iteraciones':<35}{stats['iters_mean']:>15.2f}{ultimo.get('cant_iter', 0):>15}")
    print(f"{'Tiempo (s)':<35}{stats['time_mean']:>15.4f}{ultimo.get('exe_time', 0):>15.4f}")

    if "hstry" in ultimo and ultimo["hstry"]:
        ult_hist_mean = mean(ultimo["hstry"])
        ult_prec = (1 - abs(FITNESS_GOAL - ult_hist_mean) / FITNESS_GOAL) * 100
        print(f"{'Media historial fitness':<35}{stats['fitness_historial_mean']:>15.5f}{ult_hist_mean:>15.5f}")
        print(f"{'Precisión respecto al objetivo':<35}{stats['mean_fitness_historial']:>15.10f}%{ult_prec:>14.10f}%")
    else:
        print(f"{'Historial no disponible':<35}{'-':>15}{'-':>15}")

def main():
    ruta = "./Anexos/CMGO.json"
    datos = cargar_datos_json(ruta)
    if not datos:
        print("No hay datos disponibles.")
        return

    fitness_list, iters_list, time_list, historial_total = extraer_listas(datos)
    stats = calcular_estadisticas(fitness_list, iters_list, time_list, historial_total)

    imprimir_estadisticas(stats)
    comparar_con_ultimo(stats, datos[-1])  # Compara con el último registro

if __name__ == "__main__":
    main()
