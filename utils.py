import numpy as np


def evaluate(population: np.ndarray, fitness_function) -> np.ndarray:
    fitness = np.array(
        [fitness_function(individual) for individual in population]
    ).reshape(len(population), 1)
    return fitness


def sort_population(population: np.ndarray, fitness: np.ndarray) -> None:
    order_indexes = fitness.argsort(axis=0)  # orders from smallest to largest
    population = population[order_indexes[::-1]].reshape(population.shape)
    fitness = fitness[order_indexes[::-1]].reshape(fitness.shape)
    return (population, fitness)


def get_stats(fitness: np.ndarray)->list[dict]:
    mean_fitness = np.mean(fitness)
    stdev_fitness = np.std(fitness, ddof=1)
    max_fitness = np.max(fitness)
    min_fitness = np.min(fitness)
    median_fitness = np.median(fitness)

    return {"mean":mean_fitness, 
            "stdev":stdev_fitness,
            "max_f":max_fitness,
            "min_fitness": min_fitness,
            "median":median_fitness}


