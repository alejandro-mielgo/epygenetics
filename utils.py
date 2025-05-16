import numpy as np
import logging
import time

from collections.abc import Callable


def evaluate(population: np.ndarray, fitness_function:Callable) -> np.ndarray:
    return fitness_function(population)


def sort_population(population: np.ndarray, fitness: np.ndarray) -> tuple:
    order_indexes = fitness.argsort(axis=0)[::-1]  #argsort orders from smallest to largest, is reversed
    oredered_population = population[order_indexes].reshape(population.shape)
    fitness = fitness[order_indexes].reshape(fitness.shape)
    return oredered_population, fitness


def get_stats(fitness: np.ndarray) -> dict:
    mean_fitness = np.mean(fitness)
    stdev_fitness = np.std(fitness, ddof=1)
    max_fitness = np.max(fitness)
    min_fitness = np.min(fitness)
    median_fitness = np.median(fitness)

    return {"mean":float(mean_fitness), 
            "stdev":float(stdev_fitness),
            "max_fit":float(max_fitness),
            "min_fit": float( min_fitness),
            "median":float(median_fitness)}


def start_logs():
    today = time.strftime("%Y-%m-%d")
    logging.basicConfig(
        filename=f"./logs/{today}_gen.log",
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_config(param:dict)->None:
    logging.info("new execution _________________________________________________________________________________________________")
    for key,val in param.items():
        logging.info(f"{key}:{val}")


def log_row(generation:int, metrics:dict) -> None:
    row:str=f"{str(generation)[:8]:<8}\t"
    for key,value in metrics.items():
        row = row + f"{str(value)[:8]:8}\t"
    logging.info(row)


if __name__=='__main__':

    def f(x:np.ndarray):
        return x[:,0]*x[:,1]
    
    population = np.random.uniform(low=(0,0), high=(10,10), size=(5,2))

    fitness = evaluate(population=population,fitness_function=f)
    print(fitness)
    sorted_population = sort_population(population,fitness)


    # evaluate(population,f)
    
