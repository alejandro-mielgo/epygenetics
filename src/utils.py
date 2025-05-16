import numpy as np
import logging
import time

from collections.abc import Callable


def evaluate(population: np.ndarray, fitness_function:Callable) -> np.ndarray:
    return fitness_function(population)


def create_bound_matrix(lower_bounds:list,upper_bounds:list,pop_size:int) -> tuple[np.ndarray,np.ndarray]:
    n_vars = len(lower_bounds)
    lower_bound_matrix:np.ndarray = np.repeat(lower_bounds,repeats=pop_size).reshape(n_vars,pop_size).transpose()
    upper_bound_matrix:np.ndarray = np.repeat(upper_bounds,repeats=pop_size).reshape(n_vars,pop_size).transpose()
    return lower_bound_matrix, upper_bound_matrix


def bounce_population(population: np.ndarray,lower_bound_matrix:list,upper_bound_matrix:list):
    
    bounced_pop = population.copy()
    bounced_pop = np.maximum(bounced_pop,lower_bound_matrix)
    bounced_pop = np.minimum(bounced_pop,upper_bound_matrix)

    return bounced_pop


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

    print(population)
    bounced_pop = bounce_population(population=population, lower_bounds=(1,2),upper_bounds=(5,6))
    print(bounced_pop)

    
