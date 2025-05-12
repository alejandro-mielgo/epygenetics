import numpy as np


def evaluate(population: np.ndarray, fitness_function) -> np.ndarray:
    fitness = np.ma

    print(fitness)
    return fitness


def sort_population(population: np.ndarray, fitness: np.ndarray) -> None:
    order_indexes = fitness.argsort(axis=0)[::-1]  #argsort orders from smallest to largest, is reversed 
    population = population[order_indexes].reshape(population.shape)
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


if __name__=='__main__':
    def f(x:np.ndarray):
        return x[:,0]*x[:,1]
    

    population = np.random.uniform(low=(0,-1),high=(1,2),size=(10,2))
    print(population)
   
    print(f(population))

    # evaluate(population,f)
    
