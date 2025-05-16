import numpy as np
import logging

from generation import generate_uniform
from selection import roulette_wheel_selection, tournament_sampling, stochastic_sampling
from crossover import crossover
from mutation import mutate_real, mutate_discrete

from utils import start_logs, log_config, evaluate, sort_population, get_stats, log_row


def target_function(x:np.ndarray) -> float:

    return 60 - (
        x[:,0] ** 2  - 10 * np.cos(2 * np.pi * x[:,0])
        + x[:,1] ** 2 - 10 * np.cos(2 * np.pi * x[:,1])
    )


param : dict = {
    "n_generations"  : 100,
    "lower_bound" : (-8,-8),
    "upper_bound"  : (8, 8),
    "pop_size"  : 100,
    "mutation_rate" : 0.5,
    "crossover_method" : "opc"
}


if __name__ == "__main__":


    start_logs()
    log_config(param=param)
    history = {"mean":[],"stdev":[],"max_fit":[],"min_fit":[],"median":[]}



    population:np.ndarray = generate_uniform(lower_bound=param["lower_bound"],
                                             upper_bound=param["upper_bound"],
                                             pop_size=param["pop_size"])

    
    logging.info("gen\t\tmean\t\tstdev\t\tmax_fit\t\tmin_fit\t\tmedian ") #Table headers
    

    for i in range(param['n_generations']):
        
        #Evaluate population fitness
        fitness:np.ndarray = evaluate(population=population, fitness_function=target_function)
        
        #Log current iteration metrics
        generation_metrics=get_stats(fitness=fitness)
        for key in generation_metrics:
            history[key].append(generation_metrics[key])
        log_row(generation=i,metrics=generation_metrics)

        # Select parents
        sorted_population,sorted_fitness = sort_population(population=population,fitness=fitness)
        parents_indexes:np.ndarray = tournament_sampling(pop_size=param['pop_size'] , tournament_size=2)
        parents = sorted_population[parents_indexes]
        best_individual = sorted_population[0].copy()


        #Generate children for next population crossover and mutation
        children = crossover(parents,method=param['crossover_method'])
        mutated_children = mutate_real(children,mutation_rate=param['mutation_rate'])
        
        population = mutated_children
        population[-1] = best_individual




    # Evaluate the las gen out of the loop
    fitness = evaluate(population=population,fitness_function=target_function)
    sorted_population,sorted_fitness = sort_population(population=population,fitness=fitness)


    print(f"Best solution: {sorted_fitness[0]}")
    print(f"Found in point {sorted_population[0]}")
