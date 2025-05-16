import numpy as np
import logging
from collections.abc import Callable

from .generation import generate_uniform
from .selection import roulette_wheel_selection, tournament_sampling, stochastic_sampling
from .crossover import perform_crossover
from .mutation import mutate_real, mutate_discrete
from .log import start_logs, log_config,log_row
from .utils import evaluate, sort_population, get_stats, bounce_population, create_bound_matrix
 


def maximize(param:dict, target_function:Callable[[np.ndarray],np.ndarray]) -> tuple:
    """Returns a tuple (arg_max, max_value) for the target function """

    start_logs()
    log_config(param=param)
    history = {"mean":[],"stdev":[],"max_fit":[],"min_fit":[],"median":[]}

    lower_bound_matrix, upper_bound_matrix = create_bound_matrix(lower_bounds=param['lower_bound'],
                                                                 upper_bounds=param["upper_bound"],
                                                                 pop_size=param["pop_size"])

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
        children = perform_crossover(parents,method=param['crossover_method'])
        mutated_children = mutate_real(children,mutation_rate=param['mutation_rate'])
        
        population = mutated_children
        population[-1] = best_individual  # keep the best individual

        population = bounce_population(population,lower_bound_matrix,upper_bound_matrix) #truncate if they are out of the allowed zone




    # Evaluate the las gen out of the loop
    fitness = evaluate(population=population,fitness_function=target_function)
    sorted_population,sorted_fitness = sort_population(population=population,fitness=fitness)


    print(f"Best solution: {sorted_fitness[0]}")
    print(f"Found in point {sorted_population[0]}")
    return sorted_population[0], sorted_fitness[0]
