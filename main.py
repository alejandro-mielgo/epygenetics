# import matplotlib.pyplot as plt
import numpy as np
from generation import generate_uniform
from selection import roulette_wheel_selection, tournament_sampling, stochastic_sampling
from crossover import crossover
from Population import Population
from mutation import mutate_real, mutate_discrete
import logging


def f(x: list[float]) -> float:

    return 60 - (
        x[0] ** 2
        - 10 * np.cos(2 * np.pi * x[0])
        + x[1] ** 2
        - 10 * np.cos(2 * np.pi * x[1])
    )


# def f(x: np.ndarray) -> float:
#     return -x[0] ** 2 -x[1] ** 2 + 20



if __name__ == "__main__":


    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(message)s',filename='logs/genetics.log',level=logging.INFO)


    n_genarations :int = 100
    lower_bound :list[float] = [-5,-5]
    upper_bound :list[float]= [5, 5]
    pop_size :int = 100
    mutation_rate = 0.15
    crossover_method="opc"
    history = {}

    population : Population = Population(kind="real", population_size=pop_size, dimension=2)
    population.generate_uniform(lower_bound=lower_bound, upper_bound=upper_bound)


    for i in range(n_genarations):

        population.evaluate(f)
        population.sort_by_fitness()
        parents_indexes :np.ndarray=  tournament_sampling(population.population_size, 2)
        parents :np.ndarray = population.individuals[parents_indexes]
        children = crossover(parents, method=crossover_method)
        population.individuals = mutate_real(children, mutation_rate=mutation_rate)


    population.sort_by_fitness()
    population.evaluate(f)
    population.get_stats()
    logging.info(f"n_gen:{n_genarations}, pop size: {pop_size}, mut_rate:{mutation_rate}")
    logging.info(population.stats)
    print(population.fitness[0])
    print(population.individuals[0])