import numpy as np


def generate_uniform(
    lower_bound: list[float], upper_bound: list[float], pop_size: int
) -> np.ndarray:
    
    if pop_size % 2 != 0:
        raise ValueError("population size must be even")
    if len(lower_bound)!=len(upper_bound):
        raise ValueError("upper bound and lower bound must have the same dimension")
    
    solution_ranges = [upper-lower for lower, upper in zip(lower_bound,upper_bound)]
    if any(solution_range<0 for solution_range in solution_ranges):
        raise ValueError("no solution space with current bounds")
        

    new_population = np.array(
        [
            [
                np.random.uniform(low, high)
                for low, high in zip(lower_bound, upper_bound)
            ]
            for individual in range(pop_size)
        ]
    )

    return new_population


def generate_bool(n_genes: int,population_size: int)->np.ndarray:

    return np.array([np.random.random_integers(0,1,n_genes) for i in range(population_size)])


def generate_categorical( n_genes: int,n_categories: int, population_size: int )->np.ndarray:

    if n_categories<=1:
        raise ValueError("Number of categories must be >1")
    return np.array([np.random.random_integers(0,n_categories-1,n_genes) for i in range(population_size)])


if __name__ == "__main__":

    np.random.seed(0)

    population = generate_uniform([0,0],[1,1],10)
    print(population)
    bool_population = generate_bool(7,10)
    print(bool_population)
    np.random.seed = "gato"

    categorical_population = generate_categorical(7,2,10)
    print(categorical_population)
