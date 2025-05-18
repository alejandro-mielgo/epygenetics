import numpy as np

from src.genetic_optimizer import optimize
from src.target_functions import rastrigin, booth, himmelblau

    
parameters : dict = {
    "n_generations"  : 100,
    "lower_bound" : (-5.12,-5.12),
    "upper_bound"  : ( 5.12, 5.12),
    "pop_size"  : 100,
    "mutation_rate" : 0.5,      # how ofter a gene mutates
    "mutation_n_stdevs" : 1,    # how much it mutates in variable stdvs 
    "crossover_method" : "tpc", # one point crossover or two point crossover
    "minimize": True
}

if __name__ == "__main__":
    _,_,history = optimize(param=parameters, target_function=himmelblau)

