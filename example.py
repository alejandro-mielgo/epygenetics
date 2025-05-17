import numpy as np

from src.optimizer import optimize
from src.target_functions import rastrigin, booth

    

parameters : dict = {
    "n_generations"  : 100,
    "lower_bound" : (-10,-10),
    "upper_bound"  : ( 10, 10),
    "pop_size"  : 100,
    "mutation_rate" : 0.4,      # how ofter a gene mutates
    "mutation_n_stdevs" : 1,    # how much it mutates in variable stdvs 
    "crossover_method" : "opc",
    "minimize": True
}

if __name__ == "__main__":
    optimize(param=parameters, target_function=rastrigin)

