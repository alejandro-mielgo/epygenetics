from epygenetics import genetic_optimize
from epygenetics import rastrigin, booth, himmelblau

    
parameters : dict = {
    "n_generations"  : 100,
    "lower_bound" : (-5.12,-5.12),  # lower limits for each variable
    "upper_bound"  : ( 5.12, 5.12), # upper limits for each variable
    "pop_size"  : 100,
    "mutation_rate" : 0.5,          # how ofter a gene mutates
    "mutation_n_stdevs" : 1,        # how much it mutates in variable stdvs 
    "crossover_method" : "tpc",     # one point crossover or two point crossover
    "minimize": True
}

if __name__ == "__main__":
    best_indidual,best_solution,history = genetic_optimize(param=parameters, target_function=rastrigin)
    

