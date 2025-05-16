import numpy as np

from src.optimizer import maximize

def rastrigin(x:np.ndarray) -> float:

    return  -(10  + ( 
                    (x[:,0]**2 - 10*np.cos(2*np.pi*x[:,0])) +
                    (x[:,1]**2 - 10*np.cos(2*np.pi*x[:,1]))
                 ))


def booth(x:np.ndarray):
    return -((x[:,0] + 2*x[:,1] - 7)**2 + (2*x[:,0] + x[:,1] -5)**2)
# entre -10 y 10

    

parameters : dict = {
    "n_generations"  : 100,
    "lower_bound" : (-10,-10),
    "upper_bound"  : ( 10, 10),
    "pop_size"  : 100,
    "mutation_rate" : 0.5,
    "crossover_method" : "opc"
}

if __name__ == "__main__":
    maximize(param=parameters,target_function=booth)
