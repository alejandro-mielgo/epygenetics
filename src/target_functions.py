import numpy as np

def rastrigin(x:np.ndarray) -> float:

    return  -(10  + ( 
                    (x[:,0]**2 - 10*np.cos(2*np.pi*x[:,0])) +
                    (x[:,1]**2 - 10*np.cos(2*np.pi*x[:,1]))
                 ))


def booth(x:np.ndarray):
    return -((x[:,0] + 2*x[:,1] - 7)**2 + (2*x[:,0] + x[:,1] -5)**2)
# entre -10 y 10