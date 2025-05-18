import numpy as np

def rastrigin(x:np.ndarray) -> float:
    """rastrigin function for n=2"""

    return  (10*2  + ( 
                    (x[:,0]**2 - 10*np.cos(2*np.pi*x[:,0])) +
                    (x[:,1]**2 - 10*np.cos(2*np.pi*x[:,1]))
                 ))
# between -5.12 and 5.12

def booth(x:np.ndarray) ->float:
    return ((x[:,0] + 2*x[:,1] - 7)**2 + (2*x[:,0] + x[:,1] -5)**2)
# entre -10 y 10


def himmelblau(x:np.ndarray) -> float:
    return (x[:,0]**2 + x[:,1] -11 )**2 + (x[:,0] + x[:,1]**2 -7 )**2