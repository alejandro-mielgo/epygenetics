## Genetic Algorithm

Genetic algorithm for function optimization using numpy.

Install:

```pip install epygenetics```

### Format to define target function:

```math
 f(x,y) = (x+2y-7)^2 +2(x+y-5)^2
```
would be defined as 

```python
def booth(x:np.ndarray)->np.ndarray:
    return -((x[:,0] + 2*x[:,1] - 7)**2 + (2*x[:,0] + x[:,1] -5)**2)
```

see the examples folder for more details

### Crossover methods:
- one point crossover
- two point crossover

### Selection methods:
- roulette wheel selection
- stochastic_sampling
- tournament_sampling


[Test functions for optimization algorithms](https://en.wikipedia.org/wiki/Test_functions_for_optimization)