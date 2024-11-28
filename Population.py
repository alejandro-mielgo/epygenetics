import numpy as np
from generation import generate_uniform


class Population:


    def __init__(
        self, kind: str, population_size: int, dimension: int, **kwargs
    ) -> None:
        self.individuals = None
        self.fitness = None
        self.stats = None
        self.kind = kind
        self.population_size = population_size
        self.dimension = dimension

    def __iter__(self):
        return (individual for individual in self.individuals)

    def __len__(self)->int:
        return self.population_size

    def __str__(self)->str:
        iniciated = False if self.individuals is None else True
        header: str = (
            f"population kind:\t{self.kind}\npopulation size:\t{self.population_size}\nIniciated:\t\t{iniciated}\n"
        )
        return header


    def bounce(self, lower_bound: list[float], upper_bound: list[float]) -> None:
        
        for i, individual in enumerate(self.individuals):
            for gen in range(self.dimension):
                
                if self.individuals[i,gen]< lower_bound[gen]:
                    self.individuals[i,gen] = lower_bound[gen]

                if self.individuals[i,gen]> upper_bound[gen]:
                    self.individuals[i,gen] = upper_bound[gen]
                

    def evaluate(self, fitness_function) -> np.ndarray:
        self.fitness = np.array(
            [fitness_function(individual) for individual in self.individuals]
        ).reshape(len(self.individuals), 1)
        return self.fitness
    

    def generate_uniform(self, lower_bound, upper_bound) -> np.ndarray:

        if self.kind != "real":
            raise ValueError(
                "domain problem is not real, can't generate uniform population"
            )
        if self.dimension != len(upper_bound) or self.dimension != len(lower_bound):
            raise ValueError(
                "wrong dimension of limits or problem dimension (aka number of genes per individual)"
            )

        self.individuals = generate_uniform(
            lower_bound, upper_bound, self.population_size
        )
        return self.individuals


    def generate_categorical(self, n_categories):

        if self.kind!="categorical":
            raise ValueError("problem must be categorical to use this method")


    def get_stats(self) -> dict:
        if self.fitness is not None:         
            self.stats =  {
                "mean": float(np.mean(self.fitness)),
                "stdev": float(np.std(self.fitness, ddof=1)),
                "max": float(np.max(self.fitness)),
                "min": float(np.min(self.fitness)),
                "median": float(np.median(self.fitness))
            }
            return self.stats


    def sort_by_fitness(self) -> None:
        order_indexes = self.fitness.argsort(axis=0)  # orders from smallest to largest
        self.individuals = self.individuals[order_indexes[::-1]].reshape(
            self.individuals.shape
        )
        self.fitness = self.fitness[order_indexes[::-1]].reshape(self.fitness.shape)


if __name__ == "__main__":

    def f(x):
        return x[0] + x[1] + x[2]

    pop_1: Population = Population(kind="real",population_size=10, dimension=3)
    pop_1.generate_uniform([-2, -2, -2], [2, 2, 2])

    pop_1.evaluate(f)
    pop_1.sort_by_fitness()
    print(pop_1)


