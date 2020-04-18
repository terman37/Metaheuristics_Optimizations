# Metaheuristics Assignment

## Discrete Optimization:

### TSP problems:

National Traveling salesman problems from [math.uwaterloo.ca](http://www.math.uwaterloo.ca/tsp/world/countries.html)

**Djibouti:**

- 38 cities
- optimal tour: 6656

**Qatar:**

- 194 cities
- optimal tour: 9352

**Algorithm used:** 

- Genetic algorithm, well suited for this kind of problem: each chromosome represent a route through all cities. Each gene represent a city. All chromosomes are permutation of the genes.

**Comments:**

- For Djibouti, GA shows good results in a relatively short computational time. 

- For Qatar, I was not able to get good results with applying GA right away. Jmetalpy library was slow, so I decided to switch to DEAP library. Results were better and faster but still far from optimal route. 

  I found some interesting reading on how to initialize population [here](https://www.researchgate.net/publication/283031756_An_Improved_Genetic_Algorithm_with_Initial_Population_Strategy_for_Symmetric_TSP): the idea is to:

  - create clusters of cities using simple k-means
  - find the optimal path between clusters (using GA)
  - for each cluster find optimal path (using GA)
  - link clusters together by disconnecting one edge of the cluster and linking it to next one randomly to create initial population
  - run a GA on this new suboptimal population.

Comparison results in table below for Djibouti

| jmetalpy                                                     | DEAP                                                         | Kmean + DEAP                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [notebook](1-tsp_dj38/tsp_dj38_jmetalpy.ipynb)               | notebook                                                     | notebook                                                     |
| <img src="1-tsp_dj38/dj38-jmetalpy.png" alt="dj38-jmetalpy" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-deap.png" alt="dj38-deap" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-kmean-deap.png" alt="dj38-kmean-deap" style="zoom:50%;" /> |
|                                                              |                                                              |                                                              |









The chosen algorithm and a justification of this choice
\- The parameters of the algorithm
\- The final results, both solution and fitness
\- The number of function evaluations
\- The stopping criterion
\- The computational time
\- The convergence curve (fitness as a function of time)  





conda install ipykernel

conda install matplotlib

conda install pandas

conda install scipy

conda install numba



pip install jmetalpy

conda install -c conda-forge pygmo