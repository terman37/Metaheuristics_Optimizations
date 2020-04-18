# Metaheuristics optimizations:

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

### Comparison results in table below for Djibouti:

| jmetalpy                                                     | DEAP                                                         | Kmean + DEAP                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [notebook](1-tsp_dj38/tsp_dj38_jmetalpy.ipynb)               | [notebook](1-tsp_dj38/tsp_dj38_deap.ipynb)                   | [notebook](1-tsp_dj38/tsp_dj38_deap_kmeans_init.ipynb)       |
| <img src="1-tsp_dj38/dj38-jmetalpy.png" alt="dj38-jmetalpy" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-deap.png" alt="dj38-deap" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-kmean-deap.png" alt="dj38-kmean-deap" style="zoom:50%;" /> |
| <img src="1-tsp_dj38/dj38-jmetalpy_tour.png" alt="dj38-jmetalpy_tour" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-deap-tour.png" alt="dj38-deap-tour" style="zoom:50%;" /> | <img src="1-tsp_dj38/dj38-kmean-deap_tour.png" alt="dj38-kmean-deap_tour" style="zoom:50%;" /> |



### Comparison results in table below for Qatar:

| jmetalpy                                                     | DEAP                                                         | Kmean + DEAP                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [notebook](2-tsp_qa194/tsp_qa194_jmetalpy.ipynb)             | [notebook](2-tsp_qa194/tsp_qa194_deap.ipynb)                 | [notebook](2-tsp_qa194/tsp_qa194_deap_kmeans_init.ipynb)     |
| <img src="2-tsp_qa194/qa194-jmetalpy.png" alt="qa194-jmetalpy" style="zoom:50%;" /> | <img src="2-tsp_qa194/qa194-deap.png" alt="qa194-deap" style="zoom:50%;" /> | <img src="2-tsp_qa194/qa194-kmean-deap.png" alt="qa194-kmean-deap" style="zoom:50%;" /> |
| <img src="2-tsp_qa194/qa194-jmetalpy-tour.png" alt="qa194-jmetalpy-tour" style="zoom:50%;" /> | <img src="2-tsp_qa194/qa194-deap-tour.png" alt="qa194-deap-tour" style="zoom:50%;" /> | <img src="2-tsp_qa194/qa194-kmean-deap-tour.png" alt="qa194-kmean-deap-tour" style="zoom:50%;" /> |



