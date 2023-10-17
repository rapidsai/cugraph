# Cugraph test and benchmark data

## Python

This directory contains small public datasets in `mtx` and `csv` format used by cuGraph's python tests. Graph details:

| Graph         | V     | E     | Directed | Weighted |
| ------------- | ----- | ----- | -------- | -------- |
| karate        | 34    | 156   | No       | No       |
| dolphin       | 62    | 318   | No       | No       |
| netscience    | 1,589 | 5,484 | No       | Yes      |

**karate** : The graph "karate" contains the network of friendships between the 34 members of a karate club at a US university, as described by Wayne Zachary in 1977.

**dolphin** : The graph dolphins contains an undirected social network of frequent associations between 62 dolphins in a community living off Doubtful Sound, New Zealand, as compiled by Lusseau et al. (2003).

**netscience** : The graph netscience contains a coauthorship network of scientists working on network theory and experiment, as compiled by M. Newman in May 2006.



### Modified datasets 

The datasets below were added to provide input that contains self-loops, string vertex IDs, isolated vertices, and multiple edges.

| Graph               | V       | E          | Directed | Weighted  | self-loops | Isolated V | String V IDs | Multi-edges | 
| ------------------- | ------- | ---------- | -------- | --------- | ---------- | ---------- | ------------ | ----------- |
| karate_multi_edge   | 34      | 160        | No       | Yes       | No         | No         | No           | Yes         |
| dolphins_multi_edge | 62      | 325        | No       | Yes       | No         | No         | No           | Yes         |
| karate_s_loop       | 34      | 160        | No       | Yes       | Yes        | No         | No           | No          |
| dolphins_s_loop     | 62      | 321        | No       | Yes       | Yes        | No         | No           | No          |
| karate_mod          | 37      | 156        | No       | No        | No         | Yes        | No           | No          |
| karate_str          | 34      | 156        | No       | Yes       | No         | No         | Yes          | No          |

**karate_multi_edge** : The graph "karate_multi_edge" is a modified version of the  "karate" graph where multi-edges were added

**dolphins_multi_edge** : The graph "dolphins_multi_edge" is a modified version of the  "dolphin" graph where multi-edges were added

**karate_s_loop** : The graph "karate_s_loop" is a modified version of the  "karate" graph where self-loops were added

**dolphins_s_loop** : The graph "dolphins_s_loop" is a modified version of the  "dolphin" graph where self-loops were added

**karate_mod** : The graph "karate_mod" is a modified version of the  "karate" graph where vertices and edges were added

**karate_str** : The graph "karate_str" contains the network of friendships between the 34 members of a karate club at a US university, as described by Wayne Zachary in 1977. The integer vertices were replaced by strings


### Additional datasets

Larger datasets containing self-loops can be downloaded by running the provided script from the `datasets` directory using the `--self_loops` 
option: 
```
cd <repo>/datasets
./get_test_data.sh --self_loops
```
```
<repo>/datasets/self_loops
 |-ca-AstroPh  (5.3M) 
 |-ca-CondMat  (2.8M)
 |-ca-GrQc     (348K)
 |-ca-HepTh    (763K)
```
These datasets are not currently used by any tests or benchmarks

| Graph         | V       | E          | Directed | Weighted | self-loops | Isolated V | String V IDs | Multi-edges |  
| ------------- | ------- | --------   | -------- | -------- | ---------- | ---------- | ------------ | ----------- |
| ca-AstroPh    | 18,772  | 198,110    | No       | No       | Yes        | No         | No           | No          |
| ca-CondMat    | 23,133  | 93,497     | No       | Yes      | Yes        | No         | No           | No          |
| ca-GrQc       | 5,242   | 14,387     | No       | No       | Yes        | No         | No           | No          |
| ca-HepTh      | 9,877   | 25,998     | No       | Yes      | Yes        | No         | No           | No          |

**ca-AstroPh** : The graph "ca-AstroPh" covers scientific collaborations between authors papers submitted to Astro Physics category in the period from January 1993 to April 2003 (124 months), as described by J. Leskovec, J. Kleinberg and C. Faloutsos in 2007.

**ca-CondMat** : The graph "ca-CondMat" covers scientific collaborations between authors papers submitted to Condense Matter category in the period from January 1993 to April 2003 (124 months), as described by J. Leskovec, J. Kleinberg and C. Faloutsos in 2007.

**ca-GrQc** : The graph "ca-GrQc" covers scientific collaborations between authors papers submitted to General Relativity and Quantum Cosmology category in the period from January 1993 to April 2003 (124 months), as described by J. Leskovec, J. Kleinberg and C. Faloutsos in 2007.

**ca-HepTh** : The graph "ca-HepTh" covers scientific collaborations between authors papers submitted to High Energy Physics - Theory category in the period from January 1993 to April 2003 (124 months), as described by J. Leskovec, J. Kleinberg and C. Faloutsos in 2007.


## Custom path to larger datasets directory  

Cugraph's C++ and Python analytics tests need larger datasets (>5GB uncompressed) and reference results (>125MB uncompressed). They can be downloaded by running the provided script from the `datasets` directory.
```
cd <repo>/datasets
./get_test_data.sh
```
You may run this script from elsewhere and store C++ or Python test input to another location.

Before running the tests, you should let cuGraph know where to find the test input by using:
```
export RAPIDS_DATASET_ROOT_DIR=<path_to_datasets_dir>
```


## Benchmarks

Cugraph benchmarks (which can be found [here](../benchmarks)) also use datasets installed to this folder. Because the datasets used for benchmarking are also quite large (~14GB uncompressed), they are not installed by default. To install datasets for benchmarks, run the same script shown above from the `datasets` directory using the `--benchmark` option:
```
cd <repo>/datasets
./get_test_data.sh --benchmark
```
The datasets installed for benchmarks currently include CSV files for use in creating both directed and undirected graphs:
```
<repo>/datasets/csv
 |- directed
 |--- cit-Patents.csv       (250M)
 |--- soc-LiveJournal1.csv  (965M)
 |- undirected
 |--- europe_osm.csv        (1.8G)
 |--- hollywood.csv         (1.5G)
 |--- soc-twitter-2010.csv  (8.8G)
```
The benchmark datasets are described below:
| Graph             | V          | E             | Directed | Weighted |
| ----------------- | ---------- | ------------- | -------- | -------- |
| cit-Patents       |  3,774,768 |    16,518,948 | Yes      | No       |
| soc-LiveJournal1  |  4,847,571 |    43,369,619 | Yes      | No       |
| europe_osm        | 50,912,018 |    54,054,660 | No       | No       |
| hollywood         |  1,139,905 |    57,515,616 | No       | No       |
| soc-twitter-2010  | 21,297,772 |   265,025,809 | No       | No       |

**cit-Patents** : A citation graph that includes all citations made by patents granted between 1975 and 1999, totaling 16,522,438 citations.

**soc-LiveJournal** : A graph of the LiveJournal social network.

**europe_osm** : A graph of OpenStreetMap data for Europe.

**hollywood** : A graph of movie actors where vertices are actors, and two actors are joined by an edge whenever they appeared in a movie together.

**soc-twitter-2010** : A network of follower relationships from a snapshot of Twitter in 2010, where an edge from i to j indicates that j is a follower of i.

_NOTE: the benchmark datasets were converted to a CSV format from their original format described in the reference URL below, and in doing so had edge weights and isolated vertices discarded._

## Reference
The SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection) : https://sparse.tamu.edu/
The Stanford Network Analysis Platform (SNAP) 
