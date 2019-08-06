# Cugraph test data


## Python

This directory contains small public datasets in `mtx` and `csv` format used by cuGraph's python tests. Graph details:

| Graph         | V     | E     | Directed | Weighted |
| ------------- | ----- | ----- | -------- | -------- |
| karate        | 34    | 156   | No       | No       |
| dolphin       | 62    | 318   | No       | No       |
| netscience    | 1,589 | 5,484 | No       | Yes      |


**karate** :The graph "karate" contains the network of friendships between the 34 members of a karate club at a US university, as described by Wayne Zachary in 1977.

**dolphin** : The graph dolphins contains an undirected social network of frequent associations between 62 dolphins in a community living off Doubtful Sound, New Zealand, as compiled by Lusseau et al. (2003).                        

**netscience** : The graph netscience contains a coauthorship network of scientists working on network theory and experiment, as compiled by M. Newman in May 2006.


## C++
Cugraph's C++ analytics tests need larger datasets (>5GB uncompressed) and reference results (>125MB uncompressed). They can be downloaded using the provided script.
```
source get_test_data.sh
``` 
You may run this script from elsewhere and store C++ test input to another location. 

Before running the tests, you should let cuGraph know where to find the test input by using:
```
export RAPIDS_DATASET_ROOT_DIR=<path_to_ccp_test_and_reference_data>
```
## Reference
The SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection) : https://sparse.tamu.edu/
