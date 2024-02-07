# libcugraph examples

Example codes on how to use libcugraph.

## Contents

- single_gpu_application: example code on how to use libgraph to run different graph algorithms on a single-GPU node.

- multi_gpu_application: example code on how to use libgraph to run different graph algorithms on a multi-GPU node.

-  graph_partition: code to explain vertex and edge partitioning in cugraph.

-  cugraph_operations: example code for using cugraph primitives for simple graph operations needed to implement graph algorithms. 

## Build instructions:

Run `build.sh` to build the above listed examples.

## Run instructions

For single_gpu

`path_to_executable path_to_a_csv_graph_file [memory allocation mode]`

For multi_gpu and graph_partitioning

`mpirun -np 2 path_to_executable  path_to_a_csv_graph_file [memory allocation mode]`

Memory allocation mode can be one of the followings -

- cuda
- pool
- binning
- managed