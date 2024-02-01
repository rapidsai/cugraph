# libcugraph examples

Example codes on how to use libcugraph.

## Contents

- single_gpu: simple code on how to use libcugraph to run different graph algorithms in a single GPU node.

- multi_gpu: example code on how to use libcugraph to run different graph algorithms in a multi GPU node.

-  graph_partitioning: contains code to explain vertex and edge partitioning in cugraph.

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