# libcugraph examples


## Contents

- users: Example codes on how to use libcugraph.

  - single_gpu_application: example code on how to use libgraph to run different graph algorithms in single-GPU.

  - multi_gpu_application: example code on how to use libgraph to run different graph algorithms in multi-GPU.

- developers: Example codes to demonstrate graph partition and primitives in cugraph.

  -  graph_partition: code to explain vertex and edge partitioning in cugraph.

  -  cugraph_operations: example code for using cugraph primitives for simple graph operations needed to implement graph algorithms.

## Build instructions:

Run `build.sh all` from the examples dir to build the above listed examples.

```sh
~/cugraph/cpp/examples$./build.sh all
```

## Run instructions

For single-GPU application

`./path_to_executable path_to_a_csv_graph_file`

For muti-GPU application

`mpirun -np 2 path_to_executable  path_to_a_csv_graph_file`
