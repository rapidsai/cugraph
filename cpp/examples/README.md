# libcugraph examples

Example codes on how to use libcugraph.

## Contents

- sg_examples: simple code on how to use libcugraph to run different graph algorithms in a single GPU node.

- mg_examples: example code on how to use libcugraph to run different graph algorithms in a multi GPU node.

-  graph_partitioning: contains code to explain vertex and edge partitioning in cugraph.

## Build instructions:

Run `build.sh` to build the above listed examples.

## Run instructions

`executable path_to_your_csv_graph_file [memory allocation mode]`

Memory allocation mode can be one of the followings -

- cuda
- pool 
- binning
- managed