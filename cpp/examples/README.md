# libcugraph examples

Example codes on how to use libcugraph.

## Contents

- sg_examples: contains simple code on how to use libcugraph to run different cugraph algorithms in a signle GPU node.

- mg_examples: contains example code on how to use libcugraph to run different cugraph algorithms in a multi GPU node.

## Build instructions:

Run `build.sh` to build the above listed examples.

## Run instructions

`executable path_to_your_csv_graph_file [memory allocation mode]`

Memory allocation mode can be one of the followings

- cuda
- pool 
- binning
- managed