# cugraph-equivariant

## Description

cugraph-equivariant library provides fast symmetry-preserving (equivariant) operations and convolutional layers, to accelerate the equivariant neural networks in drug discovery and other domains.

## Build from source

Developers are suggested to create a conda environment that includes the runtime and test dependencies and pip install `cugraph-equivariant` in an editable mode.

```bash
# for cuda 11.8
mamba env create -n cugraph_equivariant -f python/cugraph-equivariant/conda/cugraph_equivariant_dev_cuda-118_arch-x86_64.yaml
conda activate cugraph_equivariant
./build_component.sh -n cugraph-equivariant
```
