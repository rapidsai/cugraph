# LAP
Implementation of ***O(n^3) Alternating Tree Variant*** of Hungarian Algorithm on NVIDIA CUDA-enabled GPU.

This implementation solves a batch of ***k*** **Linear Assignment Problems (LAP)**, each with ***nxn*** matrix of single floating point cost values. At optimality, the algorithm produces an assignment with ***minimum*** cost.

The API can be used to query optimal primal and dual costs, optimal assignment vector, and optimal row/column dual vectors for each subproblem in the batch.

cuGraph exposes the Hungarian algorithm, the actual implementation is contained in the RAFT library which contains some common tools and kernels shared between cuGraph and cuML.

Following parameters can be used to tune the performance of algorithm:

1. epsilon: (in raft/lap/lap_kernels.cuh) This parameter controls the tolerance on the floating point precision. Setting this too small will result in increased solution time because the algorithm will search for precise solutions. Setting it too high may cause some inaccuracies.

2. BLOCKDIMX, BLOCKDIMY: (in raft/lap/lap_functions.cuh) These parameters control threads_per_block to be used along the given dimension. Set these according to the device specifications and occupancy calculation.

***This library is licensed under Apache License 2.0. Please cite our paper, if this library helps you in your research.***

- Harvard citation style

  Date, K. and Nagi, R., 2016. GPU-accelerated Hungarian algorithms for the Linear Assignment Problem. Parallel Computing, 57, pp.52-72.

- BibTeX Citation block to be used in LaTeX bibliography file:

```
@article{date2016gpu,
  title={GPU-accelerated Hungarian algorithms for the Linear Assignment Problem},
  author={Date, Ketan and Nagi, Rakesh},
  journal={Parallel Computing},
  volume={57},
  pages={52--72},
  year={2016},
  publisher={Elsevier}
}
```

The paper is available online on [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S016781911630045X).
