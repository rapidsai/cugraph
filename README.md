# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of graph analytics that process data found in GPU Dataframes - see [cuDF](https://github.com/rapidsai/cudf).  cuGraph aims to provide a NetworkX-like API that will be familiar to data scientists, so they can now build GPU-accelerated workflows more easily.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the `master` branch.



```markdown
import cugraph

# assuming that data has been loaded into a cuDF (using read_csv) Dataframe
# create a Graph using the source (src) and destination (dst) vertex pairs the GDF  
G = cugraph.Graph()
G.add_edge_list(gdf["src"], gdf["dst"])

# Call cugraph.pagerank to get the pagerank scores
gdf_page = cugraph.pagerank(G)

for i in range(len(gdf_page)):
	print("vertex " + str(gdf_page['vertex'][i]) + 
		" PageRank is " + str(gdf_page['pagerank'][i]))  
```



## Supported Algorithms:

| Algorithm                                     | Scale      | Notes |
| --------------------------------------------- | ---------- | ----- |
| PageRank                                      | Single-GPU |       |
| Jaccard Similarity                            | Single-GPU |       |
| Weighted Jaccard                              | Single-GPU |       |
| Overlap Similarity                            | Single-GPU |       |
| SSSP                                          | Single-GPU |       |
| BSF                                           | Single-GPU |       |
| Triangle Counting                             | Single-GPU |       |
| Subgraph Extraction                           | Single-GPU |       |
| Spectral Clustering - Balanced-Cut            | Single-GPU |       |
| Spectral Clustering - Modularity Maximization | Single-GPU |       |
| Louvain                                       | Single-GPU |       |
| Renumbering                                   | Single-GPU |       |
| Basic Graph Statistics                        | Single-GPU |       |



## cuGraph 0.7 Notice

cuGraph version 0.7 has some limitations:

- Only Int32 Vertex ID are supported
- Only float (FP32) edge data is supported
- Vertex numbering is assumed to start at zero

These limitations are being addressed and will be fixed future versions.











## Getting cuGraph
### Intro
There are 4 ways to get cuGraph :
1. [Quick start with Docker Demo Repo](#quick)
1. [Conda Installation](#conda)
1. [Build from Source](#source)



<a name="quick"></a>

## Quick Start  

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.



<a name="conda"></a>

### Conda

It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=9.2

# CUDA 10.0
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.0
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.



<a name="source"></a>

### Build from Source and Contributing 

Please see our [guide for building and contributing to cuGraph](CONTRIBUTING.md).



## Documentation

Python API documentation can be generated from [docs](docs) directory.



## C++ ABI issues

cuGraph builds with C++14 features.  By default, we build cuGraph with the latest ABI (the ABI changed with C++11).  The version of cuDF pointed to in the conda installation above is build with the new ABI.

If you see link errors indicating trouble finding functions that use C++ strings when trying to build cuGraph you may have an ABI incompatibility.

There are a couple of complications that may make this a problem:
* if you need to link in a library built with the old ABI, you may need to build the entire tool chain from source using the old ABI.
* if you build cudf from source (for whatever reason), the default behavior for cudf (at least through version 0.5.x) is to build using the old ABI.  You can build with the new ABI, but you need to follow the instructions in CUDF to explicitly turn that on.

If you must build cugraph with the old ABI, you can use the following command (instead of the cmake call above):

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=OFF
```



### (OPTIONAL) Set environment variable on activation

It is possible to configure the conda environment to set environmental variables on activation. Providing instructions to set PATH to include the CUDA toolkit bin directory and LD_LIBRARY_PATH to include the CUDA lib64 directory will be helpful.



```bash
cd  ~/anaconda3/envs/cugraph_dev

mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```



Next the env_vars.sh file needs to be edited

```bash
vi ./etc/conda/activate.d/env_vars.sh

#!/bin/bash
export PATH=/usr/local/cuda-10.0/bin:$PATH # or cuda-9.2 if using CUDA 9.2
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH # or cuda-9.2 if using CUDA 9.2
```



```
vi ./etc/conda/deactivate.d/env_vars.sh

#!/bin/bash
unset PATH
unset LD_LIBRARY_PATH
```





------



## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.

