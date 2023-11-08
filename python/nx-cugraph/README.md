# nx-cugraph

## Description
[RAPIDS](https://rapids.ai) nx-cugraph is a [backend to NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html#backends)
to run supported algorithms with GPU acceleration.

## System Requirements

nx-cugraph requires the following:

 * NVIDIA GPU, Pascal architecture or later
 * CUDA 11.2, 11.4, 11.5, 11.8, or 12.0
 * Python versions 3.9, 3.10, or 3.11
 * NetworkX >= version 3.2

More details about system requirements can be found in the [RAPIDS System Requirements documentation](https://docs.rapids.ai/install#system-req)..

## Installation

nx-cugraph can be installed using either conda or pip.

### conda
```
conda install -c rapidsai-nightly -c conda-forge -c nvidia nx-cugraph
```
### pip
```
python -m pip install nx-cugraph-cu11 --extra-index-url https://pypi.nvidia.com
```
Notes:

 * Nightly wheel builds will not be available until the 23.12 release, therefore the index URL for the stable release version is being used in the pip install command above.
 * Additional information relevant to installing any RAPIDS package can be found [here](https://rapids.ai/#quick-start).

## Enabling nx-cugraph

NetworkX will use nx-cugraph as the graph analytics backend if any of the
following are are used:

### `NETWORKX_AUTOMATIC_BACKENDS` environment variable.
The `NETWORKX_AUTOMATIC_BACKENDS` environment variable can be used to have NetworkX automatically dispatch to specified backends an API is called that the backend supports.
Set `NETWORKX_AUTOMATIC_BACKENDS=cugraph` to use nx-cugraph to GPU accelerate supported APIs with no code changes.
Example:
```
bash> NETWORKX_AUTOMATIC_BACKENDS=cugraph python my_networkx_script.py
```

### `backend=` keyword argument
To explicitly specify a particular backend for an API, use the `backend=`
keyword argument. This argument takes precedence over the
`NETWORKX_AUTOMATIC_BACKENDS` environment variable. This requires anyone
running code that uses the `backend=` keyword argument to have the specified
backend installed.

Example:
```
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```

### Type-based dispatching

NetworkX also supports automatically dispatching to backends associated with
specific graph types. Like the `backend=` keyword argument example above, this
requires the user to write code for a specific backend, and therefore requires
the backend to be installed, but has the advantage of ensuring a particular
behavior without the potential for runtime conversions.

To use type-based dispatching with nx-cugraph, the user must import the backend
directly in their code to access the utilities provided to create a Graph
instance specifically for the nx-cugraph backend.

Example:
```
import networkx as nx
import nx_cugraph as nxcg

G = nx.Graph()
...
nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
nx.betweenness_centrality(nxcg_G, k=1000)  # nxcg Graph type causes cugraph backend
                                           # to be used, no conversion necessary
```

## Supported Algorithms

The nx-cugraph backend to NetworkX connects
[pylibcugraph](../../readme_pages/pylibcugraph.md) (cuGraph's low-level python
interface to its CUDA-based graph analytics library) and
[CuPy](https://cupy.dev/) (a GPU-accelerated array library) to NetworkX's
familiar and easy-to-use API.

Below is the list of algorithms (many listed using pylibcugraph names),
available today in pylibcugraph or implemented using CuPy, that are or will be
supported in nx-cugraph.

| feature/algo | release/target version |
| ----- | ----- |
| analyze_clustering_edge_cut | ? |
| analyze_clustering_modularity | ? |
| analyze_clustering_ratio_cut | ? |
| balanced_cut_clustering | ? |
| betweenness_centrality | 23.10 |
| bfs | ? |
| core_number | ? |
| degree_centrality | 23.12 |
| ecg | ? |
| edge_betweenness_centrality | 23.10 |
| ego_graph | ? |
| eigenvector_centrality | 23.12 |
| get_two_hop_neighbors | ? |
| hits | 23.12 |
| in_degree_centrality | 23.12 |
| induced_subgraph | ? |
| jaccard_coefficients | ? |
| katz_centrality | 23.12 |
| k_core | ? |
| k_truss_subgraph | 23.12 |
| leiden | ? |
| louvain | 23.10 |
| node2vec | ? |
| out_degree_centrality | 23.12 |
| overlap_coefficients | ? |
| pagerank | 23.12 |
| personalized_pagerank | ? |
| sorensen_coefficients | ? |
| spectral_modularity_maximization | ? |
| sssp | 23.12 |
| strongly_connected_components | ? |
| triangle_count | ? |
| uniform_neighbor_sample | ? |
| uniform_random_walks | ? |
| weakly_connected_components | ? |

To request nx-cugraph backend support for a NetworkX API that is not listed
above, visit the [cuGraph GitHub repo](https://github.com/rapidsai/cugraph).
