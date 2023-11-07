# nx-cugraph

## Description
[RAPIDS](https://rapids.ai) nx-cugraph is a [backend to NetworkX](https://networkx.org/documentation/stable/reference/classes/index.html#backends)
to run supported algorithms with GPU acceleration.

## System Requirements

Using nx-cugraph with this notebook requires the following:

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
