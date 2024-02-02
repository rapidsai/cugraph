# nx-cugraph

## Description
[RAPIDS](https://rapids.ai) nx-cugraph is a [backend to NetworkX](https://networkx.org/documentation/stable/reference/utils.html#backends)
to run supported algorithms with GPU acceleration.

## System Requirements

nx-cugraph requires the following:

 * NVIDIA GPU, Pascal architecture or later
 * CUDA 11.2, 11.4, 11.5, 11.8, or 12.0
 * Python versions 3.9, 3.10, or 3.11
 * NetworkX >= version 3.2

More details about system requirements can be found in the [RAPIDS System Requirements documentation](https://docs.rapids.ai/install#system-req).

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
following are used:

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

Below is the list of algorithms that are currently supported in nx-cugraph.

### Algorithms

```
bipartite
 ├─ basic
 │   └─ is_bipartite
 └─ generators
     └─ complete_bipartite_graph
centrality
 ├─ betweenness
 │   ├─ betweenness_centrality
 │   └─ edge_betweenness_centrality
 ├─ degree_alg
 │   ├─ degree_centrality
 │   ├─ in_degree_centrality
 │   └─ out_degree_centrality
 ├─ eigenvector
 │   └─ eigenvector_centrality
 └─ katz
     └─ katz_centrality
cluster
 ├─ average_clustering
 ├─ clustering
 ├─ transitivity
 └─ triangles
community
 └─ louvain
     └─ louvain_communities
components
 ├─ connected
 │   ├─ connected_components
 │   ├─ is_connected
 │   ├─ node_connected_component
 │   └─ number_connected_components
 └─ weakly_connected
     ├─ is_weakly_connected
     ├─ number_weakly_connected_components
     └─ weakly_connected_components
core
 ├─ core_number
 └─ k_truss
dag
 ├─ ancestors
 └─ descendants
isolate
 ├─ is_isolate
 ├─ isolates
 └─ number_of_isolates
link_analysis
 ├─ hits_alg
 │   └─ hits
 └─ pagerank_alg
     └─ pagerank
operators
 └─ unary
     ├─ complement
     └─ reverse
reciprocity
 ├─ overall_reciprocity
 └─ reciprocity
shortest_paths
 └─ unweighted
     ├─ single_source_shortest_path_length
     └─ single_target_shortest_path_length
traversal
 └─ breadth_first_search
     ├─ bfs_edges
     ├─ bfs_layers
     ├─ bfs_predecessors
     ├─ bfs_successors
     ├─ bfs_tree
     ├─ descendants_at_distance
     └─ generic_bfs_edges
tree
 └─ recognition
     ├─ is_arborescence
     ├─ is_branching
     ├─ is_forest
     └─ is_tree
```

### Generators

```
classic
 ├─ barbell_graph
 ├─ circular_ladder_graph
 ├─ complete_graph
 ├─ complete_multipartite_graph
 ├─ cycle_graph
 ├─ empty_graph
 ├─ ladder_graph
 ├─ lollipop_graph
 ├─ null_graph
 ├─ path_graph
 ├─ star_graph
 ├─ tadpole_graph
 ├─ trivial_graph
 ├─ turan_graph
 └─ wheel_graph
community
 └─ caveman_graph
small
 ├─ bull_graph
 ├─ chvatal_graph
 ├─ cubical_graph
 ├─ desargues_graph
 ├─ diamond_graph
 ├─ dodecahedral_graph
 ├─ frucht_graph
 ├─ heawood_graph
 ├─ house_graph
 ├─ house_x_graph
 ├─ icosahedral_graph
 ├─ krackhardt_kite_graph
 ├─ moebius_kantor_graph
 ├─ octahedral_graph
 ├─ pappus_graph
 ├─ petersen_graph
 ├─ sedgewick_maze_graph
 ├─ tetrahedral_graph
 ├─ truncated_cube_graph
 ├─ truncated_tetrahedron_graph
 └─ tutte_graph
social
 ├─ davis_southern_women_graph
 ├─ florentine_families_graph
 ├─ karate_club_graph
 └─ les_miserables_graph
```

### Other

```
convert_matrix
 ├─ from_pandas_edgelist
 └─ from_scipy_sparse_array
```

To request nx-cugraph backend support for a NetworkX API that is not listed
above, visit the [cuGraph GitHub repo](https://github.com/rapidsai/cugraph).
