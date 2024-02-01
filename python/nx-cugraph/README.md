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

Below is the list of algorithms that are currently supported or planned to be
supported in nx-cugraph.

| feature/algo                         | release/target version   |
|:-------------------------------------|:-------------------------|
| ancestors                            | 24.02                    |
| average_clustering                   | 24.02                    |
| barbell_graph                        | 23.12                    |
| betweenness_centrality               | 23.10                    |
| bfs_edges                            | 24.02                    |
| bfs_layers                           | 24.02                    |
| bfs_predecessors                     | 24.02                    |
| bfs_successors                       | 24.02                    |
| bfs_tree                             | 24.02                    |
| bull_graph                           | 23.12                    |
| caveman_graph                        | 23.12                    |
| chvatal_graph                        | 23.12                    |
| circular_ladder_graph                | 23.12                    |
| clustering                           | 24.02                    |
| complement                           | 24.02                    |
| complete_bipartite_graph             | 23.12                    |
| complete_graph                       | 23.12                    |
| complete_multipartite_graph          | 23.12                    |
| connected_components                 | 23.12                    |
| core_number                          | 24.02                    |
| cubical_graph                        | 23.12                    |
| cycle_graph                          | 23.12                    |
| davis_southern_women_graph           | 23.12                    |
| degree_centrality                    | 23.12                    |
| desargues_graph                      | 23.12                    |
| descendants                          | 24.02                    |
| descendants_at_distance              | 24.02                    |
| diamond_graph                        | 23.12                    |
| dodecahedral_graph                   | 23.12                    |
| edge_betweenness_centrality          | 23.10                    |
| eigenvector_centrality               | 23.12                    |
| empty_graph                          | 23.12                    |
| florentine_families_graph            | 23.12                    |
| from_pandas_edgelist                 | 23.12                    |
| from_scipy_sparse_array              | 23.12                    |
| frucht_graph                         | 23.12                    |
| generic_bfs_edges                    | 24.02                    |
| heawood_graph                        | 23.12                    |
| hits                                 | 23.12                    |
| house_graph                          | 23.12                    |
| house_x_graph                        | 23.12                    |
| icosahedral_graph                    | 23.12                    |
| in_degree_centrality                 | 23.12                    |
| is_arborescence                      | 24.02                    |
| is_bipartite                         | 24.02                    |
| is_branching                         | 24.02                    |
| is_connected                         | 23.12                    |
| is_forest                            | 24.02                    |
| is_isolate                           | 23.10                    |
| is_strongly_connected                | 24.02                    |
| is_tree                              | 24.02                    |
| is_weakly_connected                  | 24.02                    |
| isolates                             | 23.10                    |
| k_truss                              | 23.12                    |
| karate_club_graph                    | 23.12                    |
| katz_centrality                      | 23.12                    |
| krackhardt_kite_graph                | 23.12                    |
| ladder_graph                         | 23.12                    |
| leiden                               | ?                        |
| les_miserables_graph                 | 23.12                    |
| lollipop_graph                       | 23.12                    |
| louvain_communities                  | 23.10                    |
| moebius_kantor_graph                 | 23.12                    |
| node_connected_component             | 23.12                    |
| null_graph                           | 23.12                    |
| number_connected_components          | 23.12                    |
| number_of_isolates                   | 23.10                    |
| number_strongly_connected_components | 24.02                    |
| number_weakly_connected_components   | 24.02                    |
| octahedral_graph                     | 23.12                    |
| out_degree_centrality                | 23.12                    |
| overall_reciprocity                  | 24.02                    |
| pagerank                             | 23.12                    |
| pappus_graph                         | 23.12                    |
| path_graph                           | 23.12                    |
| petersen_graph                       | 23.12                    |
| reciprocity                          | 24.02                    |
| reverse                              | 24.02                    |
| sedgewick_maze_graph                 | 23.12                    |
| single_source_shortest_path_length   | 23.12                    |
| single_target_shortest_path_length   | 23.12                    |
| star_graph                           | 23.12                    |
| strongly_connected_components        | 24.02                    |
| tadpole_graph                        | 23.12                    |
| tetrahedral_graph                    | 23.12                    |
| transitivity                         | 24.02                    |
| triangles                            | 24.02                    |
| trivial_graph                        | 23.12                    |
| truncated_cube_graph                 | 23.12                    |
| truncated_tetrahedron_graph          | 23.12                    |
| turan_graph                          | 23.12                    |
| tutte_graph                          | 23.12                    |
| uniform_neighbor_sample              | ?                        |
| weakly_connected_components          | 24.02                    |
| wheel_graph                          | 23.12                    |

To request nx-cugraph backend support for a NetworkX API that is not listed
above, visit the [cuGraph GitHub repo](https://github.com/rapidsai/cugraph).
