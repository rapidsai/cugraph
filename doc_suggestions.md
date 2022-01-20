# Suggested documentation changes for cugraph/python/cugraph/cugraph

This markdown file serves to suggest changes in the documentation for the python implementation of cugraph. These notes are split up by the organization of the modules in the repository.

NOTE: These notes can be useful if attached to an issue

Key functions should have a complete documentation, which includes:

- a specific description of the function's purpose
- a list of all arguments the function accepts with information about what types are allowed and how the function changes as the argument changes
- a description of the return value(s) with their corresponding type(s)
- at least 1 example, with an example output if possible Helper functions need not have a complete documentation, though some sort of description is still necessary

Currently, there are some tests that have been commented out because of various reasons. They are the following tests:

- Example in `hungarian` from linear_assignment/lap.py, because a bipartite graph is required (and isn't included in the dataset)
- Example in `pagerank` from dask/link_analysis/pagerank.py (due to initializing a DASK cluster)
- Both examples in `import_optional` from utilities/utils.py, because the first example works only if networkx isn't imported, and the second works only if cudf isn't imported

# Comments about html documentation on docs.rapids.ai/api/cugraph 

- Under `Components`, both cugraph.dask.components.connectivity.call_wcc and cugraph.dask.components.connectivity.weakly_connected_components don't have description (nor information about arguments, returns, and examples)

- Under `Cores`, the link for the K-Core algorithm points to cugraph.centrality.katz_centrality.katz_centrality instead of cugraph.cores.k_core.k_core

- cugraph.dask.link_analysis.pagerank.pagerank appears twice, under `Link Analysis` and `Link Prediction`

- cugraph.centrality.katz_centrality.katz_centrality appears 3 times, under `Centrality`, `Cores`, and `Link Analysis`

- Under `Link Prediction`, the algorithms cugraph.link_prediction.sorensen.sorensen and cugraph.link_prediction.wsorensen.sorensen_w aren't present

## centrality

### centrality/betweenness_centrality.py

### centrality/katz_centrality.py

## comms

### comms/comms.py

## community

### community/ecg.py

### community/egonet.py

- `ego_graph`'s example uses variable seed, which is different from parameter n. For now, this example was made to be partially commented out

### community/ktruss_subgraph.py

### community/leiden.py

### community/louvain.py

### community/spectral_clustering.py

### community/subgraph_extraction.py

### community/triangle_count.py

## components

### components/connectivity.py

## cores

### cores/core_number.py

### cores/k_core.py

## dask

### dask/centrality

#### dask/centrality/katz_centrality.py

### dask/common

#### dask/common/input_utils.py

#### dask/common/mg_utils.py

#### dask/common/part_utils.py

#### dask/common/read_utils.py

### dask/community

#### dask/community/louvain.py

### dask/components

#### dask/components/connectivity.py

### dask/link_analysis

#### dask/link_analysis/pagerank.py

### dask/structure

### dask/traversal

#### dask/traversal/bfs.py

#### dask/traversal/sssp.py

## generators

### generators/rmat.py

## layout

### layout/force_atlas.py

## linear_assignment

### linear_assignment/lap.py
- `hungarian`'s example requires a bipartite graph, but current dataset lacks this, this is also shown in the `test_hungarian.py`. Current example has been commented out

## link_analysis

### link_analysis/hits.py

### link_analysis/pagerank.py

## link_prediction

### link_prediction/jaccard.py

### link_prediction/overlap.py

### link_prediction/sorensen.py

### link_prediction/wjaccard.py

### link_prediction/woverlap.py

### link_prediction/wsorensen.py

## proto

### proto/components

#### proto/components/scc.py

### proto/structure

#### proto/structure/bicliques.py

## sampling

### sampling/random_walks.py

## structure

### structure/graph_implementation

#### structure/graph_implementation/npartiteGraph.py

#### structure/graph_implementation/simpleDistributedGraph.py

#### structure/graph_implementation/simpleGraph.py

### structure/convert_matrix.py

### structure/graph_classes.py

- Examples from `cugraph.Graph` not in-line with Parameters and Methods (RESOLVED)

### structure/hypergraph.py

### structure/number_map.py

### structure/shuffle.py

### structure/symmetrize.py

## traversal

### traversal/bfs.py

### traversal/ms_bfs.py

### traversal/sssp.py

## tree

### tree/minimum_spanning_tree.py

## utilities - could NOT make all tests passing

### utilities/grmat.py

### utilities/nx_factory.py

### utilities/path_retrieval.py

### utilities/utils.py

- `import_optional`'s examples require not knowing networkx nor cudf...