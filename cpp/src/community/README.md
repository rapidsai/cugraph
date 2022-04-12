# Louvain and Related Clustering Algorithms
cuGraph contains a GPU implementation of the Louvain algorithm and several related clustering algorithms (Leiden and ECG).

## Louvain

The Louvain implementation is designed to assign clusters attempting to optimize modularity.  The algorithm is derived from the serial implementation described in the following paper:

 * VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of community hierarchies in large networks, J Stat Mech P10008 (2008), http://arxiv.org/abs/0803.0476

It leverages some parallelism ideas from the following paper:
 * Hao Lu, Mahantesh Halappanavar, Ananth Kalyanaraman: Parallel heuristics for scalable community detection, Elsevier Parallel Computing (2015), https://www.sciencedirect.com/science/article/pii/S0167819115000472


The challenge in parallelizing Louvain lies in the primary loop which visits the vertices in serial.  For each vertex v the change in modularity is computed for moving the vertex from its currently assigned cluster to each of the clusters to which v's neighbors are assigned.  The largest positive delta modularity is used to select a new cluster (if there are no positive delta modularities then the vertex is not moved).  If the vertex v is moved to a new cluster then the statistics of the vertex v's old cluster and new cluster change.  This change in cluster statistics may affect the delta modularity computations of all vertices that follow vertex v in the serial iteration, creating a dependency between the different iterations of the loop.

In order to make efficient use of the GPU parallelism, the cuGraph implementation computes the delta modularity for *all* vertex/neighbor pairs using the *current* vertex assignment.  Decisions on moving vertices will be made based upon these delta modularities.  This will potentially make choices that the serial version would not make.  In order to minimize some of the negative effects of this (as described in the Lu paper), the cuGraph implementation uses an Up/Down technique.  In even numbered iterations a vertex can only move from cluster i to cluster j if i > j; in odd numbered iterations a vertex can only move from cluster i to cluster j if i < j.  This prevents two vertices from swapping clusters in the same iteration of the loop.  We have had great success in converging on high modularity clustering using this technique.

## Calling Louvain

The unit test code is the best place to search for examples on calling louvain.

 * [SG Implementation](../../tests/community/louvain_test.cpp)
 * [MG Implementation](../../tests/community/mg_louvain_test.cpp)

The API itself is very simple.  There are two variations:
 * Return a flat clustering
 * Return a Dendrogram

### Return a flat clustering

The example assumes that you create an SG or MG graph somehow.  The caller must create the clustering vector in device memory and pass in the raw pointer to that vector into the louvain function.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow

size_t level;
weight_t modularity;

rmm::device_uvector<vertex_t> clustering_v(graph_view.number_of_vertices(), handle.get_stream());

// louvain optionally supports two additional parameters:
//     max_level - maximum level of the Dendrogram
//     resolution - constant in the modularity computation
std::tie(level, modularity) = cugraph::louvain(handle, graph_view, clustering_v.data());
```

### Return a Dendrogram

The Dendrogram represents the levels of hierarchical clustering that the Louvain algorithm computes.  There is a separate function that will flatten the clustering into the same result as above.  Returning the Dendrogram, however, provides a finer level of detail on the intermediate results which can be helpful in more fully understanding the data.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow

cugraph::Dendrogram dendrogram;
weight_t modularity;

// louvain optionally supports two additional parameters:
//     max_level - maximum level of the Dendrogram
//     resolution - constant in the modularity computation
std::tie(dendrogram, modularity) = cugraph::louvain(handle, graph_view);

//  This will get the equivalent result to the earlier example
rmm::device_uvector<vertex_t> clustering_v(graph_view.number_of_vertices(), handle.get_stream());
cugraph::flatten_dendrogram(handle, graph_view, dendrogram, clustering.data());
```

## Leiden

## ECG
