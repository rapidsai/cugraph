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

`cugraph::leiden` clusters an undirected graph by maximizing the generalized modularity
Q(gamma) = sum_c [L_c / 2m - gamma (K_c / 2m)^2] with the Leiden algorithm:

 * V.A. Traag, L. Waltman, N.J. van Eck: From Louvain to Leiden: guaranteeing well-connected communities, Scientific Reports 9, 5233 (2019), https://doi.org/10.1038/s41598-019-41695-z

The implementation:

 * **Iterations.** Each Leiden iteration (default 2; -1 iterates until the partition is stable) runs, at
   every level of the hierarchy, a synchronous local-moving phase in hashed sub-rounds, a refinement that splits every
   community into well-connected sub-communities along a spanning forest of its best edges, and an aggregation by the
   refined partition. Every iteration but the last ends with a V-cycle, and every iteration ends with a
   connected-components split.
 * **Guarantees.** Every returned community is connected. The returned modularity is the exact (fp64) modularity of the
   returned clustering. Labels are ordered by decreasing community size.
 * **Determinism.** Every decision is made in exact 64-bit fixed point (weights are quantized once), with total-order tie
   breaks and counter-based hashes of (seed, iteration, level, phase, sweep, vertex) instead of a random number
   generator; the remaining floating-point operations are explicitly rounded (`_rn` intrinsics, `--fmad=false`) and the
   modularity is summed in a fixed order. Identical inputs, parameters and seed give bitwise identical results,
   independently of thread scheduling, launch configuration, the vertex / edge id types, the weight type (for equal
   values) and the number of GPUs. By construction this holds across GPU architectures; `LEIDEN_TEST.GoldenValues` pins
   bitwise results and checks it on every architecture CI runs on.
 * **Edge semantics.** The graph must be symmetric (the data is checked). Weights must be finite and non-negative.
   Self-loops and zero-weight edges are ignored; parallel edges are summed (fp64, in ascending order of their weights).
 * **Memory and launches.** One workspace allocation per call (about 2.7x the CSR with int32 ids and float weights);
   nothing is allocated inside the iterations. Most kernels run as CUDA graphs (built once per call, replayed with
   updated arguments; with CUDA >= 12.4 the local-moving sweeps run in a conditional WHILE node), and the stream is
   synchronized a few times per level.
 * **Multi-GPU.** The current implementation all-gathers the edge list to every GPU and runs the single-GPU algorithm with
   the seed of rank 0 on every rank, so the result equals the single-GPU result on the same (internal) vertex ids for any
   number of GPUs. The graph must fit on one GPU.

Code layout:

 * `leiden_impl.cuh`: `cugraph::leiden` (single-GPU: the graph view's CSR is passed to the core as is; multi-GPU: the
   all-gathered edge list); `leiden_{sg,mg}_v*_e*.cu` instantiate it.
 * `detail/leiden/`: the core, compiled once into `libcugraph_common` and shared by the single-GPU and multi-GPU
   libraries. `leiden_csr.hpp` declares `detail::leiden_csr` (raw CSR) and `detail::leiden_coo` (edge list in any
   order); `leiden_csr_impl.cuh` holds the input check, the canonicalization of parallel edges (`canonicalize.cuh`), the
   workspace and the level-0 quantization (`kernels_ingest.cuh`); the type-independent engine (`numerics.cuh` fixed
   point and hashes, `arena.cuh` workspace layout, `graph.cuh` CUDA graphs, `kernels_scan.cuh` graph-friendly scans,
   `kernels_{move,refine,aggregate,final}.cuh`, `driver.cuh`) is compiled once in `driver_common.cu` (interface:
   `driver.hpp`).

The unit tests are the best place to look for examples: [SG](../../tests/community/leiden_test.cpp),
[MG](../../tests/community/mg_leiden_test.cpp).

```cpp
#include <cugraph/algorithms.hpp>
...
raft::random::RngState rng_state{seed};
rmm::device_uvector<vertex_t> clustering_v(graph_view.local_vertex_partition_range_size(), handle.get_stream());
cugraph::leiden_params_t params{};  // resolution 1, 2 iterations
auto result = cugraph::leiden(handle,
                              rng_state,
                              graph_view,
                              edge_weight_view,  // std::nullopt: unit weights
                              raft::device_span<vertex_t>{clustering_v.data(), clustering_v.size()},
                              params);
// result.num_clusters, result.modularity, result.num_iterations, result.num_levels
```

## ECG
