# Traversal
cuGraph traversal algorithms are contained in this directory

## SSSP

The unit test code is the best place to search for examples on calling SSSP.

 * [SG Implementation](../../tests/traversal/sssp_test.cpp)
 * [MG Implementation](../../tests/traversal/mg_sssp_test.cpp)

## Simple SSSP

The example assumes that you create an SG or MG graph somehow.  The caller must create the distances and predecessors vectors in device memory and pass in the raw pointers to those vectors into the SSSP function.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
using result_t = weight_t;      // could specify float or double also
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow
vertex_t source;                // Initialized by user

rmm::device_uvector<weight_t> distances_v(graph_view.number_of_vertices(), handle.get_stream());
rmm::device_uvector<vertex_t> predecessors_v(graph_view.number_of_vertices(), handle.get_stream());

cugraph::sssp(handle, graph_view, distances_v.begin(), predecessors_v.begin(), source, std::numeric_limits<weight_t>::max(), false);
```

## BFS

The unit test code is the best place to search for examples on calling BFS.

 * [SG Implementation](../../tests/traversal/bfs_test.cpp)
 * [MG Implementation](../../tests/traversal/mg_bfs_test.cpp)

## Simple BFS

The example assumes that you create an SG or MG graph somehow.  The caller must create the distances and predecessors vectors in device memory and pass in the raw pointers to those vectors into the BFS function.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
using result_t = weight_t;      // could specify float or double also
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow
vertex_t source;                // Initialized by user

rmm::device_uvector<weight_t> distances_v(graph_view.number_of_vertices(), handle.get_stream());
rmm::device_uvector<vertex_t> predecessors_v(graph_view.number_of_vertices(), handle.get_stream());

cugraph::bfs(handle, graph_view, d_distances.begin(), d_predecessors.begin(), source, false, std::numeric_limits<vertex_t>::max(), false);
```
