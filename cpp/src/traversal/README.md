# Traversal
cuGraph traversal algorithms are contained in this directory

## SSSP

The unit test code is the best place to search for examples on calling louvain.

 * [SG Implementation](../../tests/experimental/sssp_test.cpp)
 * MG Implementation - TBD

## Simple SSSP

The example assumes that you create an SG or MG graph somehow.  The caller must create the pageranks vector in device memory and pass in the raw pointer to that vector into the pagerank function.

```cpp
#include <algorithm.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
using result_t = weight_t;      // could specify float or double also
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow
vertex_t source;                // Initialized by user

rmm::device_uvector<weight_t> distances_v(graph_view.get_number_of_vertices(), handle.get_stream());
rmm::device_uvector<vertex_t> predecessors_v(graph_view.get_number_of_vertices(), handle.get_stream());

cugraph::experimental::sssp(handle, graph_view, distances_v.begin(), predecessors_v.begin(), source, std::numeric_limits<weight_t>::max(), false);
```

## BFS
