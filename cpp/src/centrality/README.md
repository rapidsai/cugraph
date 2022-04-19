# Centrality algorithms
cuGraph Pagerank is implemented using our graph primitive library

## Pagerank

The unit test code is the best place to search for examples on calling pagerank.

 * [SG Implementation](../../tests/pagerank/pagerank_test.cpp)
 * [MG Implementation](../../tests/pagerank/mg_pagerank_test.cpp)

## Simple pagerank

The example assumes that you create an SG or MG graph somehow.  The caller must create the pageranks vector in device memory and pass in the raw pointer to that vector into the pagerank function.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;       // or int64_t, whichever is appropriate
using weight_t = float;         // or double, whichever is appropriate
using result_t = weight_t;      // could specify float or double also
raft::handle_t handle;          // Must be configured if MG
auto graph_view = graph.view(); // assumes you have created a graph somehow

result_t constexpr alpha{0.85};
result_t constexpr epsilon{1e-6};

rmm::device_uvector<result_t> pageranks_v(graph_view.number_of_vertices(), handle.get_stream());

// pagerank optionally supports three additional parameters:
//     max_iterations     - maximum number of iterations, if pagerank doesn't coverge by
//                          then we abort
//     has_initial_guess  - if true, values in the pagerank array when the call is initiated
//                          will be used as the initial pagerank values.  These values will
//                          be normalized before use.  If false (the default), the values
//                          in the pagerank array will be set to 1/num_vertices before
//                          starting the computation.
//     do_expensive_check - perform extensive validation of the input data before
//                          executing algorithm.  Off by default.  Note: turning this on
//                          is expensive
cugraph::pagerank(handle, graph_view, nullptr, nullptr, nullptr, vertex_t{0},
                                pageranks_v.data(), alpha, epsilon);
```

## Personalized Pagerank

The example assumes that you create an SG or MG graph somehow.  The caller must create the pageranks vector in device memory and pass in the raw pointer to that vector into the pagerank function.  Additionally, the caller must create personalization_vertices and personalized_values vectors in device memory, populate them and pass in the raw pointers to those vectors.

```cpp
#include <cugraph/algorithms.hpp>
...
using vertex_t = int32_t;                    // or int64_t, whichever is appropriate
using weight_t = float;                      // or double, whichever is appropriate
using result_t = weight_t;                   // could specify float or double also
raft::handle_t handle;                       // Must be configured if MG
auto graph_view = graph.view();              // assumes you have created a graph somehow
vertex_t number_of_personalization_vertices; // Provided by caller

result_t constexpr alpha{0.85};
result_t constexpr epsilon{1e-6};

rmm::device_uvector<result_t> pageranks_v(graph_view.number_of_vertices(), handle.get_stream());
rmm::device_uvector<vertex_t> personalization_vertices(number_of_personalization_vertices, handle.get_stream());
rmm::device_uvector<result_t> personalization_values(number_of_personalization_vertices, handle.get_stream());

//  Populate personalization_vertices, personalization_values with user provided data

// pagerank optionally supports three additional parameters:
//     max_iterations     - maximum number of iterations, if pagerank doesn't coverge by
//                          then we abort
//     has_initial_guess  - if true, values in the pagerank array when the call is initiated
//                          will be used as the initial pagerank values.  These values will
//                          be normalized before use.  If false (the default), the values
//                          in the pagerank array will be set to 1/num_vertices before
//                          starting the computation.
//     do_expensive_check - perform extensive validation of the input data before
//                          executing algorithm.  Off by default.  Note: turning this on
//                          is expensive
cugraph::pagerank(handle, graph_view, nullptr, personalization_vertices.data(),
                                personalization_values.data(), number_of_personalization_vertices,
                                pageranks_v.data(), alpha, epsilon);
```
