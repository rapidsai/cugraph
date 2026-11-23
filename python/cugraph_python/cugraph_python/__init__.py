# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cugraph_python: nanobind prototype for cuGraph's Python construction, BFS, PageRank, and sampling.

Status: prototype. Builds against this checkout's libcugraph and passes the GPU tests in
cugraph_python/tests/.

Construction
    context = cg.Context()
    coo = cg.make_cycle_coo(context, size=100_000)   # native producer, owns RMM buffers
    graph, properties = cg.create_graph(coo)         # consumes coo; properties is empty

    cp.cuda.get_current_stream().synchronize()
    coo = cg.copy_from_arrays(context, sources, destinations, weights)  # copies; inputs stay valid
    graph, properties = cg.create_graph(coo, store_transposed=True)

    - Vertex and edge IDs are int32 or int64, and the edge type matches the vertex type.
      store_transposed=False builds the orientation BFS uses. store_transposed=True builds the
      orientation PageRank uses. multi_gpu=True is rejected. Graphs are directed and simple.
    - weights is an optional float32 or float64 array, one value per edge. edge_properties is an
      optional list of int32, int64, float32, or float64 arrays in caller order. The returned
      property vector keeps that order.
    - Edge properties hold the context and the graph identity. An algorithm rejects a property
      that was built for a different graph.
    - A second create_graph on the same COO raises RuntimeError, including through aliases.
    - Factories and create_graph synchronize before returning.

Graph
    - Python aliases share one object; assignment copies nothing.
    - graph.number_of_vertices() and graph.number_of_edges() read metadata from native storage.
    - Destroying the last reference frees native storage before the context reference.
    - Edge properties are not stored on the graph. Dropping the graph leaves them valid, and
      dropping the properties leaves the graph valid.

Vertex preparation and BFS
    starts = cg.prepare_vertices(graph, [10])        # external -> internal, sorted, unique
    vertices, distances, predecessors = cg.bfs(graph, starts, compute_predecessors=True)
    cp.from_dlpack(vertices)                         # external IDs, number-map order
    cp.from_dlpack(distances)                        # aligned with vertices
    cp.from_dlpack(predecessors)                     # aligned; None unless requested

    Return contract:
    - vertices[i] is the external ID of internal vertex i. The order is the number map's
      order, which is unspecified but fixed for a given graph.
    - distances[i] is the hop count from the nearest start. Unreached vertices hold
      the maximum value of the vertex integer type (cugraph's invalid distance).
    - predecessors[i] is the external ID of the BFS parent. Starts and unreached vertices hold
      -1 (cugraph's invalid vertex ID).
    - The arrays are owned copies. Each DLPack export owns its device buffer, so arrays
      stay valid after the tuple, the graph, or the context is dropped.
    - prepare_vertices and bfs throw ValueError when a PreparedVertices was made for a
      different graph, numbering map, partition, or version.
    - BFS rejects a graph created with store_transposed=True.

PageRank
    vertices, pageranks, iterations, converged = cg.pagerank(graph, properties[0])
    cp.from_dlpack(pageranks)                    # aligned with vertices

    - The graph must be created with store_transposed=True. That is the orientation the current
      C++ PageRank API reads.
    - vertices[i] is the external ID of internal vertex i. pageranks[i] is that vertex's score.
    - iterations and converged come from the algorithm metadata.
    - Pass the weight property explicitly. Omitting it treats every edge weight as 1 and returns
      float32 scores. A float32 property returns float32 scores, and a float64 property returns
      float64 scores. No personalization vector.

Neighbor sampling
    rng = cg.RngState(0)
    options = cg.SamplingOptions(return_hops=True, with_replacement=False)
    sources, destinations, hops, sampled, offsets = cg.neighbor_sample(
        graph, rng, starting_vertices=[10], fan_out=[2, 1], edge_properties=properties,
        sampling_options=options)
    options = cg.SamplingOptions(
        return_hops=True, with_replacement=False, disjoint_sampling=True,
        temporal_sampling_comparison="strictly_increasing")
    sources, destinations, hops, sampled, offsets = cg.neighbor_sample(
        graph, rng, edge_start_times=properties[0], starting_vertices=[10],
        starting_vertex_start_times=[0], fan_out=[2, 2], edge_properties=properties,
        sampling_options=options)

    - Parameter order matches cugraph::neighbor_sample. RngState is the rng_state argument and
      is updated by each call. SamplingOptions matches sampling_options_t, including its
      defaults (with_replacement=True, return_hops=False).
    - The graph must be created with store_transposed=False.
    - sources and destinations are external IDs, one pair per sampled edge. hops is aligned,
      or None when return_hops is false. offsets is None unless starting_vertex_labels is set.
    - sampled[i] holds the values of properties[i] on the sampled edges, in the same order.
    - A temporal comparison requires edge_start_times and disjoint_sampling. Heterogeneous
      sampling sets num_edge_types and requires edge_types.

Open verification items (check when compiled):
    - nanobind shared_ptr holder behavior for Context and for the Graph's Context reference.
    - Sentinel values (-1 predecessor, int32 max distance) for starts and unreached vertices.
    - Moves of cugraph::graph_t inside the Graph aggregate.
"""

# If libcugraph was installed as a wheel, load its symbols before the extension. A conda
# install puts libcugraph.so on the loader path, so the import is optional.
try:
    import libcugraph
except ModuleNotFoundError:
    pass
else:
    libcugraph.load_library()
    del libcugraph

from cugraph_python import _native

__all__ = [
    "Context",
    "make_cycle_coo",
    "copy_from_arrays",
    "create_graph",
    "prepare_vertices",
    "bfs",
    "pagerank",
    "neighbor_sample",
    "SamplingOptions",
    "RngState",
    "DeviceArray",
    "EdgeProperties",
    "EdgeProperty",
]


Context = _native.Context
RngState = _native.RngState
DeviceArray = _native.DeviceArray
EdgeProperties = _native.EdgeProperties
EdgeProperty = _native.EdgeProperty


def make_cycle_coo(context, size):
    """Allocate and fill a directed cycle 0 -> 1 -> ... -> size-1 -> 0 on the device."""
    return _native.make_cycle_coo(context, size)


def copy_from_arrays(context, sources, destinations, weights=None, *, edge_properties=None):
    """Copy device arrays of int32 or int64 endpoints into an owned COO.

    Both endpoint arrays must have the same dtype. weights, when given, is one float32 or
    float64 value per edge and becomes the only edge property. edge_properties, when given, is
    a list of int32, int64, float32, or float64 arrays in caller order, one value per edge.
    Pass weights or edge_properties, not both. The caller must synchronize the producing stream
    first. The input arrays are not consumed.
    """
    if weights is not None and edge_properties is not None:
        raise ValueError("copy_from_arrays: pass weights or edge_properties, not both")
    if weights is not None:
        kind = getattr(getattr(weights, "dtype", None), "kind", None)
        itemsize = getattr(getattr(weights, "dtype", None), "itemsize", None)
        if kind != "f" or itemsize not in (4, 8):
            raise ValueError("copy_from_arrays: weights must be float32 or float64")
        edge_properties = [weights]
    return _native.copy_from_arrays(context, sources, destinations, edge_properties)


def create_graph(coo, *, store_transposed=False, multi_gpu=False):
    """Build a graph from an OwnedCOO, consuming it.

    Returns (graph, properties). properties is in the order the COO carried them, and is empty
    when the COO had none. The vertex dtype of the COO selects int32 or int64. The edge type
    matches. store_transposed=False is the BFS orientation. store_transposed=True is the PageRank
    orientation. multi_gpu=True is rejected and does not consume the COO.
    """
    return _native.create_graph(coo, store_transposed, multi_gpu)


def prepare_vertices(graph, external_ids):
    """Translate external vertex IDs to internal IDs for this graph."""
    return _native.prepare_vertices(graph, [int(v) for v in external_ids])


def bfs(graph, starts, *, compute_predecessors=False):
    """Breadth-first search from prepared start vertices.

    Returns (vertices, distances, predecessors). predecessors is None unless requested.
    vertices[i] is the external ID of internal vertex i, in number-map order.
    The graph must be created with store_transposed=False.
    """
    return _native.bfs(graph, starts, compute_predecessors)


def pagerank(graph, weights=None, *, alpha=0.85, epsilon=1.0e-5, max_iterations=500):
    """PageRank on a graph created with store_transposed=True.

    Returns (vertices, pageranks, iterations, converged). vertices[i] is the external ID of
    internal vertex i. pageranks[i] is the score of that vertex. weights is an EdgeProperty from
    this graph, or None. The score dtype matches that property, and is float32 when weights is
    omitted.
    """
    return _native.pagerank(graph, weights, float(alpha), float(epsilon), int(max_iterations))


def _enum_value(names, value, what):
    try:
        return names[value]
    except KeyError as error:
        raise ValueError(f"SamplingOptions: unknown {what} {value!r}") from error


def _optional_ints(values):
    return None if values is None else [int(value) for value in values]


class SamplingOptions:
    """Flags for neighbor_sample. Fields and defaults match cugraph::sampling_options_t."""

    def __init__(
        self,
        prior_sources_behavior="default",
        return_hops=False,
        dedupe_sources=False,
        with_replacement=True,
        temporal_sampling_comparison="none",
        disjoint_sampling=False,
        neighbor_selection="random",
        fixed_window=False,
    ):
        options = _native.SamplingOptions()
        options.prior_sources_behavior = _enum_value(
            {
                "default": _native.PriorSourcesBehavior.DEFAULT,
                "carry_over": _native.PriorSourcesBehavior.CARRY_OVER,
                "exclude": _native.PriorSourcesBehavior.EXCLUDE,
            },
            prior_sources_behavior,
            "prior_sources_behavior",
        )
        options.return_hops = bool(return_hops)
        options.dedupe_sources = bool(dedupe_sources)
        options.with_replacement = bool(with_replacement)
        options.temporal_sampling_comparison = _enum_value(
            {
                "none": _native.TemporalSamplingComparison.NONE,
                "strictly_increasing": _native.TemporalSamplingComparison.STRICTLY_INCREASING,
                "monotonically_increasing": (
                    _native.TemporalSamplingComparison.MONOTONICALLY_INCREASING
                ),
                "strictly_decreasing": _native.TemporalSamplingComparison.STRICTLY_DECREASING,
                "monotonically_decreasing": (
                    _native.TemporalSamplingComparison.MONOTONICALLY_DECREASING
                ),
            },
            temporal_sampling_comparison,
            "temporal_sampling_comparison",
        )
        options.disjoint_sampling = bool(disjoint_sampling)
        options.neighbor_selection = _enum_value(
            {
                "random": _native.NeighborSelection.RANDOM,
                "last": _native.NeighborSelection.LAST,
            },
            neighbor_selection,
            "neighbor_selection",
        )
        options.fixed_window = bool(fixed_window)
        self._native = options


def neighbor_sample(
    graph,
    rng_state,
    edge_weights=None,
    edge_ids=None,
    edge_types=None,
    edge_start_times=None,
    edge_end_times=None,
    edge_biases=None,
    starting_vertices=None,
    starting_vertex_start_times=None,
    starting_vertex_end_times=None,
    starting_vertex_labels=None,
    label_to_output_comm_rank=None,
    fan_out=None,
    num_edge_types=None,
    edge_properties=None,
    sampling_options=None,
    do_expensive_check=False,
):
    """Neighbor sample from external starting vertices.

    Returns (sources, destinations, hops, sampled_properties, offsets). The parameter order
    matches cugraph::neighbor_sample. rng_state is an RngState and is updated in place.
    starting_vertices and fan_out are required.
    edge_properties is the caller-order vector from create_graph. sampling_options is a
    SamplingOptions; the default matches cugraph::sampling_options_t.

    sources[i] and destinations[i] are external IDs of one sampled edge. hops[i] is the hop
    that produced it, or None when return_hops is false. sampled_properties[k] is the values
    of edge_properties[k] on those edges. offsets is the label offsets as int64, or None when
    starting_vertex_labels is omitted. The graph must be created with store_transposed=False.

    starting_vertex_start_times is one window lower bound per starting vertex, and is the hop-0
    frontier time for an increasing walk. starting_vertex_end_times is the matching upper bound.
    A temporal comparison requires edge_start_times and disjoint_sampling. num_edge_types selects
    heterogeneous sampling and requires edge_types; fan_out then has one value per (hop, edge
    type). label_to_output_comm_rank is rejected for single-GPU sampling.
    """
    if starting_vertices is None or fan_out is None:
        raise TypeError("neighbor_sample requires starting_vertices and fan_out")
    if sampling_options is None:
        sampling_options = SamplingOptions()
    return _native.neighbor_sample(
        graph,
        rng_state,
        edge_weights,
        edge_ids,
        edge_types,
        edge_start_times,
        edge_end_times,
        edge_biases,
        [int(vertex) for vertex in starting_vertices],
        _optional_ints(starting_vertex_start_times),
        _optional_ints(starting_vertex_end_times),
        _optional_ints(starting_vertex_labels),
        _optional_ints(label_to_output_comm_rank),
        [int(fan) for fan in fan_out],
        None if num_edge_types is None else int(num_edge_types),
        edge_properties,
        sampling_options._native,
        bool(do_expensive_check),
    )
