# Copyright (c) 2021-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cudf

import warnings


def _get_feasibility(G, sources, components=None, depth_limit=None):
    """
    Evaluate the feasibility for breadth first traversal from multiple sources
    in a graph.

    Parameters
    ----------
    G : cugraph.Graph
        The adjacency list will be computed if not already present.

    sources :  cudf.Series
        Subset of vertices from which the traversals start. A BFS is run for
        each source in the Series.
        The size of the series should be at least one and cannot exceed
        the size of the graph.

    depth_limit : Integer, optional, default=None
        Limit the depth of the search. Terminates if no more vertices are
        reachable within the distance of depth_limit

    components : cudf.DataFrame, optional, default=None
        GPU Dataframe containing the component information.
        Passing this information may impact the return type.
        When no component information is passed BFS uses one component
        behavior settings.

        components['vertex'] : cudf.Series
            vertex IDs
        components['color'] : cudf.Series
            component IDs/color for vertices.

    Returns
    -------
    mem_footprint : integer
        Estimated memory foot print size in Bytes
    """

    # Fixme not implemented in RMM yet
    # using 96GB upper bound for now
    # mem = get_device_memory_info()
    mem = 9.6e10
    n_sources = sources.size
    V = G.number_of_vertices()
    E = G.number_of_edges()
    mean_component_sz = V
    n_components = 1

    # Retreive types
    size_of_v = 4
    size_of_e = 4
    size_of_w = 0
    if G.adjlist.weights is not None:
        if G.adjlist.weights.dtype is np.float64:
            size_of_w = 8
        else:
            size_of_w = 4
    if G.adjlist.offsets.dtype is np.float64:
        size_of_v = 8
    if G.adjlist.indices.dtype is np.float64:
        size_of_e = 8

    # Graph size
    G_sz = E * size_of_e + E * size_of_w + V * size_of_v

    # The impact of depth limit depends on the sparsity
    # pattern and diameter. We cannot leverage it without
    # traversing the full dataset a the moment.

    # dense output
    output_sz = n_sources * 2 * V * size_of_v

    # sparse output
    if components is not None:
        tmp = components["color"].value_counts()
        n_components = tmp.size
        if n_sources / n_components > 100:
            warnings.warn("High number of seeds per component result in large output.")
        mean_component_sz = tmp.mean()
        output_sz = mean_component_sz * n_sources * 2 * size_of_e

    # counting 10% for context, handle and temporary allocations
    mem_footprint = (G_sz + output_sz) * 1.1
    if mem_footprint > mem:
        warnings.warn(f"Cannot execute in-memory :{mem_footprint} Bytes")

    return mem_footprint


def concurrent_bfs(Graphs, sources, depth_limit=None, offload=False):
    """
    Find the breadth first traversals of multiple graphs with multiple sources
    in each graph.

    Parameters
    ----------
    Graphs : list of cugraph.Graph
        The adjacency lists will be computed if not already present.

    sources : list of cudf.Series
        For each graph, subset of vertices from which the traversals start.
        A BFS is run in Graphs[i] for each source in the Series at sources[i].
        The size of this list must match the size of the graph list.
        The size of each Series (ie. the number of sources per graph)
        is flexible, but cannot exceed the size of the corresponding graph.


    depth_limit : Integer, optional, default=None
        Limit the depth of the search. Terminates if no more vertices are
        reachable within the distance of depth_limit

    offload : boolean, optional, default=False
        Indicates if output should be written to the disk.
        When not provided, the algorithms decides if offloading is needed
        based on the input parameters.

    Returns
    -------
    Return type is decided based on the input parameters (size of
    sources, size of the graph, number of graphs and offload setting)

    If G is a cugraph.Graph and output fits in memory:
        BFS_edge_lists : cudf.DataFrame
            GPU data frame containing all BFS edges
        source_offsets: cudf.Series
            Series containing the starting offset in the returned edge list
            for each source.

    If offload is True, or if the output does not fit in memory :
        Writes csv files containing BFS output to the disk.
    """
    raise NotImplementedError(
        "concurrent_bfs is coming soon! Please up vote the github issue 1465\
             to help us prioritize"
    )
    if not isinstance(Graphs, list):
        raise TypeError("Graphs should be a list of cugraph.Graph")
    if not isinstance(sources, list):
        raise TypeError("sources should be a list of cudf.Series")
    if len(Graphs) != len(sources):
        raise ValueError(
            "The size of the sources list must match\
             the size of the graph list."
        )
    if offload is True:
        raise NotImplementedError(
            "Offloading is coming soon! Please up vote the github issue 1461\
             to help us prioritize"
        )

    # Consolidate graphs in a single graph and record components

    # Renumber and concatenate sources in a single df

    # Call multi_source_bfs
    # multi_source_bfs(
    #    G,
    #    sources,
    #    components=components,
    #    depth_limit=depth_limit,
    #    offload=offload,
    # )


def multi_source_bfs(G, sources, components=None, depth_limit=None, offload=False):
    """
    Find the breadth first traversal from multiple sources in a graph.

    Parameters
    ----------
    G : cugraph.Graph
        The adjacency list will be computed if not already present.

    sources :  cudf.Series
        Subset of vertices from which the traversals start. A BFS is run for
        each source in the Series.
        The size of the series should be at least one and cannot exceed the
        size of the graph.

    depth_limit : Integer, optional, default=None
        Limit the depth of the search. Terminates if no more vertices are
        reachable within the distance of depth_limit

    components : cudf.DataFrame, optional, default=None
        GPU Dataframe containing the component information.
        Passing this information may impact the return type.
        When no component information is passed BFS uses one component
        behavior settings.

        components['vertex'] : cudf.Series
            vertex IDs
        components['color'] : cudf.Series
            component IDs/color for vertices.

    offload : boolean, optional, default=False
        Indicates if output should be written to the disk.
        When not provided, the algorithms decides if offloading is needed
        based on the input parameters.

    Returns
    -------
    Return value type is decided based on the input parameters (size of
    sources, size of the graph, number of components and offload setting)
    If G is a cugraph.Graph, returns :
       cudf.DataFrame
          df['vertex'] vertex IDs

          df['distance_<source>'] path distance for each vertex from the
          starting vertex. One column per source.

          df['predecessor_<source>'] for each i'th position in the column,
          the vertex ID immediately preceding the vertex at position i in
          the 'vertex' column. One column per source.

    If G is a cugraph.Graph and component information is present returns :
        BFS_edge_lists : cudf.DataFrame
            GPU data frame containing all BFS edges
        source_offsets: cudf.Series
            Series containing the starting offset in the returned edge list
            for each source.

    If offload is True, or if the output does not fit in memory :
        Writes csv files containing BFS output to the disk.
    """
    raise NotImplementedError(
        "concurrent_bfs is coming soon! Please up vote the github issue 1465\
             to help us prioritize"
    )
    # if components is not None:
    #    null_check(components["vertex"])
    #    null_check(components["colors"])
    #
    # if depth_limit is not None:
    #    raise NotImplementedError(
    #        "depth limit implementation of BFS is not currently supported"
    #    )

    # if offload is True:
    #    raise NotImplementedError(
    #        "Offloading is coming soon! Please up vote the github issue 1461
    #         to help us prioritize"
    #    )
    if isinstance(sources, list):
        sources = cudf.Series(sources)
    if G.renumbered is True:
        sources = G.lookup_internal_vertex_id(cudf.Series(sources))
    if not G.adjlist:
        G.view_adj_list()
    # Memory footprint check
    footprint = _get_feasibility(
        G, sources, components=components, depth_limit=depth_limit
    )
    print(footprint)
    # Call multi_source_bfs
    # FIXME remove when implemented
    # raise NotImplementedError("Commming soon")
