# Copyright (c) 2021, NVIDIA CORPORATION.
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

def concurrent_bfs(Graphs, sources, depth_limit=None, offload=False):
    """
    Find the breadth first traversals of multiple graphs with multiple sources
    in each graph.

    Parameters
    ----------
    Graphs : list of cugraph.Graph or cugraph.DiGraph
        The adjacency lists will be computed if not already present.

    sources : list of cudf.Series
        For each graph, subset of vertices from which the traversals start. 
        A BFS is run in Graphs[i] for each source in the Series at sources[i].
        The size of this list must match the size of the graph list.
        The size of each Series (ie. the number of sources per graph) is flexible,
        but cannot exceed the size of the corresponding graph.


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
            Series containing the starting offset in the returned edge list for each source.

    If offload is True, or if the output does not fit in memory :
        Writes csv files containing BFS output to the disk.
    """
    if not isinstance(Graphs, list):
        raise TypeError(
                "Graphs should be a list of cugraph.Graph or cugraph.DiGraph"
            )
    if not isinstance(sources, list):
        raise TypeError(
                "sources should be a list of cudf.Series"
            )
    if len(Graphs) != len(sources):
        raise ValueError(
                "The size of the sources list must match the size of the graph list."
            )
    if offload is True:
        raise NotImplementedError(
            "Offloading is coming soon! Please up vote the github issue #1461
             to help us prioritize"
        )
    
    # Consolidate graphs in a single graph and record components

    # Renumber and concatenate sources in a single df 

    # Call multi_source_bfs


def multi_source_bfs(G, sources, components=None, depth_limit=None, offload=False):
    """
    Find the breadth first traversal from multiple sources in a graph.

    Parameters
    ----------
    G : cugraph.Graph or cugraph.DiGraph
        The adjacency list will be computed if not already present.

    sources :  cudf.Series
        Subset of vertices from which the traversals start. A BFS is run for
        each source in the Series. 
        The size of the series should be at least one and cannot exceed the size
        of the graph.

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

          df['distance_<source>'] path distance for each vertex from the starting vertex
          One column per source.

          df['predecessor_<source>'] for each i'th position in the column, the vertex ID
          immediately preceding the vertex at position i in the 'vertex' column
          One column per source.

    If G is a cugraph.Graph and component information is present returns :
        BFS_edge_lists : cudf.DataFrame
            GPU data frame containing all BFS edges
        source_offsets: cudf.Series
            Series containing the starting offset in the returned edge list for each source.

    If offload is True, or if the output does not fit in memory :
        Writes csv files containing BFS output to the disk.
    """
    if components is not None:
        null_check(components["vertex"])
        null_check(components["colors"])
    
    if depth_limit is not None:
        raise NotImplementedError(
            "depth limit implementation of BFS is not currently supported"
        )

    if offload is True:
        raise NotImplementedError(
            "Offloading is coming soon! Please up vote the github issue #1461
             to help us prioritize"
        )
    
    # Memory footprint check

    # Call multi_source_bfs


