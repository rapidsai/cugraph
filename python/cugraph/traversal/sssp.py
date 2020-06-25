# Copyright (c) 2019 - 2020, NVIDIA CORPORATION.
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

from cugraph.traversal import sssp_wrapper
import numpy as np
import cudf


def sssp(G, source):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The distances column will store
    the distance from the source to each vertex. The predecessors column will
    store each vertex's predecessor in the shortest path. Vertices that are
    unreachable will have a distance of infinity denoted by the maximum value
    of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1. Graphs with negative weight cycles are not
    supported.

    Parameters
    ----------
    graph : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. Edge weights,
        if present, should be single or double precision floating point values.
    source : int
        Index of the source vertex.

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex.
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex.
        df['predecessor'][i] gives the vertex id of the vertex that was reached
        before the i'th vertex in the traversal.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> distances = cugraph.sssp(G, 0)
    """

    if G.renumbered is True:
        source = G.edgelist.renumber_map.to_vertex_id(cudf.Series([source]))[0]

    df = sssp_wrapper.sssp(G, source)

    if G.renumbered:
        # FIXME: multi-column vertex support
        tmp = G.edgelist.renumber_map.from_vertex_id(df['vertex'])
        df['vertex'] = tmp['0']
        df['predecessor'][df['predecessor'] > -1] = G.edgelist.renumber_map.\
                                                    from_vertex_id(df['predecessor'][df['predecessor'] >- 1])['0']

        #if isinstance(input_graph.edgelist.renumber_map, cudf.DataFrame): # Multicolumns renumbering
        #    n_cols = len(input_graph.edgelist.renumber_map.columns) - 1
        #    unrenumbered_df_ = df.merge(input_graph.edgelist.renumber_map, left_on='vertex', right_on='id', how='left').drop(['id', 'vertex'])
        #    unrenumbered_df = unrenumbered_df_.merge(input_graph.edgelist.renumber_map, left_on='predecessor', right_on='id', how='left').drop(['id', 'predecessor'])
        #    unrenumbered_df.columns = ['distance'] + ['vertex_' + str(i) for i in range(n_cols)] + ['predecessor_' + str(i) for i in range(n_cols)]
        #    cols = unrenumbered_df.columns.to_list()
        #    df = unrenumbered_df[cols[1:n_cols + 1] + [cols[0]] + cols[n_cols:]]

    return df


def filter_unreachable(df):
    """
    Remove unreachable vertices from the result of SSSP or BFS

    Parameters
    ----------
    df : cudf.DataFrame
        cudf.DataFrame that is the output of SSSP or BFS

    Returns
    -------
    df : filtered cudf.DataFrame with only reachable vertices
        df['vertex'][i] gives the vertex id of the i'th vertex.
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex.
        df['predecessor'][i] gives the vertex that was reached before the i'th
        vertex in the traversal.
    """
    if('distance' not in df):
        raise KeyError("No distance column found in input data frame")
    if(np.issubdtype(df['distance'].dtype, np.integer)):
        max_val = np.iinfo(df['distance'].dtype).max
        return df[df.distance != max_val]
    elif(np.issubdtype(df['distance'].dtype, np.inexact)):
        max_val = np.finfo(df['distance'].dtype).max
        return df[df.distance != max_val]
    else:
        raise TypeError("distance type unsupported")
