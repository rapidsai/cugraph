# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.utilities import ensure_cugraph_obj_for_nx
import cudf

from pylibcugraph.experimental import triangle_count as \
    pylibcugraph_triangle_count

from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          SGGraph
                          )


def EXPERIMENTAL__triangle_count(G, start_list=None):
    """
    Compute the number of triangles (cycles of length three) in the
    input graph.

    Parameters
    ----------
    G : cugraph.graph or networkx.Graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    start_list : not supported
        list of vertices for triangle count. if None the entire set of vertices
        in the graph is processed

    Returns
    -------
    result : cudf.DataFrame
        GPU data frame containing 2 cudf.Series

    ddf['vertex']: cudf.Series
            Contains the triangle counting vertices
    ddf['counts']: cudf.Series
        Contains the triangle counting counts

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv',
    ...                     delimiter = ' ',
    ...                     dtype=['int32', 'int32', 'float32'],
    ...                     header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr='2')
    >>> count = cugraph.experimental.triangle_count(G)

    """
    # FIXME: start_list is disabled
    start_list = None
    G, _ = ensure_cugraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)
            if start_list.dtype != 'int32':
                raise ValueError(f"'start_list' must have int32 values, "
                                 f"got: {start_list.dtype}")
        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                    f"'start_list' must be either a list or a cudf.Series,"
                    f"got: {start_list.dtype}")

        if G.renumbered is True:
            if isinstance(start_list, cudf.DataFrame):
                start_list = G.lookup_internal_vertex_id(
                    start_list, start_list.columns)
            else:
                start_list = G.lookup_internal_vertex_id(start_list)

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights']

    if srcs.dtype != 'int32':
        raise ValueError(f"Graph vertices must have int32 values, "
                         f"got: {srcs.dtype}")

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(
        is_symmetric=True, is_multigraph=G.is_multigraph())
    store_transposed = False

    # FIXME:  This should be based on the renumber parameter set when creating
    # the graph
    renumber = False
    do_expensive_check = False

    sg = SGGraph(resource_handle, graph_props, srcs, dsts, weights,
                 store_transposed, renumber, do_expensive_check)

    vertex, counts = pylibcugraph_triangle_count(
        resource_handle, sg, start_list, do_expensive_check)

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["counts"] = counts

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return df
