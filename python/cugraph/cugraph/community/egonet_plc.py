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

from pylibcugraph import triangle_count as pylibcugraph_triangle_count

from pylibcugraph import ResourceHandle


def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    n : int, list or cudf.Series, cudf.DataFrame
        A node or a list or cudf.Series of nodes or a cudf.DataFrame if nodes
        are represented with multiple columns. If a cudf.DataFrame is provided,
        only the first row is taken as the node input.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    undirected: bool, optional
        Defaults to False. True is not supported

    distance: key, optional (default=None)
        Distances are counted in hops from n. Other cases are not supported.

    Returns
    -------
    G_ego : cuGraph.Graph or networkx.Graph
        A graph descriptor with a minimum spanning tree or forest.
        The networkx graph will not have all attributes copied over

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> ego_graph = cugraph.ego_graph(G, 1, radius=2)

    """
    (G, input_type) = ensure_cugraph_obj(G, nx_weight_attr="weight")
    result_graph = type(G)()

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    if start_list is not None:
        if isinstance(start_list, int):
            start_list = [start_list]
        if isinstance(start_list, list):
            start_list = cudf.Series(start_list)

        if not isinstance(start_list, cudf.Series):
            raise TypeError(
                f"'start_list' must be either a list or a cudf.Series,"
                f"got: {start_list.dtype}"
            )

        if G.renumbered is True:
            if isinstance(start_list, cudf.DataFrame):
                start_list = G.lookup_internal_vertex_id(start_list, start_list.columns)
            else:
                start_list = G.lookup_internal_vertex_id(start_list)

    do_expensive_check = False

    vertex, counts = pylibcugraph_triangle_count(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        start_list=start_list,
        do_expensive_check=do_expensive_check,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["counts"] = counts

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return df
