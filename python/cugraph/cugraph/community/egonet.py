# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from cugraph.utilities import (
    ensure_cugraph_obj,
    is_nx_graph_type,
)
from cugraph.utilities import cugraph_to_nx

import cudf

from pylibcugraph import ego_graph as pylibcugraph_ego_graph

from pylibcugraph import ResourceHandle
import warnings


def _convert_graph_to_output_type(G, input_type):
    """
    Given a cugraph.Graph, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if is_nx_graph_type(input_type):
        return cugraph_to_nx(G)

    else:
        return G


def _convert_df_series_to_output_type(df, offsets, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if is_nx_graph_type(input_type):
        return df.to_pandas(), offsets.values_host.tolist()

    else:
        return df, offsets


def ego_graph(G, n, radius=1, center=True, undirected=None, distance=None):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    n : integer or list, cudf.Series, cudf.DataFrame
        A single node as integer or a cudf.DataFrame if nodes are
        represented with multiple columns. If a cudf.DataFrame is provided,
        only the first row is taken as the node input.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    undirected: bool, optional
        This parameter is here for NetworkX compatibility and is ignored

    distance: key, optional (default=None)
        This parameter is here for NetworkX compatibility and is ignored

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

    result_graph = type(G)(directed=G.is_directed())

    if undirected is not None:
        warning_msg = (
            "The parameter 'undirected' is deprecated and "
            "will be removed in the next release"
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

    if isinstance(n, (int, list)):
        n = cudf.Series(n)
    if isinstance(n, cudf.Series):
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n)
    elif isinstance(n, cudf.DataFrame):
        if G.renumbered is True:
            n = G.lookup_internal_vertex_id(n, n.columns)
    else:
        raise TypeError(
            f"'n' must be either an integer or a list or a cudf.Series"
            f" or a cudf.DataFrame, got: {type(n)}"
        )

    # Match the seed to the vertex dtype
    n_type = G.edgelist.edgelist_df["src"].dtype
    n = n.astype(n_type)
    do_expensive_check = False

    source, destination, weight, _ = pylibcugraph_ego_graph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source_vertices=n,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    df["weight"] = weight

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)
    else:
        # FIXME: THe original 'src' and 'dst' are not stored in 'simpleGraph'
        src_names = "src"
        dst_names = "dst"

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)
    return _convert_graph_to_output_type(result_graph, input_type)


def batched_ego_graphs(G, seeds, radius=1, center=True, undirected=None, distance=None):
    """
    Compute the induced subgraph of neighbors for each node in seeds
    within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    seeds : cudf.Series or list or cudf.DataFrame
        Specifies the seeds of the induced egonet subgraphs.

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
    ego_edge_lists : cudf.DataFrame or pandas.DataFrame
        GPU data frame containing all induced sources identifiers,
        destination identifiers, edge weights
    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each seed.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> b_ego_graph, offsets = cugraph.batched_ego_graphs(G, seeds=[1,5],
    ...                                                   radius=2)

    """

    (G, input_type) = ensure_cugraph_obj(G, nx_weight_attr="weight")

    if seeds is not None:
        if isinstance(seeds, int):
            seeds = [seeds]
        if isinstance(seeds, list):
            seeds = cudf.Series(seeds)

        if G.renumbered is True:
            if isinstance(seeds, cudf.DataFrame):
                seeds = G.lookup_internal_vertex_id(seeds, seeds.columns)
            else:
                seeds = G.lookup_internal_vertex_id(seeds)

    # Match the seed to the vertex dtype
    seeds_type = G.edgelist.edgelist_df["src"].dtype
    seeds = seeds.astype(seeds_type)

    do_expensive_check = False
    source, destination, weight, offset = pylibcugraph_ego_graph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        source_vertices=seeds,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )

    offsets = cudf.Series(offset)

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    df["weight"] = weight

    if G.renumbered:
        df = G.unrenumber(df, "src", preserve_order=True)
        df = G.unrenumber(df, "dst", preserve_order=True)

    return _convert_df_series_to_output_type(df, offsets, input_type)
