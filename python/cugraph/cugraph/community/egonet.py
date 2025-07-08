# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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


import warnings

import cudf
from cugraph.utilities import (
    ensure_cugraph_obj
)
from pylibcugraph import ego_graph as pylibcugraph_ego_graph
from pylibcugraph import ResourceHandle


def _convert_graph_to_output_type(G, input_type):
    """
    Given a cugraph.Graph, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    return G


def _convert_df_series_to_output_type(df, offsets, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    return df, offsets


# TODO: add support for a 'batch-mode' option.
def ego_graph(G, n, radius=1, center=True, undirected=None, distance=None):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, CuPy or SciPy sparse matrix
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
    G_ego : cuGraph.Graph
        A graph descriptor with a minimum spanning tree or forest.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
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
    # FIXME: 'n' should represent a single vertex, but is not being verified
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
    if weight is not None:
        df["weight"] = weight

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)
    else:
        # FIXME: The original 'src' and 'dst' are not stored in 'simpleGraph'
        src_names = "src"
        dst_names = "dst"

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)

    return _convert_graph_to_output_type(result_graph, input_type)
