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

from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )
import cudf
import warnings

from pylibcugraph import (pagerank as pylibcugraph_pagerank,
                          personalized_pagerank as pylibcugraph_p_pagerank,
                          ResourceHandle
                          )


def pagerank(
    G, alpha=0.85, personalization=None,
    precomputed_vertex_out_weight_sums=None, max_iter=100, tol=1.0e-5,
    nstart=None, weight=None, dangling=None, has_initial_guess=None
):
    """
    Compute the core numbers for the nodes of the graph G. A k-core of a graph
    is a maximal subgraph that contains nodes of degree k or more.
    A node has a core number of k if it belongs a k-core but not to k+1-core.
    This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph should contain undirected edges where undirected edges are
        represented as directed edges in both directions. While this graph
        can contain edge weights, they don't participate in the calculation
        of the core numbers.

    degree_type: str
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.
        This option is currently ignored in this release, and setting it will
        result in a warning.

    Returns
    -------
    df : cudf.DataFrame or python dictionary (in NetworkX input)
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding core number values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['core_number'] : cudf.Series
            Contains the core number of vertices

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> df = cugraph.core_number(G)

    """

    # FIXME: fix this
    has_initial_guess = False

    G, isNx = ensure_cugraph_obj_for_nx(G)
    do_expensive_check = False

    if nstart is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = 'vertex'
            nstart = G.add_internal_vertex_id(
                nstart, "vertex", cols
            )

    if personalization is not None:
        if not isinstance(personalization, cudf.DataFrame):
            raise NotImplementedError(
                "personalization other than a cudf dataframe "
                "currently not supported"
            )
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = personalization.columns[:-1].to_list()
            else:
                cols = 'vertex'
            personalization = G.add_internal_vertex_id(
                personalization, "vertex", cols
            )

        vertex, pagerank_values = \
            pylibcugraph_p_pagerank(
                resource_handle=ResourceHandle(),
                graph=G._plc_graph,
                precomputed_vertex_out_weight_sums=precomputed_vertex_out_weight_sums,
                personalization_vertices=personalization["vertex"],
                personalization_values=personalization["values"],
                alpha=alpha,
                epsilon=tol,
                max_iterations=max_iter,
                has_initial_guess=has_initial_guess,
                do_expensive_check=do_expensive_check
            )
    else:
        vertex, pagerank_values = \
            pylibcugraph_pagerank(
                resource_handle=ResourceHandle(),
                graph=G._plc_graph,
                precomputed_vertex_out_weight_sums=precomputed_vertex_out_weight_sums,
                alpha=alpha,
                epsilon=tol,
                max_iterations=max_iter,
                has_initial_guess=has_initial_guess,
                do_expensive_check=do_expensive_check
            )


    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["pagerank"] = pagerank_values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, 'pagerank')

    return df
