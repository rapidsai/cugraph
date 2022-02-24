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

from cugraph.cores import core_number_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )


def core_number(G):
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

    G, isNx = ensure_cugraph_obj_for_nx(G)

    df = core_number_wrapper.core_number(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, 'core_number')

    return df
