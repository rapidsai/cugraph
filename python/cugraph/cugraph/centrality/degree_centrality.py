# Copyright (c) 2022, NVIDIA CORPORATION.
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
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)


def degree_centrality(G, normalized=True):
    """
    Computes the degree centrality of each vertex of the input graph.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    normalized : bool, optional, default=True
        If True normalize the resulting degree centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding degree centrality values.
        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['degree_centrality'] : cudf.Series
            Contains the degree centrality of vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> dc = cugraph.degree_centrality(G)

    """
    G, isNx = ensure_cugraph_obj_for_nx(G)

    df = G.degree()
    df.rename(columns={"degree": "degree_centrality"}, inplace=True)

    if normalized:
        df["degree_centrality"] /= G.number_of_nodes() - 1

    if isNx is True:
        dict = df_score_to_dictionary(df, "degree_centrality")
        return dict
    else:
        return df
