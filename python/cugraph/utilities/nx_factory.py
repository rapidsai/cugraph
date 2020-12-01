# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cugraph
from .utils import import_optional
from cudf import from_pandas

nx = import_optional("networkx")


def convert_from_nx(nxG, weight=None):
    """
    weight, if given, is the string/name of the edge attr in nxG to use for
    weights in the resulting cugraph obj.  If nxG has no edge attributes,
    weight is ignored even if specified.
    """
    if nx is None:
        raise RuntimeError("NetworkX could not be imported, "
                           "cannot convert nxG")
    if type(nxG) == nx.classes.graph.Graph:
        G = cugraph.Graph()
    elif type(nxG) == nx.classes.digraph.DiGraph:
        G = cugraph.DiGraph()
    else:
        raise ValueError("nxG does not appear to be a NetworkX graph type")

    pdf = nx.to_pandas_edgelist(nxG)
    num_col = len(pdf.columns)

    if num_col < 2:
        raise ValueError("NetworkX graph did not contain edges")

    if weight is None:
        num_col == 2
        pdf = pdf[["source", "target"]]

    if num_col >= 3 and weight is not None:
        pdf = pdf[["source", "target", weight]]
        num_col = 3

    gdf = from_pandas(pdf)

    if num_col == 2:
        G.from_cudf_edgelist(gdf, "source", "target")
    else:
        G.from_cudf_edgelist(gdf, "source", "target", weight)

    del gdf
    del pdf

    return G


def is_networkx_graph(G):
    if nx is None:
        raise RuntimeError("NetworkX could not be imported, cannot check G")
    return isinstance(G, nx.classes.graph.Graph)


def check_nx_graph(G, weight=None):
    """
    This is a convenience function that will ensure the proper graph type

    Parameters
    ----------
    G : cudf.Graph or networkx.Graph
    weight : str or None
        which column to use for weight. Default is None

    Returns
    -------
    G : cudf.Graph
        returns a cugraph.Graph that is either the orginal input or
        a conversion from NetworkX

    is_nx : Boolean
        indicates rather or not the Graph was converted
    """

    if nx is None:
        raise RuntimeError("NetworkX could not be imported, cannot check G")
    if isinstance(G, nx.classes.graph.Graph):
        return convert_from_nx(G, weight), True
    else:
        return G, False


def df_score_to_dictionary(df, k, v="vertex"):
    """
    Convert a dataframe to a dictionary

    Parameters
    ----------
     df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding score values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df[..] : cudf.Series
            Contains the scores of the vertices

    k : str
        score column name
    v : str
        the vertex column name. Default is "vertex"


    Returns
    -------
    dict : Dictionary of vertices and score

    """
    df = df.sort_values(by=v)
    return df.to_pandas().set_index(v).to_dict()[k]


def df_edge_score_to_dictionary(df, k, src="src", dst="dst"):
    """
    Convert a dataframe to a dictionary

    Parameters
    ----------
     df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding score values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df[X] : cudf.Series
            Contains the scores of the vertices

    k : str
        score column name

    src : str
        source column name
    dst : str
        destination column name


    Returns
    -------
    dict : Dictionary of vertices and score

    """
    pdf = df.sort_values(by=[src, dst]).to_pandas()
    d = {}
    for i in range(len(pdf)):
        d[(pdf[src][i], pdf[dst][i])] = pdf[k][i]

    return d


def cugraph_to_nx(G):
    if nx is None:
        raise RuntimeError("NetworkX could not be imported, cannot convert G")

    pdf = G.view_edge_list().to_pandas()
    num_col = len(pdf.columns)

    if num_col == 2:
        Gnx = nx.from_pandas_edgelist(pdf, source="src", target="dst")
    else:
        Gnx = nx.from_pandas_edgelist(pdf, source="src", target="dst",
                                      edge_attr="weights")

    return Gnx
