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
import networkx as nx
import cugraph
import cudf


def convert_from_nx(nxG):
    if type(nxG) == nx.classes.graph.Graph:
        G = cugraph.Graph()
    elif type(nxG) == nx.classes.digraph.DiGraph:
        G = cugraph.DiGraph()
    else:
        raise ValueError("nxG does not appear to be a NetworkX graph type")

    pdf = nx.to_pandas_edgelist(nxG)
    gdf = cudf.from_pandas(pdf)
    num_col = len(gdf.columns)

    if num_col < 2:
        raise ValueError("NetworkX graph did not contain edges")
    elif num_col == 2:
        gdf.columns = ["0", "1"]
        G.from_cudf_edgelist(gdf, "0", "1")
    else:
        gdf.columns = ["0", "1", "2"]
        G.from_cudf_edgelist(gdf, "0", "1", "2")

    del gdf
    del pdf

    return G


def check_nx_graph(G):
    """
    This is a convenience function that will ensure the proper graph type
    """

    if isinstance(G, nx.classes.graph.Graph):
        return convert_from_nx(G), True
    else:
        return G, False


def df_score_to_dictionary(df, k):
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


    Returns
    -------
    dict : Dictionary of vertices and score

    """
    df = df.sort_values(by="vertex")
    return df.to_pandas().set_index("vertex").to_dict()[k]


def df_edge_score_to_dictionary(df, k):
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


    Returns
    -------
    dict : Dictionary of vertices and score

    """
    pdf = df.sort_values(by=["src", "dst"]).to_pandas()
    d = {}
    for i in range(len(pdf)):
        d[(pdf["src"][i], pdf["dst"][i])] = pdf[k][i]

    return d
