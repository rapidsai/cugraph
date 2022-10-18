# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

"""
Utilities specific to converting to/from NetworkX.

NetworkX is required at runtime in order to call any of these functions, so
ensure code using these utilities has done the proper checks prior to calling.
"""

import cugraph
from .utils import import_optional
import cudf
from cudf import from_pandas

# nx will be a MissingModule instance if NetworkX is not installed (any
# attribute access on a MissingModule instance results in a RuntimeError).
nx = import_optional("networkx")


def convert_unweighted_to_gdf(NX_G):
    _edges = NX_G.edges(data=False)
    src = [s for s, _ in _edges]
    dst = [d for _, d in _edges]

    _gdf = cudf.DataFrame()
    _gdf["src"] = src
    _gdf["dst"] = dst

    return _gdf


def convert_weighted_named_to_gdf(NX_G, weight):
    _edges = NX_G.edges(data=weight)

    src = [s for s, _, _ in _edges]
    dst = [d for _, d, _ in _edges]
    wt = [w for _, _, w in _edges]

    _gdf = cudf.DataFrame()
    _gdf["src"] = src
    _gdf["dst"] = dst
    _gdf["weight"] = wt

    # FIXME: The weight dtype is hardcoded.
    _gdf = _gdf.astype({"weight": "float32"})

    return _gdf


def convert_weighted_unnamed_to_gdf(NX_G):
    _pdf = nx.to_pandas_edgelist(NX_G)
    nx_col = ["source", "target"]
    wt_col = [col for col in _pdf.columns if col not in nx_col]
    if len(wt_col) != 1:
        raise ValueError("Unable to determine weight column name")

    if wt_col[0] != "weight":
        _pdf.rename(columns={wt_col[0]: "weight"})

    _gdf = from_pandas(_pdf)
    return _gdf


def convert_from_nx(nxG, weight=None, do_renumber=True, store_transposed=False):
    """
    Convert a NetworkX Graph into a cuGraph Graph.
    This might not be the most effecient way since the
    process first extracts the data from Nx into a Pandas array.

    Parameters
    ----------
     nxG : NetworkX Graph
        The NetworkX Graph top be converted.

    weight : str or None
        the weight column name.  If the graph is weighted this
        identifies which column in the Nx data to extract

    do_renumber : boolean, default is True
        Should the data be renumbered

    store_transposed : boolean, defaukt is False
        should the cuGraph Graph store the transpose of the graph

    Returns
    -------
    G : cuGraph Graph

    """

    if isinstance(nxG, nx.classes.digraph.DiGraph):
        G = cugraph.Graph(directed=True)
    elif isinstance(nxG, nx.classes.graph.Graph):
        G = cugraph.Graph()
    else:
        raise TypeError(
            f"nxG must be either a NetworkX Graph or DiGraph, got {type(nxG)}"
        )

    is_weighted = nx.is_weighted(nxG)

    if is_weighted is False:
        _gdf = convert_unweighted_to_gdf(nxG)
        G.from_cudf_edgelist(
            _gdf,
            source="src",
            destination="dst",
            edge_attr=None,
            renumber=do_renumber,
            store_transposed=store_transposed,
        )
    else:
        if weight is None:
            _gdf = convert_weighted_unnamed_to_gdf(nxG)
            G.from_cudf_edgelist(
                _gdf,
                source="source",
                destination="target",
                edge_attr="weight",
                renumber=do_renumber,
                store_transposed=store_transposed,
            )
        else:
            _gdf = convert_weighted_named_to_gdf(nxG, weight)
            G.from_cudf_edgelist(
                _gdf,
                source="src",
                destination="dst",
                edge_attr="weight",
                renumber=do_renumber,
                store_transposed=store_transposed,
            )

    return G


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
    pdf = G.view_edge_list().to_pandas()
    num_col = len(pdf.columns)

    if num_col == 2:
        Gnx = nx.from_pandas_edgelist(pdf, source="src", target="dst")
    else:
        Gnx = nx.from_pandas_edgelist(
            pdf, source="src", target="dst", edge_attr="weights"
        )

    return Gnx
