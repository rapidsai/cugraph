# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cugraph.community import egonet_wrapper
from cugraph.structure.graph import Graph, DiGraph
import cudf
from cugraph.utilities import check_nx_graph
from cugraph.utilities import (ensure_cugraph_obj,
                               is_matrix_type,
                               is_cp_matrix_type,
                               import_optional,
                               )
# optional dependencies used for handling different input types
nx = import_optional("networkx")

cp = import_optional("cupy")
cp_coo_matrix = import_optional("coo_matrix",
                                import_from="cupyx.scipy.sparse.coo")
cp_csr_matrix = import_optional("csr_matrix",
                                import_from="cupyx.scipy.sparse.csr")
cp_csc_matrix = import_optional("csc_matrix",
                                import_from="cupyx.scipy.sparse.csc")

sp = import_optional("scipy")
sp_coo_matrix = import_optional("coo_matrix",
                                import_from="scipy.sparse.coo")
sp_csr_matrix = import_optional("csr_matrix",
                                import_from="scipy.sparse.csr")
sp_csc_matrix = import_optional("csc_matrix",
                                import_from="scipy.sparse.csc")

def _convert_df_to_output_type(df, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if input_type in [Graph, DiGraph]:
        return df

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return df.to_pandas()

    elif is_matrix_type(input_type):
        # A CuPy/SciPy input means the return value will be a 2-tuple of:
        #   distance: cupy.ndarray
        #   predecessor: cupy.ndarray
        sorted_df = df.sort_values("vertex")
        if is_cp_matrix_type(input_type):
            distances = cp.fromDlpack(sorted_df["distance"].to_dlpack())
            preds = cp.fromDlpack(sorted_df["predecessor"].to_dlpack())
            if "sp_counter" in df.columns:
                return (distances, preds,
                        cp.fromDlpack(sorted_df["sp_counter"].to_dlpack()))
            else:
                return (distances, preds)
        else:
            distances = sorted_df["distance"].to_array()
            preds = sorted_df["predecessor"].to_array()
            if "sp_counter" in df.columns:
                return (distances, preds,
                        sorted_df["sp_counter"].to_array())
            else:
                return (distances, preds)
    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    """
    Compute the  induced subgraph of neighbors centered at node n within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    n : integer
        A single node

    radius: integer, optional
        Include all neighbors of distance<=radius from n.
    
    center: bool, optional
        Defaults to True. False is not supported
    undirected: bool, optional
        Defaults to False. True is not supported
    distance: key, optional
        Distances are counted in hops from n. Other cases are not supported.

    Returns
    -------
    G_ego : cuGraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        A graph descriptor with a minimum spanning tree or forest.
        The networkx graph will not have all attributes copied over

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.
    """

    #TODO check this
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight")

    if G.renumbered is True:
        n = G.lookup_internal_vertex_id(cudf.Series([n]))

    df = egonet_wrapper.egonet(G, n, radius)

    if G.renumbered:
        df = G.unrenumber(df, "src")
        df = G.unrenumber(df, "dst")

    # TODO need to return a graph and casting to the use input/output format
    #return _convert_df_to_output_type(df, input_type)

def batched_ego_graphs(G, seeds, radius=1, center=True, undirected=False, distance=None):
    """
    Compute the  induced subgraph of neighbors for each node in seeds in within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    seeds : cudf.Series
        Specifies the seeds of the induced egonet subgraphs

    radius: integer, optional
        Include all neighbors of distance<=radius from n.
    
    center: bool, optional
        Defaults to True. False is not supported
    undirected: bool, optional
        Defaults to False. True is not supported
    distance: key, optional
        Distances are counted in hops from n. Other cases are not supported.

    Returns
    -------
    ego_edge_lists : cudf.DataFrame
        GPU data frame containing all induced sources identifiers, destination identifiers, edge weights
    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list for each seed.
    """
    #TODO check this after the regular ego_graph works
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight")

    if G.renumbered is True:
        n = G.lookup_internal_vertex_id(cudf.Series([n]))[0]

    df,offsets = egonet_wrapper.egonet(G, seeds, radius)

    if G.renumbered:
        df = G.unrenumber(df, "src")
        df = G.unrenumber(df, "dst")
    return _convert_df_to_output_type(df, input_type), s
