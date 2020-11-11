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

from numba import cuda

import cudf

# optional dependencies
try:
    import cupy as cp
    from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
except ModuleNotFoundError:
    cp = None
try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None


def get_traversed_path(df, id):
    """
    Take the DataFrame result from a BFS or SSSP function call and extract
    the path to a specified vertex.

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call
    id : Int
        The vertex ID

    Returns
    ---------
    df : cudf.DataFrame
        a dataframe containing the path steps


    Examples
    --------
    >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>>
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> sssp_df = cugraph.sssp(G, 1)
    >>> path = cugraph.utils.get_traversed_path(sssp_df, 32)
    """

    if 'vertex' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'vertex' column missing")
    if 'distance' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'distance' column missing")
    if 'predecessor' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'predecessor' column missing")
    if type(id) != int:
        raise ValueError("The vertex 'id' needs to be an integer")

    # There is no guarantee that the dataframe has not been filtered
    # or edited.  Therefore we cannot assume that using the vertex ID
    # as an index will work

    ddf = df[df['vertex'] == id]
    if len(ddf) == 0:
        raise ValueError("The vertex (", id, " is not in the result set")
    pred = ddf['predecessor'].iloc[0]

    answer = []
    answer.append(ddf)

    while pred != -1:
        ddf = df[df['vertex'] == pred]
        pred = ddf['predecessor'].iloc[0]
        answer.append(ddf)

    return cudf.concat(answer)


def get_traversed_path_list(df, id):
    """
    Take the DataFrame result from a BFS or SSSP function call and extract
    the path to a specified vertex as a series of steps

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call
    id : Int
        The vertex ID

    Returns
    ---------
    a : Python array
        a ordered array containing the steps from id to root

    Examples
    --------
    >>> gdf = cudf.read_csv(graph_file)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> sssp_df = cugraph.sssp(G, 1)
    >>> path = cugraph.utils.get_traversed_path_list(sssp_df, 32)
    """

    if 'vertex' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'vertex' column missing")
    if 'distance' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'distance' column missing")
    if 'predecessor' not in df.columns:
        raise ValueError("DataFrame does not appear to be a BFS or "
                         "SSP result - 'predecessor' column missing")
    if type(id) != int:
        raise ValueError("The vertex 'id' needs to be an integer")

    # There is no guarantee that the dataframe has not been filtered
    # or edited.  Therefore we cannot assume that using the vertex ID
    # as an index will work

    pred = -1
    answer = []
    answer.append(id)

    ddf = df[df['vertex'] == id]
    if len(ddf) == 0:
        raise ValueError("The vertex (", id, " is not in the result set")

    pred = ddf['predecessor'].iloc[0]

    while pred != -1:
        answer.append(pred)

        ddf = df[df['vertex'] == pred]
        pred = ddf['predecessor'].iloc[0]

    return answer


def is_cuda_version_less_than(min_version=(10, 2)):
    """
    Returns True if the version of CUDA being used is less than min_version
    """
    this_cuda_ver = cuda.runtime.get_version()  # returns (<major>, <minor>)
    if this_cuda_ver[0] > min_version[0]:
        return False
    if this_cuda_ver[0] < min_version[0]:
        return True
    if this_cuda_ver[1] < min_version[1]:
        return True
    return False


def ensure_cugraph_obj(obj, weight=None, coo_graph_type=None):
    """
    Convert the input obj - if possible - to a cuGraph Graph-type obj (Graph,
    DiGraph, etc.) and return a tuple of (cugraph Graph-type obj, original
    input obj type). If weight is specified, it is a string associated with the
    attribute name for the weights column. If coo_graph_type is specified, it
    is used as the cugraph Graph-type obj to create when converting from a COO.
    """
    # FIXME: importing here to avoid circular import
    from cugraph.structure import (Graph,
                                   DiGraph,
                                   )
    from cugraph.utilities.nx_factory import convert_from_nx

    input_type = type(obj)
    if input_type in [Graph, DiGraph]:
        return (obj, input_type)

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return (convert_from_nx(obj, weight), input_type)

    elif (cp is not None) and (input_type is cp_coo_matrix):
        if coo_graph_type is None:
            coo_graph_type = Graph
        elif coo_graph_type not in [Graph, DiGraph]:
            raise TypeError(f"coo_graph_type must be either a cugraph Graph "
                            f"or DiGraph, got: {coo_graph_type}")

        if weight is not None:
            df = cudf.DataFrame({"source": cp.ascontiguousarray(obj.row),
                                 "destination": cp.ascontiguousarray(obj.col),
                                 weight: cp.ascontiguousarray(obj.data)})
        else:
            df = cudf.DataFrame({"source": cp.ascontiguousarray(obj.row),
                                 "destination": cp.ascontiguousarray(obj.col)})

        # FIXME:
        # * do a quick check that symmetry is stored explicitly in the cupy
        #   data for sym matrices (ie. for each uv, check vu is there)
        # * populate the cugraph graph with directed data and set renumbering
        #   to false in from edge list call.
        G = coo_graph_type()
        G.from_cudf_edgelist(df, edge_attr=weight, renumber=True)

        return (G, input_type)

    else:
        raise TypeError(f"obj of type {input_type} is not supported.")
