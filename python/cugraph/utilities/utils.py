# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.number_map import NumberMap
from cugraph.utilities import path_retrieval_wrapper

import cudf
from rmm._cuda.gpu import (
    getDeviceAttribute,
    cudaDeviceAttr,
)


# optional dependencies
try:
    import cupy as cp
    from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
    from cupyx.scipy.sparse.csr import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.csc import csc_matrix as cp_csc_matrix

    CP_MATRIX_TYPES = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]
    CP_COMPRESSED_MATRIX_TYPES = [cp_csr_matrix, cp_csc_matrix]
except ModuleNotFoundError:
    cp = None
    CP_MATRIX_TYPES = []
    CP_COMPRESSED_MATRIX_TYPES = []

try:
    import scipy as sp
    from scipy.sparse.coo import coo_matrix as sp_coo_matrix
    from scipy.sparse.csr import csr_matrix as sp_csr_matrix
    from scipy.sparse.csc import csc_matrix as sp_csc_matrix

    SP_MATRIX_TYPES = [sp_coo_matrix, sp_csr_matrix, sp_csc_matrix]
    SP_COMPRESSED_MATRIX_TYPES = [sp_csr_matrix, sp_csc_matrix]
except ModuleNotFoundError:
    sp = None
    SP_MATRIX_TYPES = []
    SP_COMPRESSED_MATRIX_TYPES = []

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

    if "vertex" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'vertex' column missing"
        )
    if "distance" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'distance' column missing"
        )
    if "predecessor" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'predecessor' column missing"
        )
    if type(id) != int:
        raise ValueError("The vertex 'id' needs to be an integer")

    # There is no guarantee that the dataframe has not been filtered
    # or edited.  Therefore we cannot assume that using the vertex ID
    # as an index will work

    ddf = df[df["vertex"] == id]
    if len(ddf) == 0:
        raise ValueError("The vertex (", id, " is not in the result set")
    pred = ddf["predecessor"].iloc[0]

    answer = []
    answer.append(ddf)

    while pred != -1:
        ddf = df[df["vertex"] == pred]
        pred = ddf["predecessor"].iloc[0]
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

    if "vertex" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'vertex' column missing"
        )
    if "distance" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'distance' column missing"
        )
    if "predecessor" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'predecessor' column missing"
        )
    if type(id) != int:
        raise ValueError("The vertex 'id' needs to be an integer")

    # There is no guarantee that the dataframe has not been filtered
    # or edited.  Therefore we cannot assume that using the vertex ID
    # as an index will work

    pred = -1
    answer = []
    answer.append(id)

    ddf = df[df["vertex"] == id]
    if len(ddf) == 0:
        raise ValueError("The vertex (", id, " is not in the result set")

    pred = ddf["predecessor"].iloc[0]

    while pred != -1:
        answer.append(pred)

        ddf = df[df["vertex"] == pred]
        pred = ddf["predecessor"].iloc[0]

    return answer


def get_traversed_cost(df, source_col, dest_col, value_col):
    """
    Take the DataFrame result from a BFS or SSSP function call and sums
    the given weights along the path to the starting vertex.
    The source_col, dest_col identifiers need to match with the vertex and
    predecessor columns of df.

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call
    source_col : cudf.DataFrame
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the source index for each edge.
        Source indices must be an integer type.
    dest_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the destination index for each edge.
        Destination indices must be an integer type.
    value_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains values associated with this edge.
        Weight should be a floating type.

    Returns
    ---------
    df : cudf.DataFrame
        DataFrame containing two columns 'vertex' and 'info'.
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

    src, dst, val = symmetrize(source_col,
                               dest_col,
                               value_col)

    symmetrized_df = cudf.DataFrame()
    symmetrized_df['source'] = src
    symmetrized_df['destination'] = dst
    symmetrized_df['weights'] = val

    input_df = df.merge(symmetrized_df,
                        left_on=['vertex', 'predecessor'],
                        right_on=['source', 'destination'],
                        how="left"
                        )
    input_df = input_df.fillna(0)

    numbering = NumberMap()
    numbering.from_series(df['vertex'])
    renumbered_df = numbering.add_internal_vertex_id(input_df,
                                                     "vertex_id",
                                                     ["vertex"])
    renumbered_df['predecessor_id'] = numbering.to_internal_vertex_id(
            renumbered_df['predecessor']).fillna(-1)

    out_df = path_retrieval_wrapper.get_traversed_cost(renumbered_df)
    return out_df


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


def is_device_version_less_than(min_version=(7, 0)):
    """
    Returns True if the version of CUDA being used is less than min_version
    """
    major_version = getDeviceAttribute(
        cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, 0
    )
    minor_version = getDeviceAttribute(
        cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, 0
    )
    if major_version > min_version[0]:
        return False
    if major_version < min_version[0]:
        return True
    if minor_version < min_version[1]:
        return True
    return False


def get_device_memory_info():
    """
    Returns the total amount of global memory on the device in bytes
    """
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[1]


# FIXME: if G is a Nx type, the weight attribute is assumed to be "weight", if
# set. An additional optional parameter for the weight attr name when accepting
# Nx graphs may be needed.  From the Nx docs:
# |      Many NetworkX algorithms designed for weighted graphs use
# |      an edge attribute (by default `weight`) to hold a numerical value.
def ensure_cugraph_obj(obj, nx_weight_attr=None, matrix_graph_type=None):
    """
    Convert the input obj - if possible - to a cuGraph Graph-type obj (Graph,
    DiGraph, etc.) and return a tuple of (cugraph Graph-type obj, original
    input obj type). If matrix_graph_type is specified, it is used as the
    cugraph Graph-type obj to create when converting from a matrix type.
    """
    # FIXME: importing here to avoid circular import
    from cugraph.structure import Graph, DiGraph, MultiGraph, MultiDiGraph
    from cugraph.utilities.nx_factory import convert_from_nx

    input_type = type(obj)
    if input_type in [Graph, DiGraph, MultiGraph, MultiDiGraph]:
        return (obj, input_type)

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return (convert_from_nx(obj, weight=nx_weight_attr), input_type)

    elif (input_type in CP_MATRIX_TYPES) or (input_type in SP_MATRIX_TYPES):

        if matrix_graph_type is None:
            matrix_graph_type = Graph
        elif matrix_graph_type not in [Graph, DiGraph]:
            raise TypeError(
                f"matrix_graph_type must be either a cugraph "
                f"Graph or DiGraph, got: {matrix_graph_type}"
            )

        if input_type in (
            CP_COMPRESSED_MATRIX_TYPES + SP_COMPRESSED_MATRIX_TYPES
        ):
            coo = obj.tocoo(copy=False)
        else:
            coo = obj

        if input_type in CP_MATRIX_TYPES:
            df = cudf.DataFrame(
                {
                    "source": cp.ascontiguousarray(coo.row),
                    "destination": cp.ascontiguousarray(coo.col),
                    "weight": cp.ascontiguousarray(coo.data),
                }
            )
        else:
            df = cudf.DataFrame(
                {"source": coo.row, "destination": coo.col, "weight": coo.data}
            )
        # FIXME:
        # * do a quick check that symmetry is stored explicitly in the cupy
        #   data for sym matrices (ie. for each uv, check vu is there)
        # * populate the cugraph graph with directed data and set renumbering
        #   to false in from edge list call.
        G = matrix_graph_type()
        G.from_cudf_edgelist(df, edge_attr="weight", renumber=True)

        return (G, input_type)

    else:
        raise TypeError(f"obj of type {input_type} is not supported.")


def is_cp_matrix_type(m):
    return m in CP_MATRIX_TYPES


def is_sp_matrix_type(m):
    return m in SP_MATRIX_TYPES


def is_matrix_type(m):
    return is_cp_matrix_type(m) or is_sp_matrix_type(m)


def import_optional(mod, import_from=None):
    """
    import module or object 'mod', possibly from module 'import_from', and
    return the module object or object.  If the import raises
    ModuleNotFoundError, returns None.

    This method was written to support importing "optional" dependencies so
    code can be written to run even if the dependency is not installed.

    >>> nx = import_optional("networkx")  # networkx is not installed
    >>> if nx:
    ...    G = nx.Graph()
    ... else:
    ...    print("Warning: NetworkX is not installed, using CPUGraph")
    ...    G = cpu_graph.Graph()
    >>>
    """
    namespace = {}
    try:
        if import_from:
            exec(f"from {import_from} import {mod}", namespace)
        else:
            exec(f"import {mod}", namespace)
    except ModuleNotFoundError:
        pass

    return namespace.get(mod)
