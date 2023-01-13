# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import importlib

from numba import cuda

import cudf
from cudf.core.column import as_column

from cuda.cudart import cudaDeviceAttr
from rmm._cuda.gpu import getDeviceAttribute


# optional dependencies
try:
    import cupy as cp
    from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix

    __cp_matrix_types = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]
    __cp_compressed_matrix_types = [cp_csr_matrix, cp_csc_matrix]
except ModuleNotFoundError:
    cp = None
    __cp_matrix_types = []
    __cp_compressed_matrix_types = []

cupy_package = cp

try:
    import scipy as sp
    from scipy.sparse import coo_matrix as sp_coo_matrix
    from scipy.sparse import csr_matrix as sp_csr_matrix
    from scipy.sparse import csc_matrix as sp_csc_matrix

    __sp_matrix_types = [sp_coo_matrix, sp_csr_matrix, sp_csc_matrix]
    __sp_compressed_matrix_types = [sp_csr_matrix, sp_csc_matrix]
except ModuleNotFoundError:
    sp = None
    __sp_matrix_types = []
    __sp_compressed_matrix_types = []

scipy_package = sp

try:
    import networkx as nx

    __nx_graph_types = [nx.Graph, nx.DiGraph]
except ModuleNotFoundError:
    nx = None
    __nx_graph_types = []

nx_package = nx


def get_traversed_path(df, id):
    """
    Take the DataFrame result from a BFS or SSSP function call and extract
    the path to a specified vertex.

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call

    id : vertex ID
        most be the same data types as what is in the dataframe

    Returns
    ---------
    df : cudf.DataFrame
        a dataframe containing the path steps


    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr='2')
    >>> sssp_df = cugraph.sssp(G, 1)
    >>> path = cugraph.utils.get_traversed_path(sssp_df, 32)
    >>> path
        distance  vertex  predecessor
    ...       ...     ...         ...
    ...       ...     ...         ...
    ...       ...     ...         ...

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
    if isinstance(id, type(df["vertex"].iloc[0])):
        raise ValueError("The vertex 'id' needs to be the same as df['vertex']")

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
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr='2')
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
    if isinstance(id, type(df["vertex"].iloc[0])):
        raise ValueError("The vertex 'id' needs to be the same as df['vertex']")

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
    etc.) and return a tuple of (cugraph Graph-type obj, original
    input obj type). If matrix_graph_type is specified, it is used as the
    cugraph Graph-type obj to create when converting from a matrix type.
    """
    # FIXME: importing here to avoid circular import
    from cugraph.structure import Graph
    from cugraph.utilities.nx_factory import convert_from_nx

    input_type = type(obj)
    if is_cugraph_graph_type(input_type):
        return (obj, input_type)

    elif is_nx_graph_type(input_type):
        return (convert_from_nx(obj, weight=nx_weight_attr), input_type)

    elif (input_type in __cp_matrix_types) or (input_type in __sp_matrix_types):
        if matrix_graph_type is None:
            matrix_graph_type = Graph
        elif matrix_graph_type not in [Graph]:
            if not isinstance(matrix_graph_type, Graph):
                raise TypeError(
                    f"matrix_graph_type must be either a cugraph "
                    f"Graph, got: {matrix_graph_type}"
                )
        if input_type in (__cp_compressed_matrix_types + __sp_compressed_matrix_types):
            coo = obj.tocoo(copy=False)
        else:
            coo = obj

        if input_type in __cp_matrix_types:
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
        if isinstance(matrix_graph_type, Graph):
            G = matrix_graph_type
        else:
            G = matrix_graph_type()
        G.from_cudf_edgelist(df, edge_attr="weight", renumber=True)

        return (G, input_type)

    else:
        raise TypeError(f"obj of type {input_type} is not supported.")


# FIXME: if G is a Nx type, the weight attribute is assumed to be "weight", if
# set. An additional optional parameter for the weight attr name when accepting
# Nx graphs may be needed.  From the Nx docs:
# |      Many NetworkX algorithms designed for weighted graphs use
# |      an edge attribute (by default `weight`) to hold a numerical value.
def ensure_cugraph_obj_for_nx(obj, nx_weight_attr="weight", store_transposed=False):
    """
    Ensures a cuGraph Graph-type obj is returned for either cuGraph or Nx
    Graph-type objs. If obj is a Nx type,
    """
    # FIXME: importing here to avoid circular import
    from cugraph.utilities.nx_factory import convert_from_nx

    input_type = type(obj)
    if is_nx_graph_type(input_type):
        return (
            convert_from_nx(
                obj, weight=nx_weight_attr, store_transposed=store_transposed
            ),
            True,
        )
    elif is_cugraph_graph_type(input_type):
        return (obj, False)
    else:
        raise TypeError(
            "input must be either a cuGraph or NetworkX graph "
            f"type, got {input_type}"
        )


def is_cp_matrix_type(m):
    return m in __cp_matrix_types


def is_sp_matrix_type(m):
    return m in __sp_matrix_types


def is_matrix_type(m):
    return is_cp_matrix_type(m) or is_sp_matrix_type(m)


def is_nx_graph_type(g):
    return g in __nx_graph_types


def is_cugraph_graph_type(g):
    # FIXME: importing here to avoid circular import
    from cugraph.structure import Graph, MultiGraph

    return g in [Graph, MultiGraph]


def renumber_vertex_pair(input_graph, vertex_pair):
    vertex_size = input_graph.vertex_column_size()
    columns = vertex_pair.columns.to_list()
    if vertex_size == 1:
        for col in vertex_pair.columns:
            if input_graph.renumbered:
                vertex_pair = input_graph.add_internal_vertex_id(vertex_pair, col, col)
    else:
        if input_graph.renumbered:
            vertex_pair = input_graph.add_internal_vertex_id(
                vertex_pair, "src", columns[:vertex_size]
            )
            vertex_pair = input_graph.add_internal_vertex_id(
                vertex_pair, "dst", columns[vertex_size:]
            )
    return vertex_pair


class MissingModule:
    """
    Raises RuntimeError when any attribute is accessed on instances of this
    class.

    Instances of this class are returned by import_optional() when a module
    cannot be found, which allows for code to import optional dependencies, and
    have only the code paths that use the module affected.
    """

    def __init__(self, mod_name):
        self.name = mod_name

    def __getattr__(self, attr):
        raise RuntimeError(f"This feature requires the {self.name} " "package/module")


def import_optional(mod, default_mod_class=MissingModule):
    """
    import the "optional" module 'mod' and return the module object or object.
    If the import raises ModuleNotFoundError, returns an instance of
    default_mod_class.

    This method was written to support importing "optional" dependencies so
    code can be written to run even if the dependency is not installed.

    Example
    -------
    >> from cugraph.utils import import_optional
    >> nx = import_optional("networkx")  # networkx is not installed
    >> G = nx.Graph()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      ...
    RuntimeError: This feature requires the networkx package/module

    Example
    -------
    >> class CuDFFallback:
    ..   def __init__(self, mod_name):
    ..     assert mod_name == "cudf"
    ..     warnings.warn("cudf could not be imported, using pandas instead!")
    ..   def __getattr__(self, attr):
    ..     import pandas
    ..     return getattr(pandas, attr)
    ...
    >> from cugraph.utils import import_optional
    >> df_mod = import_optional("cudf", default_mod_class=CuDFFallback)
    <stdin>:4: UserWarning: cudf could not be imported, using pandas instead!
    >> df = df_mod.DataFrame()
    >> df
    Empty DataFrame
    Columns: []
    Index: []
    >> type(df)
    <class 'pandas.core.frame.DataFrame'>
    >>
    """
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError:
        return default_mod_class(mod_name=mod)


def create_random_bipartite(v1, v2, size, dtype):
    # Creates a full bipartite graph
    import numpy as np
    from cugraph.structure import Graph

    df1 = cudf.DataFrame()
    df1["src"] = cudf.Series(range(0, v1, 1))
    df1["key"] = 1

    df2 = cudf.DataFrame()
    df2["dst"] = cudf.Series(range(v1, v1 + v2, 1))
    df2["key"] = 1

    edges = df1.merge(df2, on="key")[["src", "dst"]]
    edges = edges.sort_values(["src", "dst"]).reset_index()

    # Generate edge weights
    a = np.random.randint(1, high=size, size=(v1, v2)).astype(dtype)
    edges["weight"] = a.flatten()

    g = Graph()
    g.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="weight", renumber=False
    )

    return df1["src"], g, a


def sample_groups(df, by, n_samples):
    # Sample n_samples in the df using the by column

    # Step 1
    # first, shuffle the dataframe and reset its index,
    # so that the ordering of values within each group
    # is made random:
    df = df.sample(frac=1).reset_index(drop=True)

    # If we want to keep all samples we return
    if n_samples == -1:
        return df
    # Step 2
    # add an integer-encoded version of the "by" column,
    # since the rank aggregation seems not to work for
    # non-numeric data
    df["_"] = df[by].astype("category").cat.codes

    # Step 3
    # now do a "rank" aggregation and filter out only
    # the first N_SAMPLES ranks.
    result = df.loc[df.groupby(by)["_"].rank("first") <= n_samples, :]
    del result["_"]
    return result


def create_list_series_from_2d_ar(ar, index):
    """
    Create a cudf list series  from 2d arrays
    """
    n_rows, n_cols = ar.shape
    data = as_column(ar.flatten())
    offset_col = as_column(
        cp.arange(start=0, stop=len(data) + 1, step=n_cols), dtype="int32"
    )
    mask_col = cp.full(shape=n_rows, fill_value=True)
    mask = cudf._lib.transform.bools_to_mask(as_column(mask_col))
    lc = cudf.core.column.ListColumn(
        size=n_rows,
        dtype=cudf.ListDtype(data.dtype),
        mask=mask,
        offset=0,
        null_count=0,
        children=(offset_col, data),
    )
    return cudf.Series(lc, index=index)
