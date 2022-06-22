# Copyright (c) 2022, NVIDIA CORPORATION.
#
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
#

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          MGGraph,
                          eigenvector_centrality as pylib_eigen
                          )
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import cupy


def call_eigenvector_centrality(sID,
                                data,
                                graph_properties,
                                store_transposed,
                                do_expensive_check,
                                src_col_name,
                                dst_col_name,
                                num_verts,
                                num_edges,
                                vertex_partition_offsets,
                                aggregate_segment_offsets,
                                max_iter,
                                tol,
                                normalized):
    handle = Comms.get_handle(sID)
    h = ResourceHandle(handle.getHandle())
    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = cudf.Series(cupy.ones(srcs.size, dtype="float32"))

    if "value" in data[0].columns:
        weights = data[0]['value']

    mg = MGGraph(h,
                 graph_properties,
                 srcs,
                 dsts,
                 weights,
                 store_transposed,
                 num_edges,
                 do_expensive_check)

    result = pylib_eigen(h,
                         mg,
                         tol,
                         max_iter,
                         do_expensive_check)
    return result


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_values = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["eigenvector_centrality"] = cupy_values
    return df


def eigenvector_centrality(
    input_graph, max_iter=100, tol=1.0e-6, normalized=True
):
    """
    Compute the eigenvector centrality for a graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node i is the
    i-th element of the vector x defined by the eigenvector equation.

    Parameters
    ----------
    input_graph : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.

    tol : float, optional (default=1e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    normalized : bool, optional, default=True
        If True normalize the resulting eigenvector centrality values

    Returns
    -------
    df : dask_cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding eigenvector centrality values.
        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['eigenvector_centrality'] : cudf.Series
            Contains the eigenvector centrality of vertices

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          chunksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> ec = dcg.eigenvector_centrality(dg)

    """
    client = default_client()
    # Calling renumbering results in data that is sorted by degree
    input_graph.compute_renumber_edge_list(
        transposed=False, legacy_renum_only=True)

    graph_properties = GraphProperties(
        is_multigraph=False)

    store_transposed = False
    do_expensive_check = False

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    # FIXME Move this call to the function creating a directed
    # graph from a dask dataframe because duplicated edges need
    # to be dropped
    ddf = input_graph.edgelist.edgelist_df
    ddf = ddf.map_partitions(
        lambda df: df.drop_duplicates(subset=[src_col_name, dst_col_name]))

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    input_graph.compute_renumber_edge_list(transposed=True)
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]

    cupy_result = [client.submit(call_eigenvector_centrality,
                                 Comms.get_session_id(),
                                 wf[1],
                                 graph_properties,
                                 store_transposed,
                                 do_expensive_check,
                                 src_col_name,
                                 dst_col_name,
                                 num_verts,
                                 num_edges,
                                 vertex_partition_offsets,
                                 input_graph.aggregate_segment_offsets,
                                 max_iter,
                                 tol,
                                 normalized,
                                 workers=[wf[0]])
                   for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(cupy_result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays,
                                 workers=client.who_has(
                                     cp_arrays)[cp_arrays.key])
                   for cp_arrays in cupy_result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')
