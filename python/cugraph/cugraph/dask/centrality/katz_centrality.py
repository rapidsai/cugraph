# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cugraph.dask.centrality import\
    mg_katz_centrality_wrapper as mg_katz_centrality
import cugraph.comms.comms as Comms
import dask_cudf


def call_katz_centrality(sID,
                         data,
                         src_col_name,
                         dst_col_name,
                         num_verts,
                         num_edges,
                         vertex_partition_offsets,
                         aggregate_segment_offsets,
                         alpha,
                         beta,
                         max_iter,
                         tol,
                         nstart,
                         normalized):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    local_size = len(aggregate_segment_offsets) // Comms.get_n_workers(sID)
    segment_offsets = \
        aggregate_segment_offsets[local_size * wid: local_size * (wid + 1)]
    return mg_katz_centrality.mg_katz_centrality(data[0],
                                                 src_col_name,
                                                 dst_col_name,
                                                 num_verts,
                                                 num_edges,
                                                 vertex_partition_offsets,
                                                 wid,
                                                 handle,
                                                 segment_offsets,
                                                 alpha,
                                                 beta,
                                                 max_iter,
                                                 tol,
                                                 nstart,
                                                 normalized)


def katz_centrality(input_graph,
                    alpha=None,
                    beta=None,
                    max_iter=100,
                    tol=1.0e-5,
                    nstart=None,
                    normalized=True):
    """
    Compute the Katz centrality for the nodes of the graph G.

    Parameters
    ----------
    input_graph : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed (DiGraph) or undirected edges (Graph).

    alpha : float, optional (default=None)
        Attenuation factor. If alpha is not specified then
        it is internally calculated as 1/(degree_max) where degree_max is the
        maximum out degree.

        NOTE
            The maximum acceptable value of alpha for convergence
            alpha_max = 1/(lambda_max) where lambda_max is the largest
            eigenvalue of the graph.
            Since lambda_max is always lesser than or equal to degree_max for a
            graph, alpha_max will always be greater than or equal to
            (1/degree_max). Therefore, setting alpha to (1/degree_max) will
            guarantee that it will never exceed alpha_max thus in turn
            fulfilling the requirement for convergence.

    beta : None
        A weight scalar - currently Not Supported

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    nstart : dask_cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for katz centrality

        nstart['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        nstart['values'] : dask_cudf.Series
            Contains the katz centrality values of vertices

    normalized : bool, optional (default=True)
        If True normalize the resulting katz centrality values

    Returns
    -------
    katz_centrality : dask_cudf.DataFrame
        GPU data frame containing two dask_cudf.Series of size V: the
        vertex identifiers and the corresponding katz centrality values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['katz_centrality'] : dask_cudf.Series
            Contains the katz centrality of vertices

    Examples
    --------
    >>> # import cugraph.dask as dcg
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> # chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> # ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize)
    >>> # dg = cugraph.Graph(directed=True)
    >>> # dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    >>> #                            edge_attr='value')
    >>> # pr = dcg.katz_centrality(dg)

    """
    # FIXME: Uncomment out the above (broken) example

    nstart = None

    client = default_client()

    input_graph.compute_renumber_edge_list(transposed=True)
    ddf = input_graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    result = [client.submit(call_katz_centrality,
                            Comms.get_session_id(),
                            wf[1],
                            src_col_name,
                            dst_col_name,
                            num_verts,
                            num_edges,
                            vertex_partition_offsets,
                            input_graph.aggregate_segment_offsets,
                            alpha,
                            beta,
                            max_iter,
                            tol,
                            nstart,
                            normalized,
                            workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
