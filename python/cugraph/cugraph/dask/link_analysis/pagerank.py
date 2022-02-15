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
from cugraph.dask.link_analysis import mg_pagerank_wrapper as mg_pagerank
import cugraph.comms.comms as Comms
import dask_cudf
from dask.dataframe.shuffle import rearrange_by_column


def call_pagerank(sID,
                  data,
                  src_col_name,
                  dst_col_name,
                  num_verts,
                  num_edges,
                  vertex_partition_offsets,
                  aggregate_segment_offsets,
                  alpha,
                  max_iter,
                  tol,
                  personalization,
                  nstart):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    local_size = len(aggregate_segment_offsets) // Comms.get_n_workers(sID)
    segment_offsets = \
        aggregate_segment_offsets[local_size * wid: local_size * (wid + 1)]
    return mg_pagerank.mg_pagerank(data[0],
                                   src_col_name,
                                   dst_col_name,
                                   num_verts,
                                   num_edges,
                                   vertex_partition_offsets,
                                   wid,
                                   handle,
                                   segment_offsets,
                                   alpha,
                                   max_iter,
                                   tol,
                                   personalization,
                                   nstart)


def pagerank(input_graph,
             alpha=0.85,
             personalization=None,
             max_iter=100,
             tol=1.0e-5,
             nstart=None):

    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    input_graph : cugraph.DiGraph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe(edge weights are not used for this
        algorithm). Undirected Graph not currently supported.

    alpha : float, optional (default=0.85)
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    personalization : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the personalization information.
        Currently not supported.

        personalization['vertex'] : cudf.Series
            Subset of vertices of graph for personalization
        personalization['values'] : cudf.Series
            Personalization values for vertices

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 30.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.

    nstart : not supported
        initial guess for pagerank

    Returns
    -------
    PageRank : dask_cudf.DataFrame
        GPU data frame containing two dask_cudf.Series of size V: the
        vertex identifiers and the corresponding PageRank values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['pagerank'] : dask_cudf.Series
            Contains the PageRank score

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
    >>> # pr = dcg.pagerank(dg)

    """
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

    if personalization is not None:
        if input_graph.renumbered is True:
            personalization = input_graph.add_internal_vertex_id(
                personalization, "vertex", "vertex"
            )

        # Function to assign partition id to personalization dataframe
        def _set_partitions_pre(s, divisions):
            partitions = divisions.searchsorted(s, side="right") - 1
            partitions[
                divisions.tail(1).searchsorted(s, side="right").astype("bool")
            ] = (len(divisions) - 2)
            return partitions

        # Assign partition id column as per vertex_partition_offsets
        df = personalization
        by = ['vertex']
        meta = df._meta._constructor_sliced([0])
        divisions = vertex_partition_offsets
        partitions = df[by].map_partitions(
            _set_partitions_pre, divisions=divisions, meta=meta
        )

        df2 = df.assign(_partitions=partitions)

        # Shuffle personalization values according to the partition id
        df3 = rearrange_by_column(
            df2,
            "_partitions",
            max_branch=None,
            npartitions=len(divisions) - 1,
            shuffle="tasks",
            ignore_index=False,
        ).drop(columns=["_partitions"])

        p_data = get_distributed_data(df3)

        result = [client.submit(call_pagerank,
                                Comms.get_session_id(),
                                wf[1],
                                src_col_name,
                                dst_col_name,
                                num_verts,
                                num_edges,
                                vertex_partition_offsets,
                                input_graph.aggregate_segment_offsets,
                                alpha,
                                max_iter,
                                tol,
                                p_data.worker_to_parts[wf[0]][0],
                                nstart,
                                workers=[wf[0]])
                  for idx, wf in enumerate(data.worker_to_parts.items())]
    else:
        result = [client.submit(call_pagerank,
                                Comms.get_session_id(),
                                wf[1],
                                src_col_name,
                                dst_col_name,
                                num_verts,
                                num_edges,
                                vertex_partition_offsets,
                                input_graph.aggregate_segment_offsets,
                                alpha,
                                max_iter,
                                tol,
                                personalization,
                                nstart,
                                workers=[wf[0]])
                  for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
