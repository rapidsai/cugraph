# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import operator as op

from dask.distributed import wait, default_client

import cugraph.comms.comms as Comms
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
from cugraph.dask.community import louvain_wrapper as c_mg_louvain
from cugraph.utilities.utils import is_cuda_version_less_than

import dask_cudf


def call_louvain(sID,
                 data,
                 src_col_name,
                 dst_col_name,
                 num_verts,
                 num_edges,
                 vertex_partition_offsets,
                 aggregate_segment_offsets,
                 max_level,
                 resolution):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    local_size = len(aggregate_segment_offsets) // Comms.get_n_workers(sID)
    segment_offsets = \
        aggregate_segment_offsets[local_size * wid: local_size * (wid + 1)]
    return c_mg_louvain.louvain(data[0],
                                src_col_name,
                                dst_col_name,
                                num_verts,
                                num_edges,
                                vertex_partition_offsets,
                                wid,
                                handle,
                                segment_offsets,
                                max_level,
                                resolution)


def louvain(input_graph, max_iter=100, resolution=1.0):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method on multiple GPUs

    It uses the Louvain method described in:

    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
    community hierarchies in large networks, J Stat Mech P10008 (2008),
    http://arxiv.org/abs/0803.0476

    Parameters
    ----------
    input_graph : cugraph.Graph or NetworkX Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.

    max_iter : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: float/double, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

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
    >>> # parts, modularity_score = dcg.louvain(dg)
    """
    # FIXME: Uncomment out the above (broken) example

    # MG Louvain currently requires CUDA 10.2 or higher.
    # FIXME: remove this check once RAPIDS drops support for CUDA < 10.2
    if is_cuda_version_less_than((10, 2)):
        raise NotImplementedError("Multi-GPU Louvain is not implemented for "
                                  "this version of CUDA. Ensure CUDA version "
                                  "10.2 or higher is installed.")

    # FIXME: dask methods to populate graphs from edgelists are only present on
    # DiGraph classes. Disable the Graph check for now and assume inputs are
    # symmetric DiGraphs.
    # if type(graph) is not Graph:
    #     raise Exception("input graph must be undirected")
    client = default_client()
    # Calling renumbering results in data that is sorted by degree
    input_graph.compute_renumber_edge_list(transposed=False)

    ddf = input_graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    futures = [client.submit(call_louvain,
                             Comms.get_session_id(),
                             wf[1],
                             src_col_name,
                             dst_col_name,
                             num_verts,
                             num_edges,
                             vertex_partition_offsets,
                             input_graph.aggregate_segment_offsets,
                             max_iter,
                             resolution,
                             workers=[wf[0]])
               for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(futures)

    # futures is a list of Futures containing tuples of (DataFrame, mod_score),
    # unpack using separate calls to client.submit with a callable to get
    # individual items.
    # FIXME: look into an alternate way (not returning a tuples, accessing
    # tuples differently, etc.) since multiple client.submit() calls may not be
    # optimal.
    df_futures = [client.submit(op.getitem, f, 0) for f in futures]
    mod_score_futures = [client.submit(op.getitem, f, 1) for f in futures]

    ddf = dask_cudf.from_delayed(df_futures)
    # Each worker should have computed the same mod_score
    mod_score = mod_score_futures[0].result()

    if input_graph.renumbered:
        # MG renumbering is lazy, but it's safe to assume it's been called at
        # this point if renumbered=True
        ddf = input_graph.unrenumber(ddf, "vertex")

    return (ddf, mod_score)
