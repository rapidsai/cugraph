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

from dask.distributed import wait
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import operator as op

from pylibcugraph import ResourceHandle
from cugraph.dask.common.input_utils import get_distributed_data
from pylibcugraph import node2vec as pylibcugraph_node2vec


def convert_to_cudf(cp_paths, unrenumber=False):
    """
    Creates cudf Series from cupy arrays from pylibcugraph wrapper
    """
    # FIXME: perform the SG unrenumbering at this layer
    return cudf.Series(cp_paths)


def _call_plc_node2vec(sID, mg_graph_x, st_vtx, max_depth, compress_result, p, q):
    return pylibcugraph_node2vec(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        seed_array=st_vtx,
        max_depth=max_depth,
        compress_result=compress_result,
        p=p,
        q=q,
    )


def node2vec(input_graph, start_vertices, max_depth=1, compress_result=True, p=1.0, q=1.0):
    """
    Computes random walks for each node in 'start_vertices', under the
    node2vec sampling framework.

    Note: This is a pylibcugraph-enabled algorithm, which requires that the
    graph was created with legacy_renum_only=True.

    References
    ----------

    A Grover, J Leskovec: node2vec: Scalable Feature Learning for Networks,
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining, https://arxiv.org/abs/1607.00653

    Parameters
    ----------
    input_graph : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.

    start_vertices: int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame. Only supports int32 currently.

    max_depth: int, optional (default=1)
        The maximum depth of the random walks. If not specified, the maximum
        depth is set to 1.

    compress_result: bool, optional (default=True)
        If True, coalesced paths are returned with a sizes array with offsets.
        Otherwise padded paths are returned with an empty sizes array.

    p: float, optional (default=1.0, [0 < p])
        Return factor, which represents the likelihood of backtracking to
        a previous node in the walk. A higher value makes it less likely to
        sample a previously visited node, while a lower value makes it more
        likely to backtrack, making the walk "local". A positive float.

    q: float, optional (default=1.0, [0 < q])
        In-out factor, which represents the likelihood of visiting nodes
        closer or further from the outgoing node. If q > 1, the random walk
        is likelier to visit nodes closer to the outgoing node. If q < 1, the
        random walk is likelier to visit nodes further from the outgoing node.
        A positive float.

    Returns
    -------
    vertex_paths : dask_cudf.Series or dask_cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: dask_cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    sizes: dask_cudf.Series
        The path size or sizes in case of coalesced paths.

    """

    # Initialize dask client
    client = input_graph._client

    if (not isinstance(max_depth, int)) or (max_depth < 1):
        raise ValueError(
            f"'max_depth' must be a positive integer, " f"got: {max_depth}"
        )
    if not isinstance(compress_result, bool):
        raise ValueError(
            f"'compress_result' must be a bool, " f"got: {compress_result}"
        )
    if (not isinstance(p, float)) or (p <= 0.0):
        raise ValueError(f"'p' must be a positive float, got: {p}")
    if (not isinstance(q, float)) or (q <= 0.0):
        raise ValueError(f"'q' must be a positive float, got: {q}")

    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)
    
    if input_graph.renumbered is True:
        start_vertices = input_graph.lookup_internal_vertex_id(start_vertices).compute()
        start_vertices_type = input_graph.edgelist.edgelist_df.dtypes[0]
    else:
        # FIXME: Get the 'src' column names instead and retrieve the type
        start_vertices_type = input_graph.input_df.dtypes[0]

    # FIXME: No need to recreate a dask_cudf since start_vertices was already a dask_cudf
    start_vertices = dask_cudf.from_cudf(
        start_vertices, npartitions=min(input_graph._npartitions, len(start_vertices))
    )
    start_vertices = start_vertices.astype(start_vertices_type)
    start_vertices = get_distributed_data(start_vertices)
    wait(start_vertices)
    start_vertices = start_vertices.worker_to_parts


    result = [
        client.submit(
            _call_plc_node2vec,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            start_vertices[w][0],
            max_depth,
            compress_result,
            p,
            q,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    # futures is a list of Futures containing tuples of cudf.Series,
    # unpack using separate calls to client.submit with a callable to get
    # individual items.
    # FIXME: look into an alternate way (not returning a tuples, accessing
    # tuples differently, etc.) since multiple client.submit() calls may not be
    # optimal.
    result_vertex_paths = [client.submit(op.getitem, f, 0) for f in result]
    result_edge_wgt_paths = [client.submit(op.getitem, f, 1) for f in result]
    result_sizes = [client.submit(op.getitem, f, 2) for f in result]
    

    cudf_vertex_paths = [
        client.submit(convert_to_cudf, cp_vertex_paths)
        for cp_vertex_paths in result_vertex_paths
    ]

    cudf_edge_wgt_paths = [
        client.submit(convert_to_cudf, cp_edge_wgt_paths)
        for cp_edge_wgt_paths in result_edge_wgt_paths
    ]

    cudf_edge_sizes = [
        client.submit(convert_to_cudf, cp_sizes)
        for cp_sizes in result_sizes
    ]

    wait([cudf_vertex_paths, cudf_edge_wgt_paths, cudf_edge_sizes])

    ddf_vertex_paths = dask_cudf.from_delayed(cudf_vertex_paths).persist()
    ddf_edge_wgt_paths = dask_cudf.from_delayed(cudf_edge_wgt_paths).persist()
    ddf_sizes = dask_cudf.from_delayed(cudf_sizes).persist()

    # Wait until the inactive futures are released
    wait(
        [
            (r.release(), c_v.release(), c_e.release(), c_s.release())
            for r, c_v, c_e, c_s in zip(result, cudf_vertex_paths, cudf_edge_wgt_paths, cudf_sizes)
        ]
    )

    # FIXME: fix unrenumbering
    """
    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")
    """

    return ddf_vertex_paths, ddf_edge_wgt_paths, ddf_sizes
