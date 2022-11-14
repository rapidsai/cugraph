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
from cugraph.dask.common.input_utils import get_distributed_data
import operator as op

from pylibcugraph import ResourceHandle, ego_graph as pylibcugraph_ego_graph


def _call_ego_graph(
    sID,
    mg_graph_x,
    n,
    radius,
    do_expensive_check,
):
    return pylibcugraph_ego_graph(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        source_vertices=n,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(*cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    if len(cp_arrays) == 1:
        # offsets array
        return cudf.Series(cp_arrays[0])
    else:
        cupy_src, cupy_dst, cupy_weight = cp_arrays
        df = cudf.DataFrame()
        df["src"] = cupy_src
        df["dst"] = cupy_dst
        df["weight"] = cupy_weight

        return df


def ego_graph(input_graph, n, radius=1, center=True):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    n : int, list or cudf.Series, cudf.DataFrame
        A node or a list or cudf.Series of nodes or a cudf.DataFrame if nodes
        are represented with multiple columns. If a cudf.DataFrame is provided,
        only the first row is taken as the node input.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    Returns
    -------
    G_ego : cuGraph.Graph
        A graph descriptor with a minimum spanning tree or forest.

    """

    # Initialize dask client
    client = input_graph._client

    if n is not None:
        if isinstance(n, int):
            n = [n]
        if isinstance(n, list):
            n = cudf.Series(n)
        if not isinstance(n, cudf.Series):
            raise TypeError(
                f"'n' must be either a list or a cudf.Series," f"got: {n.dtype}"
            )

        # n uses "external" vertex IDs, but since the graph has been
        # renumbered, the node ID must also be renumbered.
        if input_graph.renumbered:
            n = input_graph.lookup_internal_vertex_id(n).compute()
            n_type = input_graph.edgelist.edgelist_df.dtypes[0]
        else:
            n_type = input_graph.input_df.dtypes[0]

    n = dask_cudf.from_cudf(n, npartitions=min(input_graph._npartitions, len(n)))
    n = n.astype(n_type)

    n = get_distributed_data(n)

    wait(n)
    n = n.worker_to_parts

    do_expensive_check = False

    result = [
        client.submit(
            _call_ego_graph,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            n[w][0],
            radius,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]
    wait(result)

    result_src = [client.submit(op.getitem, f, 0) for f in result]
    result_dst = [client.submit(op.getitem, f, 1) for f in result]
    result_wgt = [client.submit(op.getitem, f, 2) for f in result]
    result_offset = [client.submit(op.getitem, f, 3) for f in result]

    cudf_edge = [
        client.submit(convert_to_cudf, cp_src, cp_dst, cp_wgt)
        for cp_src, cp_dst, cp_wgt in zip(result_src, result_dst, result_wgt)
    ]

    cudf_offset = [
        client.submit(convert_to_cudf, cp_offsets) for cp_offsets in result_offset
    ]

    wait(cudf_edge)
    wait(cudf_offset)

    ddf = dask_cudf.from_delayed(cudf_edge).persist()
    offsets = dask_cudf.from_delayed(cudf_offset).persist()
    wait(ddf)
    wait(offsets)
    # Wait until the inactive futures are released
    wait(
        [
            (r.release(), c_e.release(), c_o.release())
            for r, c_e, c_o in zip(result, cudf_edge, cudf_offset)
        ]
    )

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "src")
        ddf = input_graph.unrenumber(ddf, "dst")

    # FIXME: Temporary return.
    return ddf, offsets
