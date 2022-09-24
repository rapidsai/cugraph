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
from cugraph.dask.common.input_utils import get_distributed_data
import dask_cudf
import cudf

from pylibcugraph import (ResourceHandle,
                          core_number as pylibcugraph_k_core
                          )


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_src_vertices, cupy_dst_vertices, cupy_weights = cp_arrays
    df = cudf.DataFrame()
    df["src"] = cupy_src_vertices
    df["dst"] = cupy_dst_vertices
    df["weights"] = cupy_weights

    return df


def _call_plc_k_core(sID,
                     mg_graph_x,
                     core_result,
                     do_expensive_check):
    return pylibcugraph_k_core(
        resource_handle=ResourceHandle(
            Comms.get_handle(sID).getHandle()
        ),
        graph=mg_graph_x,
        core_result=core_result,
        do_expensive_check=do_expensive_check
    )


def k_core(input_graph, core_number=None):
    """
    Compute the k-core of the graph G based on the out degree of its nodes. A
    k-core of a graph is a maximal subgraph that contains nodes of degree k or
    more. This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph
        should contain undirected edges where undirected edges are represented
        as directed edges in both directions. While this graph can contain edge
        weights, they don't participate in the calculation of the k-core.

    k : int, optional (default=None)
        Order of the core. This value must not be negative. If set to None, the
        main core is returned.

    core_number : cudf.DataFrame, optional (default=None)
        Precomputed core number of the nodes of the graph G containing two
        cudf.Series of size V: the vertex identifiers and the corresponding
        core number values. If set to None, the core numbers of the nodes are
        calculated internally.

        core_number['vertex'] : cudf.Series
            Contains the vertex identifiers
        core_number['values'] : cudf.Series
            Contains the core number of vertices

    Returns
    -------
    KCoreGraph : cuGraph.Graph
        K Core of the input graph

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
    >>> dg = cugraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> KCoreGraph = dcg.k_core(dg)
    """

    mytype = type(input_graph)
    KCoreGraph = mytype()

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    if not isinstance(core_number, dask_cudf.DataFrame):
        if isinstance(core_number, cudf.DataFrame):
            # convert to dask_cudf in order to distribute the edges
            core_number = dask_cudf.from_cudf(
                core_number, input_graph._npartitions)
        elif core_number is not None:
            raise TypeError(f"'core_number' must be either None or of"
                            f"type cudf/dask_cudf, got type{core_number}")

    if isinstance(core_number, dask_cudf.DataFrame):
        if input_graph.renumbered is True:
            core_number = input_graph.add_internal_vertex_id(
                core_number, "vertex", "vertex")

        data_core_number = get_distributed_data(core_number)
        data_core_number = data_core_number.worker_to_parts
    else:
        # core_number is None
        worker_list = Comms.get_workers()
        # map each worker to None
        data_core_number = dict.fromkeys(worker_list)

    # Initialize dask client
    client = input_graph._client

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_k_core,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            core_number,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w, core_number in data_core_number.items()
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays)
                   for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # FIXME: Dask doesn't always release it fast enough.
    # For instance if the algo is run several times with
    # the same PLC graph, the current iteration might try to cache
    # the past iteration's futures and this can cause a hang if some
    # of those futures get released midway
    del result
    del cudf_result

    if input_graph.renumbered:
        ddf, src_names = input_graph.unrenumber(
            ddf, "src", get_column_names=True)
        ddf, dst_names = input_graph.unrenumber(
            ddf, "dst", get_column_names=True)

    if input_graph.edgelist.weights:
        KCoreGraph.from_dask_cudf_edgelist(
            ddf, source=src_names, destination=dst_names,
            edge_attr="weight"
        )
    else:
        KCoreGraph.from_dask_cudf_edgelist(
            ddf, source=src_names, destination=dst_names,
        )

    return KCoreGraph
