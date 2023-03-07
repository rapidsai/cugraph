# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import cugraph.dask.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
import dask_cudf
import cudf

import cugraph.dask as dcg
from pylibcugraph import ResourceHandle, k_core as pylibcugraph_k_core


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


def _call_plc_k_core(sID, mg_graph_x, k, degree_type, core_result, do_expensive_check):

    return pylibcugraph_k_core(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        k=k,
        degree_type=degree_type,
        core_result=core_result,
        do_expensive_check=do_expensive_check,
    )


def k_core(input_graph, k=None, core_number=None, degree_type="bidirectional"):
    """
    Compute the k-core of the graph G based on the out degree of its nodes. A
    k-core of a graph is a maximal subgraph that contains nodes of degree k or
    more. This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    input_graph : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph
        should contain undirected edges where undirected edges are represented
        as directed edges in both directions. While this graph can contain edge
        weights, they don't participate in the calculation of the k-core.
        The current implementation only supports undirected graphs.

    k : int, optional (default=None)
        Order of the core. This value must not be negative. If set to None, the
        main core is returned.

    degree_type: str (default="bidirectional")
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.

    core_number : cudf.DataFrame or das_cudf.DataFrame, optional (default=None)
        Precomputed core number of the nodes of the graph G containing two
        cudf.Series of size V: the vertex identifiers and the corresponding
        core number values. If set to None, the core numbers of the nodes are
        calculated internally.

        core_number['vertex'] : cudf.Series or dask_cudf.Series
            Contains the vertex identifiers
        core_number['values'] : cudf.Series or dask_cudf.Series
            Contains the core number of vertices

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing the K Core of the input graph

        ddf['src']: dask_cudf.Series
            Contains sources of the K Core
        ddf['dst']: dask_cudf.Series
            Contains destinations of the K Core

        and/or

        ddf['weights']: dask_cudf.Series
            Contains weights of the K Core


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
    >>> dg = cugraph.Graph(directed=False)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> KCore_df = dcg.k_core(dg)
    """

    if degree_type not in ["incoming", "outgoing", "bidirectional"]:
        raise ValueError(
            f"'degree_type' must be either incoming, "
            f"outgoing or bidirectional, got: {degree_type}"
        )

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    if core_number is None:
        core_number = dcg.core_number(input_graph)

    if input_graph.renumbered is True:

        if len(input_graph.renumber_map.implementation.col_names) > 1:
            cols = core_number.columns[:-1].to_list()
        else:
            cols = "vertex"

            core_number = input_graph.add_internal_vertex_id(
                core_number, "vertex", cols
            )

    if not isinstance(core_number, dask_cudf.DataFrame):
        if isinstance(core_number, cudf.DataFrame):
            # convert to dask_cudf in order to distribute the edges
            core_number = dask_cudf.from_cudf(core_number, input_graph._npartitions)

        else:
            raise TypeError(
                f"'core_number' must be either None or of"
                f"type cudf/dask_cudf, got: {type(core_number)}"
            )

    core_number = core_number.rename(columns={"core_number": "values"})
    if k is None:
        k = core_number["values"].max().compute()

    core_number = get_distributed_data(core_number)
    wait(core_number)
    core_number = core_number.worker_to_parts

    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_k_core,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            k,
            degree_type,
            core_number[w][0],
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

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
        ddf = input_graph.unrenumber(ddf, "src")
        ddf = input_graph.unrenumber(ddf, "dst")

    return ddf
