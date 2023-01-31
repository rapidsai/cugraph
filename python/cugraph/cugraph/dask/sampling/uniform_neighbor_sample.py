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

import numpy
from dask.distributed import wait, Lock, get_client
from cugraph.dask.common.input_utils import get_distributed_data


import dask_cudf
import cudf
import cupy as cp

from pylibcugraph import ResourceHandle

from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

from cugraph.dask.comms import comms as Comms

src_n = "sources"
dst_n = "destinations"
indices_n = "indices"
weight_n = "weight"
edge_id_n = "edge_id"
edge_type_n = "edge_type"
batch_id_n = "batch_id"
hop_id_n = "hop_id"

start_col_name = "_START_"
batch_col_name = "_BATCH_"


def create_empty_df(indices_t, weight_t):
    df = cudf.DataFrame(
        {
            src_n: numpy.empty(shape=0, dtype=indices_t),
            dst_n: numpy.empty(shape=0, dtype=indices_t),
            indices_n: numpy.empty(shape=0, dtype=weight_t),
        }
    )
    return df


def create_empty_df_with_edge_props(indices_t, weight_t):
    df = cudf.DataFrame(
        {
            src_n: numpy.empty(shape=0, dtype=indices_t),
            dst_n: numpy.empty(shape=0, dtype=indices_t),
            weight_n: numpy.empty(shape=0, dtype=weight_t),
            edge_id_n: numpy.empty(shape=0, dtype=indices_t),
            edge_type_n: numpy.empty(shape=0, dtype="int32"),
            batch_id_n: numpy.empty(shape=0, dtype="int32"),
            hop_id_n: numpy.empty(shape=0, dtype="int32"),
        }
    )
    return df


def convert_to_cudf(cp_arrays, weight_t, with_edge_properties):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    df = cudf.DataFrame()

    if with_edge_properties:
        (
            sources,
            destinations,
            weights,
            edge_ids,
            edge_types,
            batch_ids,
            hop_ids,
        ) = cp_arrays

        df[src_n] = sources
        df[dst_n] = destinations
        df[weight_n] = weights
        df[edge_id_n] = edge_ids
        df[edge_type_n] = edge_types
        df[batch_id_n] = batch_ids
        df[hop_id_n] = hop_ids
    else:
        cupy_sources, cupy_destinations, cupy_indices = cp_arrays

        df[src_n] = cupy_sources
        df[dst_n] = cupy_destinations
        df[indices_n] = cupy_indices

        if weight_t == "int32":
            df.indices = df.indices.astype("int32")
        elif weight_t == "int64":
            df.indices = df.indices.astype("int64")

    return df


def _call_plc_uniform_neighbor_sample(
    sID,
    mg_graph_x,
    st_x,
    fanout_vals,
    with_replacement,
    weight_t,
    with_edge_properties,
    random_state=None,
):
    start_list_x = st_x[start_col_name]
    batch_id_list_x = st_x[batch_col_name] if batch_col_name in st_x else None
    cp_arrays = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        input_graph=mg_graph_x,
        start_list=start_list_x,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        with_edge_properties=with_edge_properties,
        batch_id_list=batch_id_list_x,
        random_state=random_state,
    )
    return convert_to_cudf(cp_arrays, weight_t, with_edge_properties)


def _mg_call_plc_uniform_neighbor_sample(
    client,
    session_id,
    input_graph,
    ddf,
    fanout_vals,
    with_replacement,
    weight_t,
    indices_t,
    with_edge_properties,
    random_state,
):
    result = [
        client.submit(
            _call_plc_uniform_neighbor_sample,
            session_id,
            input_graph._plc_graph[w],
            ddf[w][0],
            fanout_vals,
            with_replacement,
            weight_t=weight_t,
            with_edge_properties=with_edge_properties,
            # FIXME accept and properly transmute a numpy/cupy random state.
            random_state=hash((random_state, i)),
            workers=[w],
            allow_other_workers=False,
            pure=False,
        )
        for i, w in enumerate(Comms.get_workers())
    ]

    empty_df = (
        create_empty_df_with_edge_props(indices_t, weight_t)
        if with_edge_properties
        else create_empty_df(indices_t, weight_t)
    )

    ddf = dask_cudf.from_delayed(result, meta=empty_df, verify_meta=False).persist()
    wait(ddf)
    wait([r.release() for r in result])
    return ddf


def uniform_neighbor_sample(
    input_graph,
    start_list,
    fanout_vals,
    with_replacement=True,
    with_edge_properties=False,
    batch_id_list=None,
    random_state=None,
    _multiple_clients=False,
):
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Note: This is a pylibcugraph-enabled algorithm, which requires that the
    graph was created with legacy_renum_only=True.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph, which contains connectivity information as dask cudf
        edge list dataframe

    start_list : list or cudf.Series (int32)
        a list of starting vertices for sampling

    fanout_vals : list (int32)
        List of branching out (fan-out) degrees per starting vertex for each
        hop level.

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    with_edge_properties: bool, optional (default=False)
        Flag to specify whether to return edge properties (weight, edge id,
        edge type, batch id, hop id) with the sampled edges.

    batch_id_list: list (int32)
        List of batch ids that will be returned with the sampled edges if
        with_edge_properties is set to True.

    random_state: int, optional
        Random seed to use when making sampling calls.

    _multiple_clients: bool, optional (default=False)
        internal flag to ensure sampling works with multiple dask clients
        set to True to prevent hangs in multi-client environment

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 4 dask_cudf.Series

        If with_edge_properties=True:
            ddf['sources']: dask_cudf.Series
                Contains the source vertices from the sampling result
            ddf['destinations']: dask_cudf.Series
                Contains the destination vertices from the sampling result
            ddf['indices']: dask_cudf.Series
                Contains the indices from the sampling result for path
                reconstruction

        If with_edge_properties=False:
            df['sources']: dask_cudf.Series
                Contains the source vertices from the sampling result
            df['destinations']: dask_cudf.Series
                Contains the destination vertices from the sampling result
            df['edge_weight']: dask_cudf.Series
                Contains the edge weights from the sampling result
            df['edge_id']: dask_cudf.Series
                Contains the edge ids from the sampling result
            df['edge_type']: dask_cudf.Series
                Contains the edge types from the sampling result
            df['batch_id']: dask_cudf.Series
                Contains the batch ids from the sampling result
            df['hop_id']: dask_cudf.Series
                Contains the hop ids from the sampling result
    """

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list,
            dtype=input_graph.edgelist.edgelist_df[
                input_graph.renumber_map.renumbered_src_col_name
            ].dtype,
        )

    elif with_edge_properties and batch_id_list is None:
        batch_id_list = cudf.Series(cp.zeros(len(start_list), dtype="int32"))

    # fanout_vals must be a host array!
    # FIXME: ensure other sequence types (eg. cudf Series) can be handled.
    if isinstance(fanout_vals, list):
        fanout_vals = numpy.asarray(fanout_vals, dtype="int32")
    else:
        raise TypeError("fanout_vals must be a list, " f"got: {type(fanout_vals)}")

    if "value" in input_graph.edgelist.edgelist_df:
        weight_t = input_graph.edgelist.edgelist_df["value"].dtype
    else:
        weight_t = "float32"

    if "_SRC_" in input_graph.edgelist.edgelist_df:
        indices_t = input_graph.edgelist.edgelist_df["_SRC_"].dtype
    elif src_n in input_graph.edgelist.edgelist_df:
        indices_t = input_graph.edgelist.edgelist_df[src_n].dtype
    else:
        indices_t = numpy.int32

    if input_graph.renumbered:
        start_list = input_graph.lookup_internal_vertex_id(start_list)

    start_list = start_list.rename(start_col_name).to_frame()
    if batch_id_list is not None:
        ddf = start_list.join(batch_id_list.rename(batch_col_name))
    else:
        ddf = start_list

    if isinstance(ddf, cudf.DataFrame):
        splits = cp.array_split(cp.arange(len(ddf)), len(Comms.get_workers()))
        ddf = {w: [ddf.iloc[splits[i]]] for i, w in enumerate(Comms.get_workers())}

    else:
        ddf = get_distributed_data(ddf)
        wait(ddf)
        ddf = ddf.worker_to_parts

    client = get_client()
    session_id = Comms.get_session_id()
    if _multiple_clients:
        # Distributed centralized lock to allow
        # two disconnected processes (clients) to coordinate a lock
        # https://docs.dask.org/en/stable/futures.html?highlight=lock#distributed.Lock
        lock = Lock("plc_graph_access")
        if lock.acquire(timeout=100):
            try:
                ddf = _mg_call_plc_uniform_neighbor_sample(
                    client=client,
                    session_id=session_id,
                    input_graph=input_graph,
                    ddf=ddf,
                    fanout_vals=fanout_vals,
                    with_replacement=with_replacement,
                    weight_t=weight_t,
                    indices_t=indices_t,
                    with_edge_properties=with_edge_properties,
                    random_state=random_state,
                )
            finally:
                lock.release()
        else:
            raise RuntimeError(
                "Failed to acquire lock(plc_graph_access) while trying to sampling"
            )
    else:
        ddf = _mg_call_plc_uniform_neighbor_sample(
            client=client,
            session_id=session_id,
            input_graph=input_graph,
            ddf=ddf,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            weight_t=weight_t,
            indices_t=indices_t,
            with_edge_properties=with_edge_properties,
            random_state=random_state,
        )

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    return ddf
