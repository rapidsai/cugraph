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

from __future__ import annotations

import numpy
from dask import delayed
from dask.distributed import wait, Lock, get_client
from cugraph.dask.common.input_utils import get_distributed_data

import dask_cudf
import cudf
import cupy as cp

from pylibcugraph import ResourceHandle

from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

from cugraph.dask.comms import comms as Comms

from typing import Sequence, List, Union, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cugraph import Graph

src_n = "sources"
dst_n = "destinations"
indices_n = "indices"
weight_n = "weight"
edge_id_n = "edge_id"
edge_type_n = "edge_type"
batch_id_n = "batch_id"
offsets_n = "offsets"
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


def create_empty_df_with_edge_props(indices_t, weight_t, return_offsets=False):
    if return_offsets:
        df = cudf.DataFrame(
            {
                src_n: numpy.empty(shape=0, dtype=indices_t),
                dst_n: numpy.empty(shape=0, dtype=indices_t),
                weight_n: numpy.empty(shape=0, dtype=weight_t),
                edge_id_n: numpy.empty(shape=0, dtype=indices_t),
                edge_type_n: numpy.empty(shape=0, dtype="int32"),
                hop_id_n: numpy.empty(shape=0, dtype="int32"),
            }
        )
        empty_df_offsets = cudf.DataFrame(
            {
                offsets_n: numpy.empty(shape=0, dtype="int32"),
                batch_id_n: numpy.empty(shape=0, dtype="int32"),
            }
        )
        return df, empty_df_offsets
    else:
        df = cudf.DataFrame(
            {
                src_n: numpy.empty(shape=0, dtype=indices_t),
                dst_n: numpy.empty(shape=0, dtype=indices_t),
                weight_n: numpy.empty(shape=0, dtype=weight_t),
                edge_id_n: numpy.empty(shape=0, dtype=indices_t),
                edge_type_n: numpy.empty(shape=0, dtype="int32"),
                hop_id_n: numpy.empty(shape=0, dtype="int32"),
                batch_id_n: numpy.empty(shape=0, dtype="int32"),
            }
        )
        return df


def convert_to_cudf(cp_arrays, return_offsets=False):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    df = cudf.DataFrame()

    (
        sources,
        destinations,
        weights,
        edge_ids,
        edge_types,
        batch_ids,
        offsets,
        hop_ids,
    ) = cp_arrays

    df[src_n] = sources
    df[dst_n] = destinations
    df[weight_n] = weights
    df[edge_id_n] = edge_ids
    df[edge_type_n] = edge_types
    df[hop_id_n] = hop_ids

    if return_offsets:
        offsets_df = cudf.DataFrame(
            {
                batch_id_n: batch_ids,
                offsets_n: offsets[:-1],
            }
        )
        return df, offsets_df
    else:
        if len(batch_ids) > 0:
            batch_ids = cudf.Series(batch_ids).repeat(cp.diff(offsets))
            batch_ids.reset_index(drop=True, inplace=True)

        df[batch_id_n] = batch_ids
        return df


def _call_plc_uniform_neighbor_sample(
    sID,
    mg_graph_x,
    st_x,
    label_list,
    label_to_output_comm_rank,
    fanout_vals,
    with_replacement,
    random_state=None,
    return_offsets=False,
):
    start_list_x = st_x[start_col_name]
    batch_id_list_x = st_x[batch_col_name] if batch_col_name in st_x else None
    cp_arrays = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        input_graph=mg_graph_x,
        start_list=start_list_x,
        label_list=label_list,
        label_to_output_comm_rank=label_to_output_comm_rank,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        batch_id_list=batch_id_list_x,
        random_state=random_state,
    )
    return convert_to_cudf(
        cp_arrays, return_offsets=return_offsets
    )


def _mg_call_plc_uniform_neighbor_sample(
    client,
    session_id,
    input_graph,
    ddf,
    label_list,
    label_to_output_comm_rank,
    fanout_vals,
    with_replacement,
    weight_t,
    indices_t,
    random_state,
    return_offsets=False,
):
    result = [
        client.submit(
            _call_plc_uniform_neighbor_sample,
            session_id,
            input_graph._plc_graph[w],
            ddf[w][0],
            label_list,
            label_to_output_comm_rank,
            fanout_vals,
            with_replacement,
            # FIXME accept and properly transmute a numpy/cupy random state.
            random_state=hash((random_state, i)),
            workers=[w],
            allow_other_workers=False,
            pure=False,
            return_offsets=return_offsets,
        )
        for i, w in enumerate(Comms.get_workers())
    ]

    empty_df = (
        create_empty_df_with_edge_props(
            indices_t, weight_t, return_offsets=return_offsets
        )
    )

    if return_offsets:
        result = [delayed(lambda x: x, nout=2)(r) for r in result]
        ddf = dask_cudf.from_delayed(
            [r[0] for r in result], meta=empty_df[0], verify_meta=False
        ).persist()
        ddf_offsets = dask_cudf.from_delayed(
            [r[1] for r in result], meta=empty_df[1], verify_meta=False
        ).persist()
        wait(ddf)
        wait(ddf_offsets)
        wait([r.release() for r in result])
        return ddf, ddf_offsets
    else:
        ddf = dask_cudf.from_delayed(result, meta=empty_df, verify_meta=False).persist()
        wait(ddf)
        wait([r.release() for r in result])
        return ddf


def uniform_neighbor_sample(
    input_graph: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    batch_id_list: Sequence = None,
    label_list: Sequence = None,
    label_to_output_comm_rank: bool = None,
    random_state: int = None,
    return_offsets: bool = False,
    _multiple_clients: bool = False,
) -> Union[dask_cudf.DataFrame, Tuple[dask_cudf.DataFrame, dask_cudf.DataFrame]]:
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph, which contains connectivity information as dask cudf
        edge list dataframe

    start_list : int, list, cudf.Series, or dask_cudf.Series (int32 or int64)
        a list of starting vertices for sampling

    fanout_vals : list
        List of branching out (fan-out) degrees per starting vertex for each
        hop level.

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    batch_id_list: cudf.Series or dask_cudf.Series (int32), optional (default=None)
        List of batch ids that will be returned with the sampled edges.

    label_list: cudf.Series or dask_cudf.Series (int32), optional (default=None)
        List of unique batch id labels.  Used along with
        label_to_output_comm_rank to assign batch ids to GPUs.

    label_to_out_comm_rank: cudf.Series or dask_cudf.Series (int32),
    optional (default=None)
        List of output GPUs (by rank) corresponding to batch
        id labels in the label list.  Used to assign each batch
        id to a GPU.

    random_state: int, optional
        Random seed to use when making sampling calls.

    return_offsets: bool, optional (default=False)
        Whether to return the sampling results with batch ids
        included as one dataframe, or to instead return two
        dataframes, one with sampling results and one with
        batch ids and their start offsets per rank.

    _multiple_clients: bool, optional (default=False)
        internal flag to ensure sampling works with multiple dask clients
        set to True to prevent hangs in multi-client environment

    Returns
    -------
    result : dask_cudf.DataFrame or Tuple[dask_cudf.DataFrame, dask_cudf.DataFrame]
        GPU distributed data frame containing several dask_cudf.Series
        If return_offsets=False:
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

        If return_offsets=True:
            df['sources']: cudf.Series
                Contains the source vertices from the sampling result
            df['destinations']: cudf.Series
                Contains the destination vertices from the sampling result
            df['edge_weight']: cudf.Series
                Contains the edge weights from the sampling result
            df['edge_id']: cudf.Series
                Contains the edge ids from the sampling result
            df['edge_type']: cudf.Series
                Contains the edge types from the sampling result
            df['hop_id']: cudf.Series
                Contains the hop ids from the sampling result

            offsets_df['batch_id']: cudf.Series
                Contains the batch ids from the sampling result
            offsets_df['offsets']: cudf.Series
                Contains the offsets of each batch in the sampling result
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
    elif batch_id_list is None:
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

    start_list = start_list.rename(start_col_name)
    if batch_id_list is not None:
        batch_id_list = batch_id_list.rename(batch_col_name)
        if hasattr(start_list, "compute"):
            # mg input
            start_list = start_list.to_frame()
            batch_id_list = batch_id_list.to_frame()
            ddf = start_list.merge(
                batch_id_list,
                how="left",
                left_index=True,
                right_index=True,
            )
        else:
            # sg input
            ddf = cudf.concat(
                [
                    start_list,
                    batch_id_list,
                ],
                axis=1,
            )
    else:
        ddf = start_list.to_frame()

    if input_graph.renumbered:
        ddf = input_graph.lookup_internal_vertex_id(ddf, column_name=start_col_name)

    if hasattr(ddf, "compute"):
        ddf = get_distributed_data(ddf)
        wait(ddf)
        ddf = ddf.worker_to_parts
    else:
        splits = cp.array_split(cp.arange(len(ddf)), len(Comms.get_workers()))
        ddf = {w: [ddf.iloc[splits[i]]] for i, w in enumerate(Comms.get_workers())}

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
                    label_list=label_list,
                    label_to_output_comm_rank=label_to_output_comm_rank,
                    fanout_vals=fanout_vals,
                    with_replacement=with_replacement,
                    weight_t=weight_t,
                    indices_t=indices_t,
                    random_state=random_state,
                    return_offsets=return_offsets,
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
            label_list=label_list,
            label_to_output_comm_rank=label_to_output_comm_rank,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            weight_t=weight_t,
            indices_t=indices_t,
            random_state=random_state,
            return_offsets=return_offsets,
        )

    if return_offsets:
        ddf, offsets_ddf = ddf
    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    if return_offsets:
        return ddf, offsets_ddf

    return ddf
