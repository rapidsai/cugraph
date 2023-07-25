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

import warnings

import numpy
from dask import delayed
from dask.distributed import Lock, get_client, wait

import dask_cudf
import cudf
import cupy as cp

from pylibcugraph import ResourceHandle

from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample
from pylibcugraph.utilities.api_tools import deprecated_warning_wrapper

from cugraph.dask.comms import comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.dask import get_n_workers

from typing import Sequence, List, Union, Tuple
from typing import TYPE_CHECKING

from cugraph.dask.common.part_utils import (
    get_persisted_df_worker_map,
    persist_dask_df_equal_parts_per_worker,
)

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

map_n = "map"
map_offsets_n = "renumber_map_offsets"

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


def create_empty_df_with_edge_props(
    indices_t, weight_t, return_offsets=False, renumber=False
):
    if renumber:
        empty_df_renumber = cudf.DataFrame(
            {
                map_n: numpy.empty(shape=0, dtype=indices_t),
                map_offsets_n: numpy.empty(shape=0, dtype="int32"),
            }
        )

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

        if renumber:
            return df, empty_df_offsets, empty_df_renumber
        else:
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
        if renumber:
            return df, empty_df_renumber
        else:
            return df


def convert_to_cudf(
    cp_arrays, weight_t, with_edge_properties, return_offsets=False, renumber=False
):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    df = cudf.DataFrame()

    if with_edge_properties:
        if renumber:
            (
                sources,
                destinations,
                weights,
                edge_ids,
                edge_types,
                batch_ids,
                offsets,
                hop_ids,
                renumber_map,
                renumber_map_offsets,
            ) = cp_arrays
        else:
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

        return_dfs = [df]

        if return_offsets:
            offsets_df = cudf.DataFrame(
                {
                    batch_id_n: batch_ids,
                    offsets_n: offsets[:-1],
                }
            )

            if renumber:
                offsets_df[map_offsets_n] = renumber_map_offsets[:-1]

            return_dfs.append(offsets_df)
        else:
            batch_ids_b = batch_ids
            if len(batch_ids_b) > 0:
                batch_ids_b = cudf.Series(batch_ids_b).repeat(cp.diff(offsets))
                batch_ids_b.reset_index(drop=True, inplace=True)

            df[batch_id_n] = batch_ids_b

        if renumber:
            renumber_df = cudf.DataFrame(
                {
                    "map": renumber_map,
                }
            )

            if not return_offsets:
                batch_ids_r = cudf.Series(batch_ids).repeat(
                    cp.diff(renumber_map_offsets)
                )
                batch_ids_r.reset_index(drop=True, inplace=True)
                renumber_df["batch_id"] = batch_ids_r

            return_dfs.append(renumber_df)

        return tuple(return_dfs)
    else:
        cupy_sources, cupy_destinations, cupy_indices = cp_arrays

        df[src_n] = cupy_sources
        df[dst_n] = cupy_destinations
        df[indices_n] = cupy_indices

        if cupy_indices is not None:
            if weight_t == "int32":
                df.indices = df.indices.astype("int32")
            elif weight_t == "int64":
                df.indices = df.indices.astype("int64")

        return (df,)


def __get_label_to_output_comm_rank(min_batch_id, max_batch_id, n_workers):
    num_batches = max_batch_id - min_batch_id + 1
    num_batches = int(num_batches)
    z = cp.zeros(num_batches, dtype="int32")
    s = cp.array_split(cp.arange(num_batches), n_workers)
    for i, t in enumerate(s):
        z[t] = i

    return z


def _call_plc_uniform_neighbor_sample(
    sID,
    mg_graph_x,
    st_x,
    keep_batches_together,
    n_workers,
    min_batch_id,
    max_batch_id,
    fanout_vals,
    with_replacement,
    weight_t,
    with_edge_properties,
    random_state=None,
    return_offsets=False,
    return_hops=True,
    prior_sources_behavior=None,
    deduplicate_sources=False,
    renumber=False,
):
    st_x = st_x[0]
    start_list_x = st_x[start_col_name]
    batch_id_list_x = st_x[batch_col_name] if batch_col_name in st_x else None

    label_list = None
    label_to_output_comm_rank = None
    if keep_batches_together:
        label_list = cp.arange(min_batch_id, max_batch_id + 1, dtype="int32")
        label_to_output_comm_rank = __get_label_to_output_comm_rank(
            min_batch_id, max_batch_id, n_workers
        )

    cp_arrays = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        input_graph=mg_graph_x,
        start_list=start_list_x,
        label_list=label_list,
        label_to_output_comm_rank=label_to_output_comm_rank,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        with_edge_properties=with_edge_properties,
        batch_id_list=batch_id_list_x,
        random_state=random_state,
        prior_sources_behavior=prior_sources_behavior,
        deduplicate_sources=deduplicate_sources,
        return_hops=return_hops,
        renumber=renumber,
    )
    return convert_to_cudf(
        cp_arrays,
        weight_t,
        with_edge_properties,
        return_offsets=return_offsets,
        renumber=renumber,
    )


def _call_plc_uniform_neighbor_sample_legacy(
    sID,
    mg_graph_x,
    st_x,
    label_list,
    label_to_output_comm_rank,
    fanout_vals,
    with_replacement,
    weight_t,
    with_edge_properties,
    random_state=None,
    return_offsets=False,
    return_hops=True,
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
        with_edge_properties=with_edge_properties,
        batch_id_list=batch_id_list_x,
        random_state=random_state,
        return_hops=return_hops,
    )

    output = convert_to_cudf(
        cp_arrays, weight_t, with_edge_properties, return_offsets=return_offsets
    )

    if isinstance(output, (list, tuple)) and len(output) == 1:
        return output[0]
    return output


def _mg_call_plc_uniform_neighbor_sample_legacy(
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
    with_edge_properties,
    random_state,
    return_offsets=False,
    return_hops=True,
):
    result = [
        client.submit(
            _call_plc_uniform_neighbor_sample_legacy,
            session_id,
            input_graph._plc_graph[w],
            ddf[w][0],
            label_list,
            label_to_output_comm_rank,
            fanout_vals,
            with_replacement,
            weight_t=weight_t,
            with_edge_properties=with_edge_properties,
            # FIXME accept and properly transmute a numpy/cupy random state.
            random_state=hash((random_state, i)),
            workers=[w],
            allow_other_workers=False,
            pure=False,
            return_offsets=return_offsets,
            return_hops=return_hops,
        )
        for i, w in enumerate(Comms.get_workers())
    ]

    empty_df = (
        create_empty_df_with_edge_props(
            indices_t, weight_t, return_offsets=return_offsets
        )
        if with_edge_properties
        else create_empty_df(indices_t, weight_t)
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


def _mg_call_plc_uniform_neighbor_sample(
    client,
    session_id,
    input_graph,
    ddf,
    keep_batches_together,
    min_batch_id,
    max_batch_id,
    fanout_vals,
    with_replacement,
    weight_t,
    indices_t,
    with_edge_properties,
    random_state,
    return_offsets=False,
    return_hops=True,
    prior_sources_behavior=None,
    deduplicate_sources=False,
    renumber=False,
):
    n_workers = None
    if keep_batches_together:
        n_workers = get_n_workers()

        if hasattr(min_batch_id, "compute"):
            min_batch_id = min_batch_id.compute()
        if hasattr(max_batch_id, "compute"):
            max_batch_id = max_batch_id.compute()

    result = [
        client.submit(
            _call_plc_uniform_neighbor_sample,
            session_id,
            input_graph._plc_graph[w],
            starts,
            keep_batches_together,
            n_workers,
            min_batch_id,
            max_batch_id,
            fanout_vals,
            with_replacement,
            weight_t=weight_t,
            with_edge_properties=with_edge_properties,
            # FIXME accept and properly transmute a numpy/cupy random state.
            random_state=hash((random_state, w)),
            return_offsets=return_offsets,
            return_hops=return_hops,
            prior_sources_behavior=prior_sources_behavior,
            deduplicate_sources=deduplicate_sources,
            renumber=renumber,
            allow_other_workers=False,
            pure=False,
        )
        for w, starts in ddf.items()
    ]
    del ddf

    empty_df = (
        create_empty_df_with_edge_props(
            indices_t,
            weight_t,
            return_offsets=return_offsets,
            renumber=renumber,
        )
        if with_edge_properties
        else create_empty_df(indices_t, weight_t)
    )
    if not isinstance(empty_df, (list, tuple)):
        empty_df = [empty_df]

    wait(result)

    nout = 1
    if return_offsets:
        nout += 1
    if renumber:
        nout += 1

    result_split = [delayed(lambda x: x, nout=nout)(r) for r in result]

    ddf = dask_cudf.from_delayed(
        [r[0] for r in result_split], meta=empty_df[0], verify_meta=False
    ).persist()
    return_dfs = [ddf]

    if return_offsets:
        ddf_offsets = dask_cudf.from_delayed(
            [r[1] for r in result_split], meta=empty_df[1], verify_meta=False
        ).persist()
        return_dfs.append(ddf_offsets)

    if renumber:
        ddf_renumber = dask_cudf.from_delayed(
            [r[-1] for r in result_split], meta=empty_df[-1], verify_meta=False
        ).persist()
        return_dfs.append(ddf_renumber)

    wait(return_dfs)
    wait([r.release() for r in result_split])
    wait([r.release() for r in result])
    del result

    if len(return_dfs) == 1:
        return return_dfs[0]
    else:
        return tuple(return_dfs)


def _uniform_neighbor_sample_legacy(
    input_graph: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    with_edge_properties: bool = False,
    batch_id_list: Sequence = None,
    label_list: Sequence = None,
    label_to_output_comm_rank: bool = None,
    random_state: int = None,
    return_offsets: bool = False,
    return_hops: bool = False,
    _multiple_clients: bool = False,
) -> Union[dask_cudf.DataFrame, Tuple[dask_cudf.DataFrame, dask_cudf.DataFrame]]:
    warnings.warn(
        "The batch_id_list, label_list, and label_to_output_comm_rank "
        "parameters are deprecated.  Consider using with_batch_ids, "
        "keep_batches_together, min_batch_id, and max_batch_id instead."
    )

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
                ddf = _mg_call_plc_uniform_neighbor_sample_legacy(
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
                    with_edge_properties=with_edge_properties,
                    random_state=random_state,
                    return_offsets=return_offsets,
                    return_hops=return_hops,
                )
            finally:
                lock.release()
        else:
            raise RuntimeError(
                "Failed to acquire lock(plc_graph_access) while trying to sampling"
            )
    else:
        ddf = _mg_call_plc_uniform_neighbor_sample_legacy(
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
            with_edge_properties=with_edge_properties,
            random_state=random_state,
            return_offsets=return_offsets,
            return_hops=return_hops,
        )

    if return_offsets:
        ddf, offsets_ddf = ddf
    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    if return_offsets:
        return ddf, offsets_ddf

    return ddf


uniform_neighbor_sample_legacy = deprecated_warning_wrapper(
    _uniform_neighbor_sample_legacy
)


def uniform_neighbor_sample(
    input_graph: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    with_edge_properties: bool = False,
    batch_id_list: Sequence = None,  # deprecated
    label_list: Sequence = None,  # deprecated
    label_to_output_comm_rank: bool = None,  # deprecated
    with_batch_ids: bool = False,
    keep_batches_together=False,
    min_batch_id=None,
    max_batch_id=None,
    random_state: int = None,
    return_offsets: bool = False,
    return_hops: bool = True,
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    renumber: bool = False,
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

    with_edge_properties: bool, optional (default=False)
        Flag to specify whether to return edge properties (weight, edge id,
        edge type, batch id, hop id) with the sampled edges.

    batch_id_list: cudf.Series or dask_cudf.Series (int32), optional (default=None)
        Deprecated.
        List of batch ids that will be returned with the sampled edges if
        with_edge_properties is set to True.

    label_list: cudf.Series or dask_cudf.Series (int32), optional (default=None)
        Deprecated.
        List of unique batch id labels.  Used along with
        label_to_output_comm_rank to assign batch ids to GPUs.

    label_to_out_comm_rank: cudf.Series or dask_cudf.Series (int32),
    optional (default=None)
        Deprecated.
        List of output GPUs (by rank) corresponding to batch
        id labels in the label list.  Used to assign each batch
        id to a GPU.
        Must be in ascending order (i.e. [0, 0, 1, 2]).

    with_batch_ids: bool, optional (default=False)
        Flag to specify whether batch ids are present in the start_list

    keep_batches_together: bool (optional, default=False)
        If True, will ensure that the returned samples for each batch are on the
        same partition.

    min_batch_id: int (optional, default=None)
        Required for the keep_batches_together option.  The minimum batch id.

    max_batch_id: int (optional, default=None)
        Required for the keep_batches_together option.  The maximum batch id.

    random_state: int, optional
        Random seed to use when making sampling calls.

    return_offsets: bool, optional (default=False)
        Whether to return the sampling results with batch ids
        included as one dataframe, or to instead return two
        dataframes, one with sampling results and one with
        batch ids and their start offsets per rank.

    return_hops: bool, optional (default=True)
        Whether to return the sampling results with hop ids
        corresponding to the hop where the edge appeared.
        Defaults to True.

    prior_sources_behavior: str (Optional)
        Options are "carryover", and "exclude".
        Default will leave the source list as-is.
        Carryover will carry over sources from previous hops to the
        current hop.
        Exclude will exclude sources from previous hops from reappearing
        as sources in future hops.

    deduplicate_sources: bool, optional (default=False)
        Whether to first deduplicate the list of possible sources
        from the previous destinations before performing next
        hop.

    renumber: bool, optional (default=False)
        Whether to renumber on a per-batch basis.  If True,
        will return the renumber map and renumber map offsets
        as an additional dataframe.

    _multiple_clients: bool, optional (default=False)
        internal flag to ensure sampling works with multiple dask clients
        set to True to prevent hangs in multi-client environment

    Returns
    -------
    result : dask_cudf.DataFrame or Tuple[dask_cudf.DataFrame, dask_cudf.DataFrame]
        GPU distributed data frame containing several dask_cudf.Series

        If with_edge_properties=True:
            ddf['sources']: dask_cudf.Series
                Contains the source vertices from the sampling result
            ddf['destinations']: dask_cudf.Series
                Contains the destination vertices from the sampling result
            ddf['indices']: dask_cudf.Series
                Contains the indices from the sampling result for path
                reconstruction

        If with_edge_properties=False:
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
                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['map']: dask_cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['offsets']: dask_cudf.Series
                        Contains the batch offsets for the renumber maps

            If return_offsets=True:
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
                df['hop_id']: dask_cudf.Series
                    Contains the hop ids from the sampling result

                offsets_df['batch_id']: dask_cudf.Series
                    Contains the batch ids from the sampling result
                offsets_df['offsets']: dask_cudf.Series
                    Contains the offsets of each batch in the sampling result
                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['map']: dask_cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['offsets']: dask_cudf.Series
                        Contains the batch offsets for the renumber maps
    """

    if (
        batch_id_list is not None
        or label_list is not None
        or label_to_output_comm_rank is not None
    ):
        if prior_sources_behavior or deduplicate_sources:
            raise ValueError(
                "unique sources, carry_over_sources, and deduplicate_sources"
                " are not supported with batch_id_list, label_list, and"
                " label_to_output_comm_rank.  Consider using with_batch_ids"
                " and keep_batches_together instead."
            )

        if renumber:
            raise ValueError(
                "renumber is not supported with batch_id_list, label_list, "
                "and label_to_output_comm_rank.  Consider using "
                "with_batch_ids and keep_batches_together instead."
            )

        return uniform_neighbor_sample_legacy(
            input_graph,
            start_list,
            fanout_vals,
            with_replacement=with_replacement,
            with_edge_properties=with_edge_properties,
            batch_id_list=batch_id_list,
            label_list=label_list,
            label_to_output_comm_rank=label_to_output_comm_rank,
            random_state=random_state,
            return_offsets=return_offsets,
            return_hops=return_hops,
            _multiple_clients=_multiple_clients,
        )

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list,
            dtype=input_graph.edgelist.edgelist_df[
                input_graph.renumber_map.renumbered_src_col_name
            ].dtype,
        )
    elif with_edge_properties and not with_batch_ids:
        if isinstance(start_list, (cudf.DataFrame, dask_cudf.DataFrame)):
            raise ValueError("expected 1d input for start list without batch ids")

        start_list = start_list.to_frame()
        if isinstance(start_list, dask_cudf.DataFrame):
            start_list = start_list.map_partitions(
                lambda df: df.assign(
                    **{batch_id_n: cudf.Series(cp.zeros(len(df), dtype="int32"))}
                )
            ).persist()
        else:
            start_list = start_list.reset_index(drop=True).assign(
                **{batch_id_n: cudf.Series(cp.zeros(len(start_list), dtype="int32"))}
            )

    if keep_batches_together and min_batch_id is None:
        raise ValueError(
            "must provide min_batch_id if using keep_batches_together option"
        )
    if keep_batches_together and max_batch_id is None:
        raise ValueError(
            "must provide max_batch_id if using keep_batches_together option"
        )
    if renumber and not keep_batches_together:
        raise ValueError(
            "mg uniform_neighbor_sample requires that keep_batches_together=True "
            "when performing renumbering."
        )

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

    if isinstance(start_list, (cudf.Series, dask_cudf.Series)):
        start_list = start_list.rename(start_col_name)
        ddf = start_list.to_frame()
    else:
        ddf = start_list
        columns = ddf.columns
        ddf = ddf.rename(
            columns={columns[0]: start_col_name, columns[-1]: batch_col_name}
        )

    if input_graph.renumbered:
        ddf = input_graph.lookup_internal_vertex_id(ddf, column_name=start_col_name)

    client = get_client()
    session_id = Comms.get_session_id()
    n_workers = get_n_workers()

    if isinstance(ddf, cudf.DataFrame):
        ddf = dask_cudf.from_cudf(ddf, npartitions=n_workers)

    ddf = ddf.repartition(npartitions=n_workers)
    ddf = persist_dask_df_equal_parts_per_worker(ddf, client)
    ddf = get_persisted_df_worker_map(ddf, client)

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
                    keep_batches_together=keep_batches_together,
                    min_batch_id=min_batch_id,
                    max_batch_id=max_batch_id,
                    fanout_vals=fanout_vals,
                    with_replacement=with_replacement,
                    weight_t=weight_t,
                    indices_t=indices_t,
                    with_edge_properties=with_edge_properties,
                    random_state=random_state,
                    return_offsets=return_offsets,
                    return_hops=return_hops,
                    prior_sources_behavior=prior_sources_behavior,
                    deduplicate_sources=deduplicate_sources,
                    renumber=renumber,
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
            keep_batches_together=keep_batches_together,
            min_batch_id=min_batch_id,
            max_batch_id=max_batch_id,
            fanout_vals=fanout_vals,
            with_replacement=with_replacement,
            weight_t=weight_t,
            indices_t=indices_t,
            with_edge_properties=with_edge_properties,
            random_state=random_state,
            return_offsets=return_offsets,
            return_hops=return_hops,
            prior_sources_behavior=prior_sources_behavior,
            deduplicate_sources=deduplicate_sources,
            renumber=renumber,
        )

    if return_offsets:
        if renumber:
            ddf, offsets_df, renumber_df = ddf
        else:
            ddf, offsets_ddf = ddf
    else:
        if renumber:
            ddf, renumber_df = ddf

    if input_graph.renumbered and not renumber:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    if return_offsets:
        if renumber:
            return ddf, offsets_df, renumber_df
        else:
            return ddf, offsets_ddf

    if renumber:
        return ddf, renumber_df

    return ddf
