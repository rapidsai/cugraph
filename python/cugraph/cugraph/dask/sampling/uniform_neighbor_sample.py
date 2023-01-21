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

import numpy
from dask.distributed import wait

import dask_cudf
import cudf
import cupy as cp

from pylibcugraph import ResourceHandle

from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

from cugraph.dask.comms import comms as Comms

src_n = "sources"
dst_n = "destinations"
indices_n = "indices"


def create_empty_df(indices_t, weight_t):
    df = cudf.DataFrame(
        {
            src_n: numpy.empty(shape=0, dtype=indices_t),
            dst_n: numpy.empty(shape=0, dtype=indices_t),
            indices_n: numpy.empty(shape=0, dtype=weight_t),
        }
    )
    return df


def convert_to_cudf(cp_arrays, weight_t):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_sources, cupy_destinations, cupy_indices = cp_arrays

    df = cudf.DataFrame()
    df[src_n] = cupy_sources
    df[dst_n] = cupy_destinations
    df[indices_n] = cupy_indices

    if weight_t == "int32":
        df.indices = df.indices.astype("int32")
    elif weight_t == "int64":
        df.indices = df.indices.astype("int64")

    return df


def _call_plc_uniform_neighbor_sample(
    sID, mg_graph_x, st_x, fanout_vals, with_replacement, weight_t
):
    cp_arrays = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        input_graph=mg_graph_x,
        start_list=st_x,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
    )
    return convert_to_cudf(cp_arrays, weight_t)


def uniform_neighbor_sample(
    input_graph, start_list, fanout_vals, with_replacement=True
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


    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 4 dask_cudf.Series

        ddf['sources']: dask_cudf.Series
            Contains the source vertices from the sampling result
        ddf['destinations']: dask_cudf.Series
            Contains the destination vertices from the sampling result
        ddf['indices']: dask_cudf.Series
            Contains the indices from the sampling result for path
            reconstruction
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
        start_list = input_graph.lookup_internal_vertex_id(start_list).compute()

    client = input_graph._client

    session_id = Comms.get_session_id()
    start_list = cp.array_split(start_list.values, input_graph._npartitions)

    result = [
        client.submit(
            _call_plc_uniform_neighbor_sample,
            session_id,
            input_graph._plc_graph[w],
            start_list[i],
            fanout_vals,
            with_replacement,
            weight_t=weight_t,
            workers=[w],
            allow_other_workers=False,
            pure=False,
        )
        for i, w in enumerate(Comms.get_workers())
    ]

    ddf = dask_cudf.from_delayed(
        result, meta=create_empty_df(indices_t, weight_t), verify_meta=False
    ).persist()
    wait(ddf)
    wait([r.release() for r in result])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    return ddf
