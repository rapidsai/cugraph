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
from dask.distributed import wait, default_client

import dask_cudf
import cudf

from pylibcugraph import ResourceHandle

from pylibcugraph import \
    uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.dask.comms import comms as Comms


def convert_to_cudf(cp_arrays, weight_t):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_sources, cupy_destinations, cupy_indices = cp_arrays

    df = cudf.DataFrame()
    df["sources"] = cupy_sources
    df["destinations"] = cupy_destinations
    df["indices"] = cupy_indices

    if weight_t == "int32":
        df.indices = df.indices.astype("int32")
    elif weight_t == "int64":
        df.indices = df.indices.astype("int64")

    return df


def uniform_neighbor_sample(input_graph,
                            start_list,
                            fanout_vals,
                            with_replacement=True):
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

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
    # Initialize dask client
    client = default_client()
    # FIXME: 'legacy_renum_only' will not trigger the C++ renumbering
    # In the future, once all the algos follow the C/Pylibcugraph path,
    # compute_renumber_edge_list will only be used for multicolumn and
    # string vertices since the renumbering will be done in pylibcugraph
    input_graph.compute_renumber_edge_list(
        transposed=False, legacy_renum_only=True)

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(start_list, dtype='int32')

    if start_list.dtype != "int32":
        raise ValueError(f"'start_list' must have int32 values, "
                         f"got: {start_list.dtype}")

    # fanout_vals must be a host array!
    # FIXME: ensure other sequence types (eg. cudf Series) can be handled.
    if isinstance(fanout_vals, list):
        fanout_vals = numpy.asarray(fanout_vals, dtype="int32")
    else:
        raise TypeError("fanout_vals must be a list, "
                        f"got: {type(fanout_vals)}")

    weight_t = input_graph.edgelist.edgelist_df["value"].dtype

    # start_list uses "external" vertex IDs, but if the graph has been
    # renumbered, the start vertex IDs must also be renumbered.
    if input_graph.renumbered:
        start_list = input_graph.lookup_internal_vertex_id(
            start_list).compute()

    start_list = dask_cudf.from_cudf(
        start_list, 
        npartitions=input_graph.npartitions
    )
    start_list = get_distributed_data(start_list)

    result = [
        client.submit(
            lambda sID, mg_graph_x, st_x:
            pylibcugraph_uniform_neighbor_sample(
                resource_handle=ResourceHandle(
                    Comms.get_handle(sID).getHandle()
                ),
                input_graph=mg_graph_x,
                start_list=st_x[0].to_cupy(),
                h_fan_out=fanout_vals,
                with_replacement=with_replacement,
                # FIXME: should we add this parameter as an option?
                do_expensive_check=False
            ),
            Comms.get_session_id(),
            mg_graph,
            st,
            workers=[w],
            key='cugraph.dask.sampling.'
                'uniform_neighbor_sample.call_plc_uni_nbr_smpl'
        )
        for mg_graph, (w, st) in zip(
            input_graph._plc_graph, start_list.worker_to_parts.items()
        )
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays, weight_t)
                   for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)
    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "sources", preserve_order=True)
        ddf = input_graph.unrenumber(ddf, "destinations", preserve_order=True)

    return ddf
