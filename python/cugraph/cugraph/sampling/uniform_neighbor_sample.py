# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from pylibcugraph import ResourceHandle
from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

import numpy

import cudf
import cupy as cp
import warnings

from typing import Union, Tuple, Sequence, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cugraph import Graph


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, start_list):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes[0]
    if isinstance(start_list, cudf.Series):
        start_list_dtypes = start_list.dtype
    else:
        start_list_dtypes = start_list.dtypes[0]

    if start_list_dtypes != vertex_dtype:
        warning_msg = (
            "Uniform neighbor sample requires 'start_list' to match the graph's "
            f"'vertex' type. input graph's vertex type is: {vertex_dtype} and got "
            f"'start_list' of type: {start_list_dtypes}."
        )
        warnings.warn(warning_msg, UserWarning)
        start_list = start_list.astype(vertex_dtype)

    return start_list


def uniform_neighbor_sample(
    G: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    batch_id_list: Sequence = None,
    random_state: int = None,
    return_offsets: bool = False,
) -> Union[cudf.DataFrame, Tuple[cudf.DataFrame, cudf.DataFrame]]:
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph, which contains connectivity information as dask cudf
        edge list dataframe

    start_list : list or cudf.Series (int32)
        a list of starting vertices for sampling

    fanout_vals : list (int32)
        List of branching out (fan-out) degrees per starting vertex for each
        hop level.

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    batch_id_list: list (int32)
        List of batch ids that will be returned with the sampled edges if
        with_edge_properties is set to True.

    random_state: int, optional
        Random seed to use when making sampling calls.

    return_offsets: bool, optional (default=False)
        Whether to return the sampling results with batch ids
        included as one dataframe, or to instead return two
        dataframes, one with sampling results and one with
        batch ids and their start offsets.

    Returns
    -------
    result : cudf.DataFrame or Tuple[cudf.DataFrame, cudf.DataFrame]
        GPU data frame containing multiple cudf.Series
        If return_offsets=False:
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
            df['batch_id']: cudf.Series
                Contains the batch ids from the sampling result
            df['hop_id']: cudf.Series
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
            start_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
        )

    if batch_id_list is None:
        batch_id_list = cp.zeros(len(start_list), dtype="int32")

    # fanout_vals must be a host array!
    # FIXME: ensure other sequence types (eg. cudf Series) can be handled.
    if isinstance(fanout_vals, list):
        fanout_vals = numpy.asarray(fanout_vals, dtype="int32")
    else:
        raise TypeError("fanout_vals must be a list, " f"got: {type(fanout_vals)}")

    if "weights" in G.edgelist.edgelist_df:
        weight_t = G.edgelist.edgelist_df["weights"].dtype
    else:
        weight_t = "float32"

    start_list = ensure_valid_dtype(G, start_list)

    if G.renumbered is True:
        if isinstance(start_list, cudf.DataFrame):
            start_list = G.lookup_internal_vertex_id(start_list, start_list.columns)
        else:
            start_list = G.lookup_internal_vertex_id(start_list)

    sampling_result = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_list=start_list,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        batch_id_list=batch_id_list,
        random_state=random_state,
    )

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
    ) = sampling_result

    df["sources"] = sources
    df["destinations"] = destinations
    df["weight"] = weights
    df["edge_id"] = edge_ids
    df["edge_type"] = edge_types
    df["hop_id"] = hop_ids

    if return_offsets:
        offsets_df = cudf.DataFrame(
            {
                "batch_id": batch_ids,
                "offsets": offsets[:-1],
            }
        )

    else:
        if len(batch_ids) > 0:
            batch_ids = cudf.Series(batch_ids).repeat(cp.diff(offsets))
            batch_ids.reset_index(drop=True, inplace=True)

        df["batch_id"] = batch_ids

    if G.renumbered:
        df = G.unrenumber(df, "sources", preserve_order=True)
        df = G.unrenumber(df, "destinations", preserve_order=True)

    if return_offsets:
        return df, offsets_df

    return df
