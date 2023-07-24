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
from pylibcugraph.utilities.api_tools import deprecated_warning_wrapper

import numpy

import cudf
import cupy as cp
import warnings

from typing import Union, Tuple, Sequence, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cugraph import Graph


start_col_name = "_START_"
batch_col_name = "_BATCH_"


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


def _uniform_neighbor_sample_legacy(
    G: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    with_edge_properties: bool = False,
    batch_id_list: Sequence = None,
    random_state: int = None,
    return_offsets: bool = False,
    return_hops: bool = True,
) -> Union[cudf.DataFrame, Tuple[cudf.DataFrame, cudf.DataFrame]]:

    warnings.warn(
        "The batch_id_list parameter is deprecated. "
        "Consider passing a DataFrame where the last column "
        "is the batch ids and setting with_batch_ids=True"
    )

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
        )

    if with_edge_properties and batch_id_list is None:
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
        with_edge_properties=with_edge_properties,
        batch_id_list=batch_id_list,
        return_hops=return_hops,
        random_state=random_state,
    )

    df = cudf.DataFrame()

    if with_edge_properties:
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

    else:
        sources, destinations, indices = sampling_result

        df["sources"] = sources
        df["destinations"] = destinations

        if indices is None:
            df["indices"] = None
        else:
            df["indices"] = indices
            if weight_t == "int32":
                df["indices"] = indices.astype("int32")
            elif weight_t == "int64":
                df["indices"] = indices.astype("int64")
            else:
                df["indices"] = indices

    if G.renumbered:
        df = G.unrenumber(df, "sources", preserve_order=True)
        df = G.unrenumber(df, "destinations", preserve_order=True)

    if return_offsets:
        return df, offsets_df

    return df


uniform_neighbor_sample_legacy = deprecated_warning_wrapper(
    _uniform_neighbor_sample_legacy
)


def uniform_neighbor_sample(
    G: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    with_replacement: bool = True,
    with_edge_properties: bool = False,
    batch_id_list: Sequence = None,  # deprecated
    with_batch_ids: bool = False,
    random_state: int = None,
    return_offsets: bool = False,
    return_hops: bool = True,
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    renumber: bool = False,
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

    with_edge_properties: bool, optional (default=False)
        Flag to specify whether to return edge properties (weight, edge id,
        edge type, batch id, hop id) with the sampled edges.

    batch_id_list: list (int32)
        Deprecated.
        List of batch ids that will be returned with the sampled edges if
        with_edge_properties is set to True.

    with_batch_ids: bool, optional (default=False)
        Flag to specify whether batch ids are present in the start_list
        Assumes they are the last column in the start_list dataframe

    random_state: int, optional
        Random seed to use when making sampling calls.

    return_offsets: bool, optional (default=False)
        Whether to return the sampling results with batch ids
        included as one dataframe, or to instead return two
        dataframes, one with sampling results and one with
        batch ids and their start offsets.

    return_hops: bool, optional (default=True)
        Whether to return the sampling results with hop ids
        corresponding to the hop where the edge appeared.
        Defaults to True.

    prior_sources_behavior: str, optional (default=None)
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

    Returns
    -------
    result : cudf.DataFrame or Tuple[cudf.DataFrame, cudf.DataFrame]
        GPU data frame containing multiple cudf.Series

        If with_edge_properties=False:
            df['sources']: cudf.Series
                Contains the source vertices from the sampling result
            df['destinations']: cudf.Series
                Contains the destination vertices from the sampling result
            df['indices']: cudf.Series
                Contains the indices (edge weights) from the sampling result
                for path reconstruction

        If with_edge_properties=True:
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
                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['map']: cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['offsets']: cudf.Series
                        Contains the batch offsets for the renumber maps

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

                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['map']: cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['offsets']: cudf.Series
                        Contains the batch offsets for the renumber maps
    """

    if batch_id_list is not None:
        if prior_sources_behavior or deduplicate_sources:
            raise ValueError(
                "prior_sources_behavior and deduplicate_sources"
                " are not supported with batch_id_list."
                " Consider using with_batch_ids instead."
            )
        if renumber:
            raise ValueError(
                "renumber is not supported with batch_id_list."
                " Consider using with_batch_ids instead."
            )
        return uniform_neighbor_sample_legacy(
            G,
            start_list,
            fanout_vals,
            with_replacement=with_replacement,
            with_edge_properties=with_edge_properties,
            batch_id_list=batch_id_list,
            random_state=random_state,
            return_offsets=return_offsets,
            return_hops=return_hops,
        )

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
        )

    if with_edge_properties and not with_batch_ids:
        if isinstance(start_list, cudf.Series):
            start_list = start_list.reset_index(drop=True).to_frame()

        start_list[batch_col_name] = cudf.Series(
            cp.zeros(len(start_list), dtype="int32")
        )

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

    if isinstance(start_list, cudf.Series):
        start_list = start_list.rename(start_col_name)
        start_list = start_list.to_frame()

        if G.renumbered:
            start_list = G.lookup_internal_vertex_id(start_list, start_col_name)
    else:
        columns = start_list.columns

        if with_batch_ids:
            if G.renumbered:
                start_list = G.lookup_internal_vertex_id(start_list, columns[:-1])
            start_list = start_list.rename(
                columns={columns[0]: start_col_name, columns[-1]: batch_col_name}
            )
        else:
            if G.renumbered:
                start_list = G.lookup_internal_vertex_id(start_list, columns)
            start_list = start_list.rename(columns={columns[0]: start_col_name})

    sampling_result = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_list=start_list[start_col_name],
        batch_id_list=start_list[batch_col_name]
        if batch_col_name in start_list
        else None,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        with_edge_properties=with_edge_properties,
        random_state=random_state,
        prior_sources_behavior=prior_sources_behavior,
        deduplicate_sources=deduplicate_sources,
        return_hops=return_hops,
        renumber=renumber,
    )

    df = cudf.DataFrame()

    if with_edge_properties:
        # TODO use a dictionary at PLC w/o breaking users
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
            ) = sampling_result
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
            ) = sampling_result

        df["sources"] = sources
        df["destinations"] = destinations
        df["weight"] = weights
        df["edge_id"] = edge_ids
        df["edge_type"] = edge_types
        df["hop_id"] = hop_ids

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

        if return_offsets:
            offsets_df = cudf.DataFrame(
                {
                    "batch_id": batch_ids,
                    "offsets": offsets[:-1],
                }
            )

            if renumber:
                offsets_df["renumber_map_offsets"] = renumber_map_offsets[:-1]

        else:
            if len(batch_ids) > 0:
                batch_ids = cudf.Series(batch_ids).repeat(cp.diff(offsets))
                batch_ids.reset_index(drop=True, inplace=True)

            df["batch_id"] = batch_ids

    else:
        sources, destinations, indices = sampling_result

        df["sources"] = sources
        df["destinations"] = destinations

        if indices is None:
            df["indices"] = None
        else:
            df["indices"] = indices
            if weight_t == "int32":
                df["indices"] = indices.astype("int32")
            elif weight_t == "int64":
                df["indices"] = indices.astype("int64")
            else:
                df["indices"] = indices

    if G.renumbered and not renumber:
        df = G.unrenumber(df, "sources", preserve_order=True)
        df = G.unrenumber(df, "destinations", preserve_order=True)

    if return_offsets:
        if renumber:
            return df, offsets_df, renumber_df
        else:
            return df, offsets_df

    if renumber:
        return df, renumber_df

    return df
