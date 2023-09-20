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


def uniform_neighbor_sample(
    G: Graph,
    start_list: Sequence,
    fanout_vals: List[int],
    *,
    with_replacement: bool = True,
    with_edge_properties: bool = False,  # deprecated
    with_batch_ids: bool = False,
    random_state: int = None,
    return_offsets: bool = False,
    return_hops: bool = True,
    include_hop_column: bool = True, # deprecated
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    renumber: bool = False,
    use_legacy_names=True, # deprecated
    compress_per_hop=False,
    compression='COO',
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
        Deprecated.
        Flag to specify whether to return edge properties (weight, edge id,
        edge type, batch id, hop id) with the sampled edges.

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
    
    include_hop_column: bool, optional (default=True)
        Deprecated.  Defaults to True.
        If True, will include the hop column even if
        return_offsets is True.  This option will
        be removed in release 23.12.

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
    
    use_legacy_names: bool, optional (default=True)
        Whether to use the legacy column names (sources, destinations).
        If True, will use "sources" and "destinations" as the column names.
        If False, will use "majors" and "minors" as the column names.
    
    compress_per_hop: bool, optional (default=False)
        Whether to compress globally (default), or to produce a separate
        compressed edgelist per hop.

    compression: str, optional (default=COO)
        Sets the compression type for the output minibatches.
        Valid options are COO (default), CSR, CSR, DCSR, and DCSR.

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

    if use_legacy_names:
        major_col_name = "sources"
        minor_col_name = "destinations"
        warning_msg = (
            "The legacy column names (sources, destinations)"
            " will no longer be supported for uniform_neighbor_sample"
            " in release 23.12.  The use_legacy_names=False option will"
            " become the only option, and (majors, minors) will be the"
            " only supported column names."
        )
        warnings.warn(warning_msg, FutureWarning)
    else:
        major_col_name = "majors"
        minor_col_name = "minors"

    if with_edge_properties:
        warning_msg = (
            "The with_edge_properties flag is deprecated"
            " and will be removed in the next release."
        )
        warnings.warn(warning_msg, DeprecationWarning)

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

    # fanout_vals must be passed to pylibcugraph as a host array
    if isinstance(fanout_vals, numpy.ndarray):
        fanout_vals = fanout_vals.astype("int32")
    elif isinstance(fanout_vals, list):
        fanout_vals = numpy.asarray(fanout_vals, dtype="int32")
    elif isinstance(fanout_vals, cp.ndarray):
        fanout_vals = fanout_vals.get().astype("int32")
    elif isinstance(fanout_vals, cudf.Series):
        fanout_vals = fanout_vals.values_host.astype("int32")
    else:
        raise TypeError("fanout_vals must be a sequence, " f"got: {type(fanout_vals)}")

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
        compression=compression,
        compress_per_hop=compress_per_hop,
        return_dict=True,
    )

    results_df = cudf.DataFrame()

    if with_edge_properties:
        results_df_cols = [
            'majors',
            'minors',
            'weight',
            'edge_id',
            'edge_type',
            'hop_id'
        ]
        for col in results_df_cols:
            array = sampling_result[col]
            if array is not None:
                # The length of each of these arrays should be the same
                results_df[col] = array

        results_df.rename(columns={'majors':major_col_name, 'minors':minor_col_name},inplace=True)

        label_hop_offsets = sampling_result['label_hop_offsets']
        batch_ids = sampling_result['batch_id']

        if renumber:
            renumber_df = cudf.DataFrame({
                'map': sampling_result['renumber_map'],   
            })

            if not return_offsets:
                if len(batch_ids) > 0:
                    batch_ids_r = cudf.Series(batch_ids).repeat(
                        cp.diff(sampling_result['renumber_map_offsets'])
                    )
                    batch_ids_r.reset_index(drop=True, inplace=True)
                    renumber_df["batch_id"] = batch_ids_r
                else:
                    renumber_df['batch_id'] = None

        if return_offsets:
            batches_series = cudf.Series(
                batch_ids,
                name="batch_id",
            )
            if include_hop_column:
                # TODO remove this logic in release 23.12
                offsets_df = cudf.Series(
                    label_hop_offsets[cp.arange(len(batch_ids)+1) * len(fanout_vals)],
                    name='offsets',
                ).to_frame()
            else:
                offsets_df = cudf.Series(
                    label_hop_offsets,
                    name="offsets",
                ).to_frame()

            if len(batches_series) > len(offsets_df):
                # this is extremely rare so the inefficiency is ok
                offsets_df = offsets_df.join(batches_series, how='outer').sort_index()
            else:
                offsets_df['batch_id'] = batches_series

            if renumber:
                renumber_offset_series = cudf.Series(
                    sampling_result['renumber_map_offsets'],
                    name="renumber_map_offsets"
                )

                if len(renumber_offset_series) > len(renumber_df):
                    # this is extremely rare so the inefficiency is ok
                    renumber_df = renumber_df.join(renumber_offset_series, how='outer').sort_index()
                else:
                    renumber_df['renumber_map_offsets'] = renumber_offset_series

        else:
            if len(batch_ids) > 0:
                batch_ids_r = cudf.Series(cp.repeat(batch_ids, len(fanout_vals)))
                batch_ids_r = cudf.Series(batch_ids_r).repeat(cp.diff(label_hop_offsets))                    
                batch_ids_r.reset_index(drop=True, inplace=True)

                results_df["batch_id"] = batch_ids_r
            else:
                results_df['batch_id'] = None
        
        # TODO remove this logic in release 23.12, hops will always returned as offsets
        if include_hop_column:
            if len(batch_ids) > 0:
                hop_ids_r = cudf.Series(cp.arange(len(fanout_vals)))
                hop_ids_r = cudf.concat([hop_ids_r] * len(batch_ids),ignore_index=True)

                # generate the hop column
                hop_ids_r = cudf.Series(hop_ids_r, name='hop_id').repeat(
                    cp.diff(label_hop_offsets)
                ).reset_index(drop=True)
            else:
                hop_ids_r = cudf.Series(name='hop_id', dtype='int32')

            results_df = results_df.join(hop_ids_r, how='outer').sort_index()

        if major_col_name not in results_df:
            if use_legacy_names:
                raise ValueError("Can't use legacy names with major offsets")

            major_offsets_series = cudf.Series(sampling_result['major_offsets'], name='major_offsets')
            if len(major_offsets_series) > len(results_df):
                # this is extremely rare so the inefficiency is ok
                results_df = results_df.join(major_offsets_series, how='outer').sort_index()
            else:
                results_df['major_offsets'] = major_offsets_series

    else:
        # TODO this is deprecated, remove it in 23.12

        results_df[major_col_name] = sampling_result['sources']
        results_df[minor_col_name] = sampling_result['destinations']
        indices = sampling_result['indices']

        if indices is None:
            results_df["indices"] = None
        else:
            results_df["indices"] = indices
            if weight_t == "int32":
                results_df["indices"] = indices.astype("int32")
            elif weight_t == "int64":
                results_df["indices"] = indices.astype("int64")
            else:
                results_df["indices"] = indices

    if G.renumbered and not renumber:
        results_df = G.unrenumber(results_df, major_col_name, preserve_order=True)
        results_df = G.unrenumber(results_df, minor_col_name, preserve_order=True)

    if return_offsets:
        if renumber:
            return results_df, offsets_df, renumber_df
        else:
            return results_df, offsets_df

    if renumber:
        return results_df, renumber_df

    return results_df
