# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from itertools import chain

from pylibcugraph import ResourceHandle
from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

from cugraph.sampling.sampling_utilities import sampling_results_from_cupy_array_dict

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
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    if isinstance(start_list, cudf.Series):
        start_list_dtypes = start_list.dtype
    else:
        start_list_dtypes = start_list.dtypes.iloc[0]

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
    include_hop_column: bool = True,  # deprecated
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    renumber: bool = False,
    retain_seeds: bool = False,
    label_offsets: Sequence = None,
    use_legacy_names: bool = True,  # deprecated
    compress_per_hop: bool = False,
    compression: str = "COO",
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

    fanout_vals : list (int32) or dict
        List of branching out (fan-out) degrees per starting vertex for each
        hop level or dictionary of edge type and fanout values for
        heterogeneous fanout type.

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

    retain_seeds: bool, optional (default=False)
        If True, will retain the original seeds (original source vertices)
        in the output even if they do not have outgoing neighbors.

    label_offsets: integer sequence, optional (default=None)
        Offsets of each label within the start vertex list.
        Only used if retain_seeds is True.  Required if retain_seeds
        is True.

    use_legacy_names: bool, optional (default=True)
        Whether to use the legacy column names (sources, destinations).
        If True, will use "sources" and "destinations" as the column names.
        If False, will use "majors" and "minors" as the column names.
        Deprecated.  Will be removed in release 23.12 in favor of always
        using the new names "majors" and "minors".

    compress_per_hop: bool, optional (default=False)
        Whether to compress globally (default), or to produce a separate
        compressed edgelist per hop.

    compression: str, optional (default=COO)
        Sets the compression type for the output minibatches.
        Valid options are COO (default), CSR, CSC, DCSR, and DCSC.

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

    if compression not in ["COO", "CSR", "CSC", "DCSR", "DCSC"]:
        raise ValueError("compression must be one of COO, CSR, CSC, DCSR, or DCSC")

    if (
        (compression != "COO")
        and (not compress_per_hop)
        and prior_sources_behavior != "exclude"
    ):
        raise ValueError(
            "hop-agnostic compression is only supported with"
            " the exclude prior sources behavior due to limitations "
            "of the libcugraph C++ API"
        )

    if compress_per_hop and prior_sources_behavior != "carryover":
        raise ValueError(
            "Compressing the edgelist per hop is only supported "
            "with the carryover prior sources behavior due to limitations"
            " of the libcugraph C++ API"
        )

    if include_hop_column:
        warning_msg = (
            "The include_hop_column flag is deprecated and will be"
            " removed in the next release in favor of always "
            "excluding the hop column when return_offsets is True"
        )
        warnings.warn(warning_msg, FutureWarning)

        if compression != "COO":
            raise ValueError(
                "Including the hop id column is only supported with COO compression."
            )

    if with_edge_properties:
        warning_msg = (
            "The with_edge_properties flag is deprecated"
            " and will be removed in the next release in favor"
            " of returning all properties in the graph"
        )
        warnings.warn(warning_msg, FutureWarning)

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
    elif isinstance(fanout_vals, dict):
        # FIXME: Add expensive check to ensure all dict values are lists
        # Convert to a tuple of sequence (edge type size and fanout values)
        edge_type_size = []
        [edge_type_size.append(len(s)) for s in list(fanout_vals.values())]
        edge_type_fanout_vals = list(chain.from_iterable(list(fanout_vals.values())))
        fanout_vals = (
            numpy.asarray(edge_type_size, dtype="int32"),
            numpy.asarray(edge_type_fanout_vals, dtype="int32"))
    else:
        raise TypeError("fanout_vals must be a sequence or a dictionary, " f"got: {type(fanout_vals)}")

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

    sampling_result_array_dict = pylibcugraph_uniform_neighbor_sample(
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
        retain_seeds=retain_seeds,
        label_offsets=label_offsets,
        compression=compression,
        compress_per_hop=compress_per_hop,
        return_dict=True,
    )

    dfs = sampling_results_from_cupy_array_dict(
        sampling_result_array_dict,
        weight_t,
        len(fanout_vals),
        with_edge_properties=with_edge_properties,
        return_offsets=return_offsets,
        renumber=renumber,
        use_legacy_names=use_legacy_names,
        include_hop_column=include_hop_column,
    )

    if G.renumbered and not renumber:
        dfs[0] = G.unrenumber(dfs[0], major_col_name, preserve_order=True)
        dfs[0] = G.unrenumber(dfs[0], minor_col_name, preserve_order=True)

    if len(dfs) > 1:
        return dfs

    return dfs[0]
