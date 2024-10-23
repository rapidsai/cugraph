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

from pylibcugraph import ResourceHandle
from pylibcugraph import (
    heterogeneous_uniform_neighbor_sample as \
        pylibcugraph_heterogeneous_uniform_neighbor_sample,
)

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
def ensure_valid_dtype(input_graph, start_vertex_list):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    if isinstance(start_vertex_list, cudf.Series):
        start_vertex_list_dtypes = start_vertex_list.dtype
    else:
        start_vertex_list_dtypes = start_vertex_list.dtypes.iloc[0]

    if start_vertex_list_dtypes != vertex_dtype:
        warning_msg = (
            "Uniform neighbor sample requires 'start_vertex_list' to match the graph's "
            f"'vertex' type. input graph's vertex type is: {vertex_dtype} and got "
            f"'start_vertex_list' of type: {start_vertex_list_dtypes}."
        )
        warnings.warn(warning_msg, UserWarning)
        start_vertex_list = start_vertex_list.astype(vertex_dtype)

    return start_vertex_list


def heterogeneous_uniform_neighbor_sample(
    G: Graph,
    start_vertex_list: Sequence,
    start_vertex_offsets: Sequence,
    fanout_vals: List[int],
    num_edge_types: int,
    with_replacement: bool = True,
    with_edge_properties: bool = False,  # deprecated
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    return_hops: bool = True,
    renumber: bool = False,
    retain_seeds: bool = False,
    compression: str = "COO",
    compress_per_hop: bool = False,
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

    start_vertex_list : list or cudf.Series (int32)
        a list of starting vertices for sampling

    start_vertex_offsets: list[int] (Optional)
        Offsets of each label within the start vertex list.

    fanout_vals : list (int32)
        List of branching out (fan-out) degrees per starting vertex for each
        hop level.

    num_edge_types: int32
        Number of edge types where a value of 1 translates to homogeneous neighbor
        sample whereas a value greater than 1 translates to heterogeneous neighbor
        sample.

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    with_edge_properties: bool, optional (default=False)
        Deprecated.
        Flag to specify whether to return edge properties (weight, edge id,
        edge type, batch id, hop id) with the sampled edges.

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

    return_hops: bool, optional (default=True)
        Whether to return the sampling results with hop ids
        corresponding to the hop where the edge appeared.
        Defaults to True.

    renumber: bool, optional (default=False)
        Whether to renumber on a per-batch basis.  If True,
        will return the renumber map and renumber map offsets
        as an additional dataframe.

    retain_seeds: bool, optional (default=False)
        If True, will retain the original seeds (original source vertices)
        in the output even if they do not have outgoing neighbors.

    compression: str, optional (default=COO)
        Sets the compression type for the output minibatches.
        Valid options are COO (default), CSR, CSC, DCSR, and DCSC.

    compress_per_hop: bool, optional (default=False)
        Whether to compress globally (default), or to produce a separate
        compressed edgelist per hop.

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

    use_legacy_names = False  # Deprecated parameter
    include_hop_column = not return_offsets  # Deprecated parameter

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

    if with_edge_properties:
        warning_msg = (
            "The with_edge_properties flag is deprecated"
            " and will be removed in the next release in favor"
            " of returning all properties in the graph"
        )
        warnings.warn(warning_msg, FutureWarning)

    if isinstance(start_vertex_list, int):
        start_vertex_list = [start_vertex_list]

    if isinstance(start_vertex_list, list):
        start_vertex_list = cudf.Series(
            start_vertex_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
        )

    """
    # No batch_ids, the rank owning the vertices will wom the final
    # result.
    if with_edge_properties and not with_batch_ids:
        if isinstance(start_vertex_list, cudf.Series):
            start_vertex_list = start_vertex_list.reset_index(drop=True).to_frame()

        start_vertex_list[batch_col_name] = cudf.Series(
            cp.zeros(len(start_vertex_list), dtype="int32")
        )
    """

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

    start_vertex_list = ensure_valid_dtype(G, start_vertex_list)

    if G.renumbered:
        start_vertex_list = G.lookup_internal_vertex_id(start_vertex_list)

    sampling_result_array_dict = pylibcugraph_heterogeneous_uniform_neighbor_sample(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertex_list=start_vertex_list,
        start_vertex_offsets=start_vertex_offsets,
        h_fan_out=fanout_vals,
        num_edge_types=num_edge_types,
        with_replacement=with_replacement,
        do_expensive_check=False,
        with_edge_properties=with_edge_properties,
        prior_sources_behavior=prior_sources_behavior,
        deduplicate_sources=deduplicate_sources,
        return_hops=return_hops,
        renumber=renumber,
        retain_seeds=retain_seeds,
        compression=compression,
        compress_per_hop=compress_per_hop,
        random_state=random_state,
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
        include_hop_column=include_hop_column,  # Deprecated flag
    )

    if G.renumbered and not renumber:
        dfs[0] = G.unrenumber(dfs[0], major_col_name, preserve_order=True)
        dfs[0] = G.unrenumber(dfs[0], minor_col_name, preserve_order=True)

    if len(dfs) > 1:
        return dfs

    return dfs[0]
