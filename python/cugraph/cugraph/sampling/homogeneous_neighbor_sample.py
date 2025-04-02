# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import cudf
from pylibcugraph import ResourceHandle
from pylibcugraph import (
    homogeneous_uniform_neighbor_sample as pylibcugraph_homogeneous_uniform_neighbor_sample,
    homogeneous_biased_neighbor_sample as pylibcugraph_homogeneous_biased_neighbor_sample,
)
from cugraph.sampling.sampling_utilities import sampling_results_from_cupy_array_dict

from cugraph.structure import Graph


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


def homogeneous_neighbor_sample(
    G: Graph,
    start_list: Sequence,
    starting_vertex_label_offsets: Sequence,
    fanout_vals: List[int],
    *,
    with_replacement: bool = True,
    with_biases: bool = False,
    random_state: int = None,
    return_offsets: bool = False,
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    return_hops: bool = True,
    renumber: bool = False,
    retain_seeds: bool = False,
    compress_per_hop: bool = False,
    compression: str = "COO",
) -> Tuple[cudf.Series, cudf.Series, Union[None, int, cudf.Series]]:
    """
    Performs uniform/biased neighborhood sampling, which samples nodes from
    a graph based on the current node's neighbors, with a corresponding fan_out
    value at each hop. The edges are sampled either uniformly or with biases. Homogeneous
    neighborhood sampling translates to 1 edge type.

    parameters
    ----------
    G : cuGraph.Graph
        The graph can be either directed or undirected.

    start_list : list or cudf.Series
        a list of starting vertices for sampling

    starting_vertex_label_offsets: list or cudf.Series
        Offsets of each label within the start_list. Expanding
        'starting_vertex_label_offsets' must lead to an array of
        len(start_list)

    fanout_vals : list
        List of branching out (fan-out) degrees per starting vertex for each
        hop level.

    with_replacement: bool, optional (default=True)
        Flag to specify if the random sampling is done with replacement

    with_biases: bool, optional (default=False)
        Flag to specify whether the edges should be sampled uniformly or with biases.
        Only edge weights can be used as biases for now

    random_state: int, optional
        Random seed to use when making sampling calls.

    return_offsets: bool, optional (default=False)
        Whether to return the sampling results with batch ids
        included as one dataframe, or to instead return two
        dataframes, one with sampling results and one with
        batch ids and their start offsets.

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


        If return_offsets=False:
                df['majors']: cudf.Series
                    Contains the source vertices from the sampling result
                df['minors']: cudf.Series
                    Contains the destination vertices from the sampling result
                df['weight']: cudf.Series # if provided
                    Contains the edge weights from the sampling result
                df['edge_id']: cudf.Series # if provided
                    Contains the edge ids from the sampling result
                df['edge_type']: cudf.Series # if provided
                    Contains the edge types from the sampling result
                df['batch_id']: cudf.Series
                    Contains the batch ids from the sampling result
                df['hop_id']: cudf.Series
                    Contains the hop ids from the sampling result
                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['renumber_map']: cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['batch_id']: cudf.Series
                        Contains the batch ids for the renumber maps

        If return_offsets=True:
                df['majors']: cudf.Series
                    Contains the source vertices from the sampling result
                df['minors']: cudf.Series
                    Contains the destination vertices from the sampling result
                df['weight']: cudf.Series # if provided
                    Contains the edge weights from the sampling result
                df['edge_id']: cudf.Series # if provided
                    Contains the edge ids from the sampling result
                df['edge_type']: cudf.Series # if provided
                    Contains the edge types from the sampling result
                df['batch_id']: cudf.Series
                    Contains the batch ids from the sampling result
                df['hop_id']: cudf.Series
                    Contains the hop ids from the sampling result

                offsets_df['batch_id']: cudf.Series
                    Contains the batch ids from the sampling result
                offsets_df['offsets']: cudf.Series
                    Contains the offsets of each batch in the sampling result

                If renumber=True:
                    (adds the following dataframe)
                    renumber_df['renumber_map']: cudf.Series
                        Contains the renumber maps for each batch
                    renumber_df['batch_id']: cudf.Series
                        Contains the batch ids for the renumber maps

    """

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

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
        )

    if isinstance(starting_vertex_label_offsets, list):
        starting_vertex_label_offsets = cudf.Series(starting_vertex_label_offsets)

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

    if G.renumbered is True:
        if isinstance(start_list, cudf.DataFrame):
            start_list = G.lookup_internal_vertex_id(start_list, start_list.columns)
        else:
            start_list = G.lookup_internal_vertex_id(start_list)

    if with_biases:
        sampling_function = pylibcugraph_homogeneous_biased_neighbor_sample
    else:
        sampling_function = pylibcugraph_homogeneous_uniform_neighbor_sample

    sampling_result_array_dict = sampling_function(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertex_list=start_list,
        starting_vertex_label_offsets=starting_vertex_label_offsets,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
        prior_sources_behavior=prior_sources_behavior,
        deduplicate_sources=deduplicate_sources,
        return_hops=return_hops,
        renumber=renumber,
        retain_seeds=retain_seeds,
        compression=compression,
        compress_per_hop=compress_per_hop,
        random_state=random_state,
    )

    dfs = sampling_results_from_cupy_array_dict(
        sampling_result_array_dict,
        weight_t,
        len(fanout_vals),
        return_offsets=return_offsets,
        renumber=renumber,
    )

    if G.renumbered and not renumber:
        dfs[0] = G.unrenumber(dfs[0], major_col_name, preserve_order=True)
        dfs[0] = G.unrenumber(dfs[0], minor_col_name, preserve_order=True)

    if len(dfs) > 1:
        return dfs

    return dfs[0]
