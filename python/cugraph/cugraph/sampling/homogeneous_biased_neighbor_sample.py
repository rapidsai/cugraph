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
    homogeneous_biased_neighbor_sample as pylibcugraph_homogeneous_biased_neighbor_sample,
)

from cugraph.structure import Graph

from typing import Union, Tuple


def homogeneous_biased_neighbor_sample(
    G: Graph,
    start_list: Sequence,
    starting_vertex_label_offsets: Sequence,
    fanout_vals: List[int],
    *,
    with_replacement: bool = True,
    random_state: int = None,
    prior_sources_behavior: str = None,
    deduplicate_sources: bool = False,
    return_hops: bool = True,
    renumber: bool = False,
    retain_seeds: bool = False,
    compress_per_hop: bool = False,
    compression: str = "COO",
) -> Tuple[cudf.Series, cudf.Series, Union[None, int, cudf.Series]]:
    """
    Performs biased neighborhood sampling, which samples nodes from
    a graph based on the current node's neighbors, with a corresponding fan_out
    value at each hop. The edges are sampled biasedly. Homogeneous
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

    random_state: int, optional
        Random seed to use when making sampling calls.
    
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
        # FIXME: Update the return type
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


    x = pylibcugraph_homogeneous_biased_neighbor_sample(
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
        compress_per_hop
        random_state=random_state,
    )
