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

from pylibcugraph import ResourceHandle
from pylibcugraph import uniform_neighbor_sample as pylibcugraph_uniform_neighbor_sample

import numpy

import cudf
import cupy as cp


def uniform_neighbor_sample(
    G,
    start_list,
    fanout_vals,
    with_replacement=True,
    with_edge_properties=False,
    batch_id_list=None,
    random_state=None,
):
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Note: This is a pylibcugraph-enabled algorithm, which requires that the
    graph was created with legacy_renum_only=True.

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
        List of batch ids that will be returned with the sampled edges if
        with_edge_properties is set to True.

    random_state: int, optional
        Random seed to use when making sampling calls.

    Returns
    -------
    result : cudf.DataFrame
        GPU data frame containing multiple cudf.Series

        If with_edge_properties=True:
            df['sources']: cudf.Series
                Contains the source vertices from the sampling result
            df['destinations']: cudf.Series
                Contains the destination vertices from the sampling result
            df['indices']: cudf.Series
                Contains the indices (edge weights) from the sampling result
                for path reconstruction

        If with_edge_properties=False:
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
    """

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
            hop_ids,
        ) = sampling_result

        df["sources"] = sources
        df["destinations"] = destinations
        df["weight"] = weights
        df["edge_id"] = edge_ids
        df["edge_type"] = edge_types
        df["batch_id"] = batch_ids
        df["hop_id"] = hop_ids
    else:
        sources, destinations, indices = sampling_result

        df["sources"] = sources
        df["destinations"] = destinations

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

    return df
