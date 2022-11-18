# Copyright (c) 2022, NVIDIA CORPORATION.
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


def uniform_neighbor_sample(
    G, start_list, fanout_vals, with_replacement=True, is_edge_ids=False
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

    Returns
    -------
    result : cudf.DataFrame
        GPU data frame containing two cudf.Series

        df['sources']: cudf.Series
            Contains the source vertices from the sampling result
        df['destinations']: cudf.Series
            Contains the destination vertices from the sampling result
        df['indices']: cudf.Series
            Contains the indices from the sampling result for path
            reconstruction
    """

    if isinstance(start_list, int):
        start_list = [start_list]

    if isinstance(start_list, list):
        start_list = cudf.Series(
            start_list, dtype=G.edgelist.edgelist_df[G.srcCol].dtype
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

    if G.renumbered is True:
        if isinstance(start_list, cudf.DataFrame):
            start_list = G.lookup_internal_vertex_id(start_list, start_list.columns)
        else:
            start_list = G.lookup_internal_vertex_id(start_list)

    sources, destinations, indices = pylibcugraph_uniform_neighbor_sample(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_list=start_list,
        h_fan_out=fanout_vals,
        with_replacement=with_replacement,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
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
