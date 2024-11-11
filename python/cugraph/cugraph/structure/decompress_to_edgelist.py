# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
from pylibcugraph import decompress_to_edgelist as pylibcugraph_decompress_to_edgelist

from cugraph.structure import Graph


def decompress_to_edgelist(
    G: Graph
) -> cudf.DataFrame:
    """
    Extract a the edgelist from a graph.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list.

    Returns
    -------
    edge_lists : cudf.DataFrame
        Distributed GPU data frame containing all sources identifiers,
        destination identifiers and if applicable edge weights, edge ids and
        edge types

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> verts = np.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> edgelist = cugraph.decompress_to_edgelist(G, False)

    """


    do_expensive_check = False
    source, destination, weight, edge_ids, edge_type_ids = pylibcugraph_decompress_to_edgelist(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        do_expensive_check=do_expensive_check
    )

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    if weight is not None:
        df["weight"] = weight
    if edge_ids is not None:
        df["edge_ids"] = edge_ids
    if edge_type_ids is not None:
        df["edge_type_ids"] = edge_type_ids

    if G.renumbered:
        df, _ = G.unrenumber(df, "src", get_column_names=True)
        df, _ = G.unrenumber(df, "dst", get_column_names=True)

    return df
