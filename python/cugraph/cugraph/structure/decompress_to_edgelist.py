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
    G: Graph,
    do_expensive_check: bool
) -> cudf.DataFrame:
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    If no subgraph can be extracted from the vertices provided, a 'None' value
    will be returned.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        The current implementation only supports weighted graphs.

    do_expensive_check: bool

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
    >>> sverts = cudf.Series(verts)
    >>> edgelist = cugraph.decompress_to_edgelist(G, False)

    """


    do_expensive_check = False
    source, destination, weight, edge_ids, edge_type_ids = pylibcugraph_decompress_to_edgelist(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        do_expensive_check=do_expensive_check
    )

    print("source = ", source)
    print("detaination = ", destination)

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
