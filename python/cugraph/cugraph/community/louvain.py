# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import networkx as nx
from cugraph.utilities import (
    is_nx_graph_type,
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
import cudf

import warnings
from pylibcugraph import louvain as pylibcugraph_louvain
from pylibcugraph import ResourceHandle

VERTEX_COL_NAME = "vertex"
CLUSTER_ID_COL_NAME = "partition"


# FIXME: max_level should default to 100 once max_iter is removed
def louvain(G, max_level=None, max_iter=None, resolution=1.0, threshold=1e-7):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method

    It uses the Louvain method described in:

    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
    community hierarchies in large networks, J Stat Mech P10008 (2008),
    http://arxiv.org/abs/0803.0476

    Parameters
    ----------
    G : cugraph.Graph or NetworkX Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs.

    max_level : integer, optional (default=100)
        This controls the maximum number of levels of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of levels. No error occurs when the
        algorithm terminates early in this manner.

    max_iter : integer, optional (default=None)
        This parameter is deprecated in favor of max_level.  Previously
        it was used to control the maximum number of levels of the Louvain
        algorithm.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    threshold: float
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.
        Defaults to 1e-7.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        result_df[VERTEX_COL_NAME] : cudf.Series
            Contains the vertex identifiers
        result_df[CLUSTER_ID_COL_NAME] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> parts = cugraph.louvain(G)

    """

    # FIXME: Onece the graph construction calls support isolated vertices through
    #  the C API (the C++ interface already supports this) then there will be
    # no need to compute isolated vertices here.

    isolated_vertices = list()
    if is_nx_graph_type(type(G)):
        isolated_vertices = list(nx.isolates(G))
    else:
        # FIXME: Gather isolated vertices of G
        pass

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    # FIXME: This max_iter logic and the max_level defaulting can be deleted
    #        in favor of defaulting max_level in call once max_iter is deleted
    if max_iter:
        if max_level:
            raise ValueError(
                "max_iter is deprecated.  Cannot specify both max_iter and max_level"
            )

        warning_msg = (
            "max_iter has been renamed max_level.  Use of max_iter is "
            "deprecated and will no longer be supported in the next releases."
        )
        warnings.warn(warning_msg, FutureWarning)
        max_level = max_iter

    if max_level is None:
        max_level = 100

    if max_level > 1000:
        max_level = 1000

    vertex, partition, mod_score = pylibcugraph_louvain(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )

    result_df = cudf.DataFrame()
    result_df[VERTEX_COL_NAME] = vertex
    result_df[CLUSTER_ID_COL_NAME] = partition

    if isNx and len(isolated_vertices) > 0:
        unique_cids = result_df[CLUSTER_ID_COL_NAME].unique()
        max_cluster_id = -1 if len(result_df) == 0 else unique_cids.max()

        isolated_vtx_and_cids = cudf.DataFrame()
        isolated_vtx_and_cids[VERTEX_COL_NAME] = isolated_vertices
        isolated_vtx_and_cids[CLUSTER_ID_COL_NAME] = [
            (max_cluster_id + i + 1) for i in range(len(isolated_vertices))
        ]
        result_df = cudf.concat(
            [result_df, isolated_vtx_and_cids], ignore_index=True, sort=False
        )

    if G.renumbered:
        result_df = G.unrenumber(result_df, VERTEX_COL_NAME)

    if isNx is True:
        result_df = df_score_to_dictionary(result_df, CLUSTER_ID_COL_NAME)

    return result_df, mod_score
