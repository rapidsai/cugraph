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

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
import cudf

import warnings
from pylibcugraph import louvain as pylibcugraph_louvain
from pylibcugraph import ResourceHandle


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

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
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

    vertex, partition, mod_score = pylibcugraph_louvain(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["partition"] = partition

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "partition")

    return df, mod_score
