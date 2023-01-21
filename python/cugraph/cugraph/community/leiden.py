# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.community import leiden_wrapper
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)


def leiden(G, max_iter=100, resolution=1.0):
    """
    Compute the modularity optimizing partition of the input graph using the
    Leiden algorithm

    It uses the Louvain method described in:

    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
    doi: 10.1038/s41598-019-41695-z

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor of type Graph

        The current implementation only supports undirected weighted graphs.

        The adjacency list will be computed if not already present.

    max_iter : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the Leiden
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    resolution: float/double, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

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
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> parts, modularity_score = cugraph.leiden(G)

    """
    G, isNx = ensure_cugraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    parts, modularity_score = leiden_wrapper.leiden(G, max_iter, resolution)

    if G.renumbered:
        parts = G.unrenumber(parts, "vertex")

    if isNx is True:
        parts = df_score_to_dictionary(parts, "partition")

    return parts, modularity_score
