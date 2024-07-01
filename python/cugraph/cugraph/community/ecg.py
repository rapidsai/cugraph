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

from pylibcugraph import ecg as pylibcugraph_ecg
from pylibcugraph import ResourceHandle

from cugraph.structure import Graph
import cudf
from typing import Union
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
from cugraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in the type annotation for
# leiden() is specified using a string literal to avoid depending on
# and importing networkx. Instead, networkx is imported optionally, which may
# cause a problem for a type checker if run in an environment where networkx is
# not installed.
networkx = import_optional("networkx")


def ecg(
    G: Union[Graph, "networkx.Graph"],
    min_weight: float = 0.0001,
    ensemble_size: int = 100,
    max_level: int = 10,
    threshold: float = 1e-7,
    resolution: float = 1.0,
    random_state: int = None,
):
    """
    Compute the Ensemble Clustering for Graphs (ECG) partition of the input
    graph. ECG runs truncated Louvain on an ensemble of permutations of the
    input graph, then uses the ensemble partitions to determine weights for
    the input graph. The final result is found by running full Louvain on
    the input graph using the determined weights.

    See https://arxiv.org/abs/1809.05578 for further information.

    Parameters
    ----------
    G : cugraph.Graph or NetworkX Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.

    min_weight : float, optional (default=0.5)
        The minimum value to assign as an edgeweight in the ECG algorithm.
        It should be a value in the range [0,1] usually left as the default
        value of .05

    ensemble_size : integer, optional (default=16)
        The number of graph permutations to use for the ensemble.
        The default value is 16, larger values may produce higher quality
        partitions for some graphs.

    max_level : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the ECG
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    threshold: float
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.
        Defaults to 1e-7.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    random_state: int, optional(default=None)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.

    Returns
    -------
    parts : cudf.DataFrame or python dictionary
        GPU data frame of size V containing two columns, the vertex id and
        the partition id it is assigned to.

        df[vertex] : cudf.Series
            Contains the vertex identifiers
        df[partition] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        A floating point number containing the global modularity score
        of the partitioning.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> parts, mod = cugraph.ecg(G)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)

    vertex, partition, modularity_score = pylibcugraph_ecg(
        resource_handle=ResourceHandle(),
        random_state=random_state,
        graph=G._plc_graph,
        min_weight=min_weight,
        ensemble_size=ensemble_size,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["partition"] = partition

    if G.renumbered:
        parts = G.unrenumber(df, "vertex")
    else:
        parts = df

    if isNx is True:
        parts = df_score_to_dictionary(df, "partition")

    return parts, modularity_score
