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

from cugraph.community import ecg_wrapper
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)


def ecg(input_graph, min_weight=0.05, ensemble_size=16, weight=None):
    """
    Compute the Ensemble Clustering for Graphs (ECG) partition of the input
    graph. ECG runs truncated Louvain on an ensemble of permutations of the
    input graph, then uses the ensemble partitions to determine weights for
    the input graph. The final result is found by running full Louvain on
    the input graph using the determined weights.

    See https://arxiv.org/abs/1809.05578 for further information.

    Parameters
    ----------
    input_graph : cugraph.Graph or NetworkX Graph
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

    weight : str, optional (default=None)
        This parameter is here for NetworkX compatibility and
        represents which NetworkX data column represents Edge weights.

    Returns
    -------
    parts : cudf.DataFrame or python dictionary
        GPU data frame of size V containing two columns, the vertex id and
        the partition id it is assigned to.

        df[vertex] : cudf.Series
            Contains the vertex identifiers
        df[partition] : cudf.Series
            Contains the partition assigned to the vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> parts = cugraph.ecg(G)

    """

    input_graph, isNx = ensure_cugraph_obj_for_nx(input_graph, weight)

    parts = ecg_wrapper.ecg(input_graph, min_weight, ensemble_size)

    if input_graph.renumbered:
        parts = input_graph.unrenumber(parts, "vertex")

    if isNx is True:
        return df_score_to_dictionary(parts, "partition")
    else:
        return parts
