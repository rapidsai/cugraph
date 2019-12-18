# Copyright (c) 2019, NVIDIA CORPORATION.
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


def ecg(input_graph, min_weight=.05, ensemble_size=16):
    """
    Compute the ensemble clustering for graphs partition of the input graph

    Parameters
    ----------
    input_graph : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information
        and weights.The adjacency list will be computed if not already
        present. The graph should be undirected where an undirected edge is
        represented by an a directed edge in both directions.

    min_weight : floating point
        The minimum value to assign as an edgeweight in the ECG algorithm

    ensemble_size : integer
        The number of graph permutations to use for the ensemble.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns, the vertex id and
        the partition id it is assigned to.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv',
                          delimiter = ' ',
                          dtype=['int32', 'int32', 'float32'],
                          header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> values = cudf.Series(M['2'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, values)
    >>> parts = cugraph.ecg(G)
    """

    parts = ecg_wrapper.ecg(input_graph, min_weight, ensemble_size)

    return parts
