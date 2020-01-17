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

from cugraph.community import triangle_count_wrapper


def triangles(G):
    """
    Compute the triangle (number of cycles of length three) count of the
    input graph.

    Parameters
    ----------
    G : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm)

    Returns
    -------
    count : int64
        A 64 bit integer whose value gives the number of triangles in the
        graph.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> count = cugraph.triangles(G)
    """

    result = triangle_count_wrapper.triangles(G)

    return result
