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

from cugraph.community import leiden_wrapper

def leiden(input_graph, gamma=1.0, metric=0, max_level=100):
    """
    Compute a clustering of the graph using the Leiden method. The 
    Leiden method is an improvement on the Louvain method which adds
    a refine partitions step which eliminates disconnected clusters.
    See https://www.nature.com/articles/s41598-019-41695-z for further
    information.
    
    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        and weights. The adjacency list will be computed if not already 
        present.
        
    gamma : floating point
        The value to use for gamma in the metric computations.
        
    metric : integer
        Indicates which metric to optimize with:
        0: Use modularity
        1: Use Constant Potts Model
        
    max_level : integer
        The maximum level to compute too. When specified the computation will 
        terminate rather than coarsening to the max_level level.
        
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
    >>> parts = cugraph.leiden(G)
    """
    
    parts = leiden_wrapper.leiden(input_graph, gamma, metric, max_level)
    return parts