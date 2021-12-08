# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from cugraph.linear_assignment import lap_wrapper


def hungarian(G, workers, epsilon=None):
    """
    Execute the Hungarian algorithm against a symmetric, weighted,
    bipartite graph.

    As a bipartite graph, the vertex set of the graph can be partitioned
    into two disjoint sets such that all edges connect a vertex from
    one set to a vertex of the other set.  The workers variable identifies
    one of the sets of vertices, the other set is all vertices not in
    the workers set (V - workers).

    The edge weights reflect the cost of assigning a particular job to a
    worker.

    The Hungarian algorithm identifies the lowest cost matching of vertices
    such that all workers that can be assigned work are assigned exactly
    on job.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an an edge list.  Edge weights are required. If an edge list is
        not provided then it will be computed.

    workers : cudf.Series or cudf.DataFrame
        A series or column that identifies the vertex ids of the vertices
        in the workers set.  In case of multi-column vertices, it should be a
        cudf.DataFrame. All vertices in G that are not in the workers
        set are implicitly assigned to the jobs set.

    epsilon : float or double (matching weight type in graph)
        Used for determining when value is close enough to zero to consider 0.
        Defaults (if not specified) to 1e-6 in the C++ code.  Unused for
        integer weight types.

    Returns
    -------
    cost : matches costs.dtype
        The cost of the overall assignment
    df : cudf.DataFrame
      df['vertex'][i] gives the vertex id of the i'th vertex.  Only vertices
                      in the workers list are defined in this column.
      df['assignment'][i] gives the vertex id of the "job" assigned to the
                          corresponding vertex.

    FIXME: Update this with a real example...

    Examples
    --------
    >>  Download dataset from https://github.com/rapidsai/cugraph/datasets/...
    >>> M = cudf.read_csv('datasets/bipartite.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')
    >>> cost, df = cugraph.hungarian(G, workers)

    """

    if G.renumbered:
        if isinstance(workers, cudf.DataFrame):
            local_workers = G.lookup_internal_vertex_id(workers,
                                                        workers.columns)
        else:
            local_workers = G.lookup_internal_vertex_id(workers)
    else:
        local_workers = workers

    cost, df = lap_wrapper.sparse_hungarian(G, local_workers, epsilon)

    if G.renumbered:
        df = G.unrenumber(df, 'vertex')

    return cost, df


def dense_hungarian(costs, num_rows, num_columns, epsilon=None):
    """
    Execute the Hungarian algorithm against a dense bipartite
    graph representation.

    *NOTE*: This API is unstable and subject to change

    The Hungarian algorithm identifies the lowest cost matching of vertices
    such that all workers that can be assigned work are assigned exactly
    on job.

    Parameters
    ----------
    costs : cudf.Series
        A dense representation (row major order) of the bipartite
        graph.  Each row represents a worker, each column represents
        a task, cost[i][j] represents the cost of worker i performing
        task j.
    num_rows : int
        Number of rows in the matrix
    num_columns : int
        Number of columns in the matrix
    epsilon : float or double (matching weight type in graph)
        Used for determining when value is close enough to zero to consider 0.
        Defaults (if not specified) to 1e-6 in the C++ code.  Unused for
        integer weight types.

    Returns
    -------
    cost : matches costs.dtype
        The cost of the overall assignment
    assignment : cudf.Series
      assignment[i] gives the vertex id of the task assigned to the
                    worker i

    FIXME: Update this with a real example...

    """

    return lap_wrapper.dense_hungarian(costs, num_rows, num_columns, epsilon)
