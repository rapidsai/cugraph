# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.linear_assignment import lap_wrapper


def hungarian(G, workers):
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

    workers : cudf.Series
        A series or column that identifies the vertex ids of the vertices
        in the workers set.  All vertices in G that are not in the workers
        set are implicitly assigned to the jobs set.

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
    >>> M = cudf.read_csv('datasets/bipartite.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')
    >>> cost, df = cugraph.hungarian(G, workers)

    """

    if G.renumbered:
        local_workers = G.lookup_internal_vertex_id(workers)
    else:
        local_workers = workers

    df = lap_wrapper.sparse_hungarian(G, local_workers)

    if G.renumbered:
        df = G.unrenumber(df, 'vertex')

    return df


def dense_hungarian(costs, num_rows, num_columns):
    """
    Execute the Hungarian algorithm against a dense bipartite
    graph representation.

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


    Returns
    -------
    cost : matches costs.dtype
        The cost of the overall assignment
    assignment : cudf.Series
      assignment[i] gives the vertex id of the task assigned to the
                    worker i

    FIXME: Update this with a real example...

    """

    return lap_wrapper.dense_hungarian(costs, num_rows, num_columns)
