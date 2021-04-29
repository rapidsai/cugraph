# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cugraph.generators import rmat_wrapper
import cugraph


def _ensure_args_edgelist(scale, a, b, c, clip_and_flip, scramble_vertex_ids):
    """
    Ensures the args passed in are usable for the API api_name and raises TypeError
    or ValueError if incorrectly specified.
    """
    if not isinstance(scale, int):
        raise TypeError("'scale' must be an int")
    if (a+b+c > 1):
        raise ValueError("a + b + c should be non-negative and no larger than 1.0")
    if (clip_and_flip not in [True, False]):
        raise ValueError("'clip_and_flip' must be a bool")
    if (scramble_vertex_ids not in [True, False]):
        raise ValueError("'clip_and_flip' must be a bool")

#def _ensure_args_edgelists()

def rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids
):
    """
    Generate a Graph object using a Recursive MATrix (R-MAT) graph generation algorithm.

    Parameters
    ----------
    scale : int
            Scale factor to set the number of verties in the graph
            Vertex IDs have values in [0, V), where V = 1 << 'scale'

    num_edges : int
            Number of edges to generate

    a : float
            Probability of the first partition

    b : float
            Probability of the second partition

    c : float
            Probability of the thrid partition

    seed : int
            Seed value for the random number generator

    clip_and_flip : bool
            Flag controlling whether to generate edges only in the lower triangular part
            (including the diagonal) of the graph adjacency matrix (if set to 'true')
            or not (if set to 'false).

    scramble_vertex_ids : bool
            Flag controlling whether to scramble vertex ID bits (if set to `true`)
            or not (if set to `false`); scrambling vertx ID bits breaks correlation between
            vertex ID values and vertex degrees


    Returns
    -------
    instance of cugraph.Graph
    """
    _ensure_args_edgelist(scale, a, b, c, clip_and_flip, scramble_vertex_ids)

    df = rmat_wrapper.graph_generator_edgelist(scale, num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids)

    #Convertion to Graph
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst')

    return G


def multi_rmat(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids
):
    """
    Generate multiple Graph objects using a Recursive MATrix (R-MAT) graph generation algorithm.

    Parameters
    ----------
    n_edgelists : int
        Number of edge lists (graphs) to generate

    min_scale : int
        Scale factor to set the minimum number of vertices in the graph

    max_scale : int
        Scale factor to set the maximum number of vertices in the graph

    edge_factor : int
        Average number of edges per vertex to generate

    size_distribution :
        Distribution of the graph sizes, impacts the scale parameter of the
        R-MAT generator

    edge_distribution :
        Edges distribution for each graph, impacts how R-MAT parameters a,b,c,d,
        are set

    seed : int
        Seed value for the random number generator

    clip_and_flip : bool
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix (if set to 'true')
        or not (if set to 'false')

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to `true`)
        or not (if set to `false`); scrambling vertx ID bits breaks correlation between vertex
        ID values and vertex degrees

    Returns
    -------
    list of cugraph.Graph instances
    """
    #_ensure_args_edgelists(scale, a, b, c, clip_and_flip, scramble_vertex_ids)

    dfs = rmat_wrapper.graph_generator_edgelists(n_edgelists, min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids)
    list_G = []

    for df in dfs:
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst')
        list_G.append(G)


    return list_G
