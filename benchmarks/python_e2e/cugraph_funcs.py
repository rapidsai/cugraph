# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np

import cugraph
from cugraph.generators import rmat
import cudf


def generate_edgelist(scale,
                      edgefactor,
                      seed=None,
                      unweighted=False,
                     ):
    """
    Returns a cudf DataFrame created using the R-MAT graph generator.

    The resulting graph is weighted with random values of a uniform distribution
    from the interval [0, 1)

    scale is used to determine the number of vertices to be generated (num_verts
    = 2^scale), which is also used to determine the data type for the vertex ID
    values in the DataFrame.

    edgefactor determies the number of edges (num_edges = num_edges*edgefactor)

    seed, if specified, will be used as the seed to the RNG.

    unweighted determines if the resulting edgelist will have randomly-generated
    weightes ranging in value between [0, 1). If True, an edgelist with only 2
    columns is returned.
    """
    df = rmat(
        scale,
        (2**scale)*edgefactor,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed or 42,
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=None,  # return edgelist instead of Graph instance
        mg=False
    )
    if not unweighted:
        rng = np.random.default_rng(seed)
        df["weight"] = rng.random(size=len(df))
    return df


def read_csv(input_csv_file, scale):
    """
    Returns a cudf DataFrame from reading input_csv_file.

    All input CSV files should be weighted with random values of a uniform
    distribution from the interval [0, 1) in order to best simulate the output
    of a Graph500-compliant graph generator.

    scale is used to determine the data type for the vertex ID values in the
    DataFrame. (num verts = 2^scale), which is used to determine
    """
    vertex_t = "int32" if scale <= 32 else "int64"
    dtypes = [vertex_t, vertex_t, "float32"]
    names=["src", "dst", "weight"],

    chunksize = cugraph.dask.get_chunksize(input_csv_file)
    return cudf.read_csv(input_csv_file,
                         chunksize=chunksize,
                         delimiter=" ",
                         #names=names,
                         dtype=dtypes,
                         header=None,
                        )


################################################################################
# Benchmarked functions
#
# The "benchmark_name" attr is used by the benchmark infra for reporting and is
# set to assign more meaningful names to be displayed in reports.

def construct_graph(dataframe, symmetric=False):
    """
    dataframe contains weighted and undirected edges with self loops. Multiple
    edges will likely be present as well.  The returned Graph object must be
    symmetrized and have self loops removed.
    """
    if symmetric:
        G = cugraph.Graph(directed=False)
    else:
        G = cugraph.Graph(directed=True)

    if len(dataframe.columns) > 2:
        G.from_cudf_edgelist(
            dataframe, source="src", destination="dst", edge_attr="weight")
        #G.from_cudf_edgelist(
        #    dataframe, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(
            dataframe, source="src", destination="dst")
        #G.from_cudf_edgelist(
        #    dataframe, source="0", destination="1")
    return G
construct_graph.benchmark_name = "from_cudf_edgelist"


def bfs(G, start):
    return cugraph.bfs(G, start=start)


def sssp(G, start):
    return cugraph.sssp(G, source=start)


def wcc(G):
    return cugraph.weakly_connected_components(G)


def louvain(G):
    return cugraph.louvain(G)


def pagerank(G):
    return cugraph.pagerank(G)


def katz(G, alpha=None):
    return cugraph.katz_centrality(G, alpha)

def hits(G):
    return cugraph.hits(G)

def uniform_neighbor_sample(G, start_list=None, fanout_vals=None):
    # convert list to cudf.Series
    start_list = cudf.Series(start_list, dtype="int32")  
    return cugraph.uniform_neighbor_sample(
        G, start_list=start_list, fanout_vals=fanout_vals)

def triangle_count(G):
    return cugraph.triangle_count(G)

def eigenvector_centrality(G):
    return cugraph.eigenvector_centrality(G)

################################################################################
# Session-wide setup and teardown

def setup(*args, **kwargs):
    return tuple()


def teardown(*args, **kwargs):
    pass
