# Copyright (c) 2025, NVIDIA CORPORATION.
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

from pathlib import Path
import requests
import gzip
import pandas as pd
import time


class Timer:
    """
    Context manager to make timing blocks of code easier and cleaner.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f'Starting "{self.name}"...', end="", flush=True)
        self.start_time = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, traceback):
        run_time = time.perf_counter_ns() - self.start_time
        if exc_type is exc_value is traceback is None:
            print(f"done. Time was {(run_time/1e9):.4g} s", flush=True)


def get_edgelist():
    # From https://snap.stanford.edu/data/cit-Patents.html
    url = "https://snap.stanford.edu/data/cit-Patents.txt.gz"
    gz_file_name = Path(url.split("/")[-1])
    csv_file_name = Path(gz_file_name.stem)
    if csv_file_name.exists():
        print(f"{csv_file_name} already exists, not downloading.")
    else:
        print(f"downloading {url}...", end="", flush=True)
        req = requests.get(url)
        open(gz_file_name, "wb").write(req.content)
        print("done")
        print(f"unzipping {gz_file_name}...", end="", flush=True)
        with gzip.open(gz_file_name, "rb") as gz_in:
            with open(csv_file_name, "wb") as txt_out:
                txt_out.write(gz_in.read())
        print("done")

    print("reading csv to dataframe...", end="", flush=True)
    pandas_edgelist = pd.read_csv(
        csv_file_name.name,
        skiprows=4,
        delimiter="\t",
        names=["src", "dst"],
        dtype={"src": "int32", "dst": "int32"},
    )
    print()
    return pandas_edgelist


if __name__ == "__main__":
    import networkx as nx

    # import logging; logging.basicConfig(level=logging.DEBUG)

    pandas_edgelist = get_edgelist()
    G = nx.from_pandas_edgelist(pandas_edgelist, source="src", target="dst")
    print(
        f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    ###############################################################################
    # NetworkX+nx-cugraph
    ###############################################################################
    # Run with env var NX_CUGRAPH_AUTOCONFIG=True
    with Timer("leiden_communities using NetworkX+nx-cugraph"):
        c = nx.community.leiden_communities(G)
    print(f"Number of communities: {len(c)}")
    # Run a NX algo to ensure Graph needs no conversion to native NX Graph when timing leiden from other libs.
    # Update: doesn't seem to affect perf.
    # nx.pagerank(G)  #; print("pagerank done")

    ###############################################################################
    # cuGraph
    ###############################################################################
    import cugraph

    with Timer("cugraph graph from pandas edgelist"):
        Gcg = cugraph.from_pandas_edgelist(
            pandas_edgelist, source="src", destination="dst"
        )
    with Timer("leiden using cuGraph"):
        (df, modularity) = cugraph.leiden(Gcg)
    del Gcg
    print(f"Number of communities: {len(df['partition'].unique())}")

    ###############################################################################
    # pylibcugraph
    ###############################################################################
    import pylibcugraph as plc
    import cupy as cp

    # with Timer("PLC graph from NX graph"):
    #     edges = G.edges(data=False)
    #     (srcs, dsts) = zip(*edges)
    #     resource_handle = plc.ResourceHandle()
    #     graph_props = plc.GraphProperties(is_symmetric=True, is_multigraph=False)
    #     Gplc = plc.SGGraph(resource_handle,
    #                        graph_props,
    #                        src_or_offset_array=cp.asarray(srcs, dtype="int32"),
    #                        dst_or_index_array=cp.asarray(dsts, dtype="int32"),
    #                        weight_array=cp.ones(len(srcs), dtype="float32"),
    #                        store_transposed=False,
    #                        renumber=True,
    #                        do_expensive_check=False,
    #                        vertices_array=cp.asarray(list(set(srcs + dsts)), dtype="int32"),
    #                        drop_multi_edges=True,
    #                        symmetrize=True,
    #                        input_array_format="COO",
    #                        )
    with Timer("PLC graph from pandas edgelist"):
        srcs = cp.asarray(pandas_edgelist["src"], dtype="int32")
        dsts = cp.asarray(pandas_edgelist["dst"], dtype="int32")
        unique_vertices = cp.unique(cp.concatenate((srcs, dsts)))
        resource_handle = plc.ResourceHandle()
        Gplc = plc.SGGraph(
            resource_handle,
            graph_properties=plc.GraphProperties(
                is_symmetric=True, is_multigraph=False
            ),
            src_or_offset_array=srcs,
            dst_or_index_array=dsts,
            weight_array=cp.ones(len(srcs), dtype="float32"),
            store_transposed=False,
            renumber=True,
            do_expensive_check=False,
            vertices_array=unique_vertices,
            drop_multi_edges=True,
            symmetrize=True,
            input_array_format="COO",
        )
        del srcs, dsts, unique_vertices
    with Timer("leiden using pylibcugraph"):
        (vertices, clusters, modularity) = plc.leiden(
            resource_handle,
            random_state=None,
            graph=Gplc,
            max_level=100,
            resolution=1.0,
            theta=1.0,
            do_expensive_check=False,
        )
    del resource_handle, Gplc
    print(f"Number of communities: {cp.unique(clusters).shape[0]}")

    ###############################################################################
    # Graspologic
    ###############################################################################
    from graspologic.partition import leiden, hierarchical_leiden

    with Timer("leiden using Graspologic"):
        c = leiden(G)
    print(f"Number of communities: {len(set(c.values()))}")

    with Timer("hierarchical_leiden using Graspologic"):
        c = hierarchical_leiden(G)
    print(f"Number of communities: {len(set([cl.cluster for cl in c]))}")

    ###############################################################################
    # igraph
    ###############################################################################
    import igraph

    with Timer("creating igraph graph from networkx graph"):
        Gig = igraph.Graph.from_networkx(G)
    # with Timer("igraph graph from pandas edgelist"):
    #    Gig = igraph.Graph(edges=list(pandas_edgelist.itertuples(index=False, name=None)), directed=False)
    with Timer("leiden using igraph"):
        c = Gig.community_leiden(objective_function="modularity", n_iterations=-1)
    print(f"Number of communities: {len(c)}")

    ###############################################################################
    # leidenalg
    ###############################################################################
    import leidenalg

    with Timer("leiden using leidenalg"):
        c = leidenalg.find_partition(Gig, leidenalg.ModularityVertexPartition)
    print(f"Number of communities: {len(c)}")

    ###############################################################################
    # sknetwork
    ###############################################################################
    import sknetwork

    # Leiden did not finish after ~12 hours, but Louvain finished in about 60 seconds, bug?
    # leiden = sknetwork.clustering.leiden()
    # edgelist = list(pandas_edgelist.itertuples(index=False, name=None))
    # adjacency = sknetwork.utils.from_edge_list(edgelist, directed=False, reindex=True)["adjacency"]
    # with Timer("leiden using sknetwork"):
    #    c = leiden.fit_predict(adjacency)
    # print(f"Number of communities: {len(set(c))}")

    louvain = sknetwork.clustering.Louvain()
    edgelist = list(pandas_edgelist.itertuples(index=False, name=None))
    adjacency = sknetwork.utils.from_edge_list(edgelist, directed=False, reindex=True)[
        "adjacency"
    ]
    with Timer("louvain using sknetwork"):
        c = louvain.fit_predict(adjacency)
    print(f"Number of communities: {len(set(c))}")

    ###############################################################################
    # cdlib
    ###############################################################################
    import cdlib.algorithms

    with Timer("leiden using cdlib (on NetworkX graph)"):
        c = cdlib.algorithms.leiden(G)
    print(f"Number of communities: {len(c.communities)}")

    ###############################################################################
    # igraph
    ###############################################################################
    import igraph

    Gig = cdlib.utils.convert_graph_formats(
        G, desired_format=igraph.Graph, directed=False
    )
    with Timer("leiden using cdlib (on igraph graph)"):
        c = cdlib.algorithms.leiden(Gig)
    print(f"Number of communities: {len(c.communities)}")
