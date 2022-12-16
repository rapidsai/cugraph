# Copyright (c) 2022, NVIDIA CORPORATION.
#
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
import cudf
import dask_cudf

import cugraph
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from cugraph.experimental import datasets
from cugraph.generators import rmat


# Graph creation extensions (these are assumed to return a Graph object)
def create_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
    dataset_obj = getattr(datasets, dataset_name)
    return dataset_obj.get_graph(fetch=True)


def create_property_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
    dataset_obj = getattr(datasets, dataset_name)
    edgelist_df = dataset_obj.get_edgelist(fetch=True)

    if mg and (server is not None) and server.is_multi_gpu():
        G = MGPropertyGraph()
        edgelist_df = dask_cudf.from_cudf(edgelist_df)
    else:
        G = PropertyGraph()

    G.add_edge_data(edgelist_df, vertex_col_names=["src", "dst"])
    return G


def create_graph_from_rmat_generator(
    scale,
    num_edges,
    a=0.57,  # from Graph500
    b=0.19,  # from Graph500
    c=0.19,  # from Graph500
    seed=42,
    clip_and_flip=False,
    scramble_vertex_ids=False,  # FIXME: need to understand relevance of this
    mg=False,
    server=None,
):
    if mg and (server is not None) and server.is_multi_gpu:
        is_mg = True
    else:
        is_mg = False

    edgelist_df = rmat(
        scale,
        num_edges,
        a,
        b,
        c,
        seed,
        clip_and_flip,
        scramble_vertex_ids,
        create_using=None,  # None == return edgelist
        mg=is_mg,
    )
    edgelist_df["weight"] = np.float32(1)

    # For PropertyGraph, uncomment:
    # if is_mg:
    #     G = MGPropertyGraph()
    # else:
    #     G = PropertyGraph()
    #
    # G.add_edge_data(edgelist_df, vertex_col_names=["src", "dst"])

    # For Graph, uncomment:
    G = cugraph.Graph(directed=True)
    if is_mg:
        G.from_dask_cudf_edgelist(
            edgelist_df, source="src", destination="dst", edge_attr="weight"
        )
    else:
        G.from_cudf_edgelist(
            edgelist_df, source="src", destination="dst", edge_attr="weight"
        )

    return G


# General-purpose extensions
def gen_vertex_list(graph_id, num_start_verts, seed, server=None):
    """
    Create the list of starting vertices by picking num_start_verts random ints
    between 0 and num_verts, then map those to actual vertex IDs.  Since the
    randomly-chosen IDs may not map to actual IDs, keep trying until
    num_start_verts have been picked, or max_tries is reached.
    """
    rng = np.random.default_rng(seed)

    G = server.get_graph(graph_id)
    assert G.renumbered
    num_verts = G.number_of_vertices()

    start_list_set = set()
    max_tries = 10000
    try_num = 0
    while (len(start_list_set) < num_start_verts) and (try_num < max_tries):
        internal_vertex_ids_start_list = rng.choice(
            num_verts, size=num_start_verts, replace=False
        )
        start_list_df = cudf.DataFrame({"vid": internal_vertex_ids_start_list})
        start_list_df = G.unrenumber(start_list_df, "vid")

        if G.is_multi_gpu():
            start_list_series = start_list_df.compute()["vid"]
        else:
            start_list_series = start_list_df["vid"]

        start_list_series.dropna(inplace=True)
        start_list_set.update(set(start_list_series.values_host.tolist()))
        try_num += 1

    start_list = list(start_list_set)
    start_list = start_list[:num_start_verts]
    assert len(start_list) == num_start_verts

    return start_list
