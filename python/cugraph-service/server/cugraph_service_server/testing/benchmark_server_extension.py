# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import dask_cudf

import cugraph
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from cugraph.experimental import datasets
from cugraph.generators import rmat


# Graph creation extensions (these are assumed to return a Graph object)
def create_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
    dataset_obj = getattr(datasets, dataset_name)
    # FIXME: create an MG graph if server is mg?
    return dataset_obj.get_graph(fetch=True)


def create_property_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
    dataset_obj = getattr(datasets, dataset_name)
    edgelist_df = dataset_obj.get_edgelist(fetch=True)

    if mg and (server is not None) and server.is_multi_gpu:
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
    if mg:
        if (server is not None) and server.is_multi_gpu:
            is_mg = True
        else:
            raise RuntimeError(
                f"{mg=} was specified but the server is not indicating "
                "it is MG-capable."
            )
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
            edgelist_df,
            source="src",
            destination="dst",
            edge_attr="weight",
            legacy_renum_only=True,
        )
    else:
        G.from_cudf_edgelist(
            edgelist_df,
            source="src",
            destination="dst",
            edge_attr="weight",
            legacy_renum_only=True,
        )

    return G


# General-purpose extensions
def gen_vertex_list(graph_id, num_verts_to_return, num_verts_in_graph, server=None):
    """
    Returns a list of num_verts_to_return vertex IDs from the Graph referenced
    by graph_id.
    """
    seed = 42
    G = server.get_graph(graph_id)

    if not (isinstance(G, cugraph.Graph)):
        raise TypeError(
            f"{graph_id=} must refer to a cugraph.Graph instance, " f"got: {type(G)}"
        )

    # vertex_list is a random sampling of the src verts.
    # Dask series only support the frac arg for getting n samples.
    srcs = G.edgelist.edgelist_df["src"]
    frac = num_verts_to_return / num_verts_in_graph
    vertex_list = srcs.sample(frac=frac, random_state=seed)

    # Attempt to automatically handle a dask Series
    if hasattr(vertex_list, "compute"):
        vertex_list = vertex_list.compute()

    # frac does not guarantee exactly num_verts_to_return, so ensure only
    # num_verts_to_return are returned
    vertex_list = vertex_list[:num_verts_to_return]
    assert len(vertex_list) == num_verts_to_return

    return vertex_list.to_cupy()
