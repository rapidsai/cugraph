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
import dask_cudf

from cugraph.experimental import PropertyGraph, MGPropertyGraph
from cugraph.experimental import datasets
from cugraph.generators import rmat


# Graph creation extensions
def create_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
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
    if mg and (server is not None) and server.is_multi_gpu():
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
    rng = np.random.default_rng(seed)
    edgelist_df["weight"] = rng.random(size=len(edgelist_df))

    if is_mg:
        G = MGPropertyGraph()
    else:
        G = PropertyGraph()

    G.add_edge_data(edgelist_df, vertex_col_names=["src", "dst"])
    return G
