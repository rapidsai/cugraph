# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import warnings

import networkx as nx
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import (
    _dtype_param,
    _groupby,
    _seed_to_int,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["louvain_communities"]

# max_level argument was added to NetworkX 3.3
if nx.__version__[:3] <= "3.2":
    _max_level_param = {
        "max_level : int, optional": (
            "Upper limit of the number of macro-iterations (max: 500)."
        )
    }
else:
    _max_level_param = {}


@not_implemented_for("directed")
@networkx_algorithm(
    extra_params={
        **_max_level_param,
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
    _plc="louvain",
)
def louvain_communities(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
    *,
    dtype=None,
):
    """`seed` parameter is currently ignored, and self-loops are not yet supported."""
    # NetworkX allows both directed and undirected, but cugraph only allows undirected.
    seed = _seed_to_int(seed)  # Unused, but ensure it's valid for future compatibility
    G = _to_undirected_graph(G, weight)
    if G.src_indices.size == 0:
        return [{key} for key in G._nodeiter_to_iter(range(len(G)))]
    if max_level is None:
        max_level = 500
    elif max_level > 500:
        warnings.warn(
            f"max_level is set too high (={max_level}), setting it to 500.",
            UserWarning,
            stacklevel=2,
        )
        max_level = 500
    node_ids, clusters, modularity = plc.louvain(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(weight, 1, dtype),
        max_level=max_level,  # TODO: add this parameter to NetworkX
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )
    groups = _groupby(clusters, node_ids, groups_are_canonical=True)
    return [set(G._nodearray_to_list(ids)) for ids in groups.values()]


@louvain_communities._can_run
def _(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
    *,
    dtype=None,
):
    # NetworkX allows both directed and undirected, but cugraph only allows undirected.
    return not G.is_directed()
