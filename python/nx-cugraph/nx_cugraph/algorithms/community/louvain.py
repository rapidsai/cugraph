# Copyright (c) 2023, NVIDIA CORPORATION.
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

import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import (
    _groupby,
    _seed_to_int,
    networkx_algorithm,
    not_implemented_for,
)

from ..isolate import _isolates

__all__ = ["louvain_communities"]


@not_implemented_for("directed")
@networkx_algorithm(
    extra_params={
        "max_level : int, optional": (
            "Upper limit of the number of macro-iterations (max: 500)."
        )
    }
)
def louvain_communities(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None, *, max_level=None
):
    """`seed` parameter is currently ignored."""
    # NetworkX allows both directed and undirected, but cugraph only allows undirected.
    seed = _seed_to_int(seed)  # Unused, but ensure it's valid for future compatibility
    G = _to_undirected_graph(G, weight)
    if G.src_indices.size == 0:
        # TODO: PLC doesn't handle empty graphs gracefully!
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
    vertices, clusters, modularity = plc.louvain(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        max_level=max_level,  # TODO: add this parameter to NetworkX
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=False,
    )
    groups = _groupby(clusters, vertices, groups_are_canonical=True)
    rv = [set(G._nodearray_to_list(node_ids)) for node_ids in groups.values()]
    # TODO: PLC doesn't handle isolated vertices yet, so this is a temporary fix
    isolates = _isolates(G)
    if isolates.size > 0:
        isolates = isolates[isolates > vertices.max()]
        if isolates.size > 0:
            rv.extend({node} for node in G._nodearray_to_list(isolates))
    return rv


@louvain_communities._can_run
def _(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None, *, max_level=None
):
    # NetworkX allows both directed and undirected, but cugraph only allows undirected.
    return not G.is_directed()
