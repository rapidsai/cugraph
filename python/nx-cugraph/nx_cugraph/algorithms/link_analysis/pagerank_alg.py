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
import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    index_dtype,
    networkx_algorithm,
)

__all__ = ["pagerank"]


@networkx_algorithm(
    extra_params=_dtype_param,
    is_incomplete=True,  # dangling not supported
    plc={"pagerank", "personalized_pagerank"},
    version_added="23.12",
)
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
    *,
    dtype=None,
):
    """`dangling` parameter is not supported, but it is checked for validity."""
    G = _to_graph(G, weight, 1, np.float32)
    if (N := len(G)) == 0:
        return {}
    if dtype is not None:
        dtype = _get_float_dtype(dtype)
    elif weight in G.edge_values:
        dtype = _get_float_dtype(G.edge_values[weight].dtype)
    else:
        dtype = np.float32
    if nstart is not None:
        nstart = G._dict_to_nodearray(nstart, 0, dtype=dtype)
        if (total := nstart.sum()) == 0:
            raise ZeroDivisionError
        nstart /= total
    if personalization is not None:
        personalization = G._dict_to_nodearray(personalization, 0, dtype=dtype)
        if (total := personalization.sum()) == 0:
            raise ZeroDivisionError
        personalization /= total
    if dangling is not None:
        # Check if given dangling is valid even though we don't use it
        dangling = G._dict_to_nodearray(dangling, 0)  # Check validity
        if dangling.sum() == 0:
            raise ZeroDivisionError
        if (G._out_degrees_array() == 0).any():
            raise NotImplementedError("custom dangling weights is not supported")
    if max_iter <= 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    kwargs = {
        "resource_handle": plc.ResourceHandle(),
        "graph": G._get_plc_graph(weight, 1, dtype, store_transposed=True),
        "precomputed_vertex_out_weight_vertices": None,
        "precomputed_vertex_out_weight_sums": None,
        "initial_guess_vertices": None
        if nstart is None
        else cp.arange(N, dtype=index_dtype),
        "initial_guess_values": nstart,
        "alpha": alpha,
        "epsilon": N * tol,
        "max_iterations": max_iter,
        "do_expensive_check": False,
        "fail_on_nonconvergence": False,
    }
    if personalization is None:
        node_ids, values, is_converged = plc.pagerank(**kwargs)
    else:
        node_ids, values, is_converged = plc.personalized_pagerank(
            personalization_vertices=cp.arange(N, dtype=index_dtype),  # Why?
            personalization_values=personalization,
            **kwargs,
        )
    if not is_converged:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return G._nodearrays_to_dict(node_ids, values)


@pagerank._can_run
def _(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
    *,
    dtype=None,
):
    return dangling is None
