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
import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = ["pagerank"]


@networkx_algorithm
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """`dangling` parameter is not supported."""
    G = _to_graph(G, weight, 1)
    if (N := len(G)) == 0:
        return {}
    if nstart is not None:
        nstart = G._dict_to_nodearray(nstart, 0, dtype=np.float32)
    if personalization is not None:
        personalization = G._dict_to_nodearray(personalization, 0, dtype=np.float32)
        if personalization.sum() == 0:
            raise ZeroDivisionError
    if dangling is not None:
        dangling = G._dict_to_nodearray(dangling, 0)  # Check validity
        if (G._out_degrees_array() == 0).any():
            raise NotImplementedError("custom dangling weights is not supported")
    if max_iter <= 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    kwargs = {
        "resource_handle": plc.ResourceHandle(),
        "graph": G._get_plc_graph(weight, 1, np.float32, store_transposed=True),
        "precomputed_vertex_out_weight_vertices": None,
        "precomputed_vertex_out_weight_sums": None,
        "initial_guess_vertices": None
        if nstart is None
        else cp.arange(N, dtype=index_dtype),  # Why is this necessary?
        "initial_guess_values": nstart,
        "alpha": alpha,
        "epsilon": tol**0.5 / N,  # TODO: see if tol is handled the same by PLC and NX
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
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    return dangling is None
