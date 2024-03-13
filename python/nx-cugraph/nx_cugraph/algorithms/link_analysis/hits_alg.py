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

__all__ = ["hits"]


@networkx_algorithm(
    extra_params={
        'weight : string or None, optional (default="weight")': (
            "The edge attribute to use as the edge weight."
        ),
        **_dtype_param,
    },
    version_added="23.12",
    _plc="hits",
)
def hits(
    G,
    max_iter=100,
    tol=1.0e-8,
    nstart=None,
    normalized=True,
    *,
    weight="weight",
    dtype=None,
):
    G = _to_graph(G, weight, 1, np.float32)
    if (N := len(G)) == 0:
        return {}, {}
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    if nstart is not None:
        nstart = G._dict_to_nodearray(nstart, 0, dtype)
    if max_iter <= 0:
        if nx.__version__[:3] <= "3.2":
            raise ValueError("`maxiter` must be a positive integer.")
        raise nx.PowerIterationFailedConvergence(max_iter)
    try:
        node_ids, hubs, authorities = plc.hits(
            resource_handle=plc.ResourceHandle(),
            graph=G._get_plc_graph(weight, 1, dtype, store_transposed=True),
            tol=tol,
            initial_hubs_guess_vertices=(
                None if nstart is None else cp.arange(N, dtype=index_dtype)
            ),
            initial_hubs_guess_values=nstart,
            max_iter=max_iter,
            normalized=normalized,
            do_expensive_check=False,
        )
    except RuntimeError as exc:
        # Errors from PLC are sometimes a little scary and not very helpful
        raise nx.PowerIterationFailedConvergence(max_iter) from exc
    return (
        G._nodearrays_to_dict(node_ids, hubs),
        G._nodearrays_to_dict(node_ids, authorities),
    )
