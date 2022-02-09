# Copyright (c) 2022, NVIDIA CORPORATION.
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

# import cudf
from cugraph.sampling import node2vec_wrapper
# from cugraph.utilities import ensure_cugraph_obj_for_nx


def node2vec(G, start_vertices, max_depth, use_padding, p, q):
    """
    TODO: Write description of node2vec, plus parameters, returns, and
    possible examples.

    Computes node2vec.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    start_vertices: cudf.Series

    max_depth: int, optional

    use_padding: bool, optional

    p: double, optional

    q: double, optional

    """
    # FIXME: Fix call and perform checks of parameters
    return node2vec_wrapper.node2vec()
