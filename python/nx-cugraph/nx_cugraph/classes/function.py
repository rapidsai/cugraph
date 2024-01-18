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
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

__all__ = ["number_of_selfloops"]


@networkx_algorithm(version_added="23.12")
def number_of_selfloops(G):
    G = _to_graph(G)
    is_selfloop = G.src_indices == G.dst_indices
    return is_selfloop.sum().tolist()
