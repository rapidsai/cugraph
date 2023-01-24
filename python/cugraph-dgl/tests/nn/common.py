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
from cugraph.utilities.utils import import_optional

th = import_optional("torch")
dgl = import_optional("dgl")


def create_graph1():
    u = th.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = th.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    return g
