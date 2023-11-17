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

from cugraph.utilities.utils import import_optional, MissingModule

torch_geometric = import_optional("torch_geometric")

HAS_PYG_24 = None
if not isinstance(torch_geometric, MissingModule):
    major, minor, patch = torch_geometric.__version__.split(".")[:3]
    pyg_version = tuple(map(int, [major, minor, patch]))
    HAS_PYG_24 = pyg_version >= (2, 4, 0)


# TODO: Remove this function when dropping support to pyg 2.3
def convert_edge_type_key(edge_type_str):
    """Convert an edge_type string to one that follows PyG's convention.

    Pre v2.4.0, the keys of nn.ModuleDict in HeteroConv use
    "author__writes__paper" style." It has been changed to
    "<author___writes___paper>" since 2.4.0.
    """
    if HAS_PYG_24:
        return f"<{'___'.join(edge_type_str.split('__'))}>"

    return edge_type_str
