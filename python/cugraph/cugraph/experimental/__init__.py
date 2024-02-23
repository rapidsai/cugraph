# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

from pylibcugraph.utilities.api_tools import (
    experimental_warning_wrapper,
    promoted_experimental_warning_wrapper,
)

# Passing in the namespace name of this module to the *_wrapper functions
# allows them to bypass the expensive inspect.stack() lookup.
_ns_name = __name__

from cugraph.structure.property_graph import EXPERIMENTAL__PropertyGraph

PropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__PropertyGraph, _ns_name)

from cugraph.structure.property_graph import EXPERIMENTAL__PropertySelection

PropertySelection = experimental_warning_wrapper(
    EXPERIMENTAL__PropertySelection, _ns_name
)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertyGraph

MGPropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__MGPropertyGraph, _ns_name)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertySelection

MGPropertySelection = experimental_warning_wrapper(
    EXPERIMENTAL__MGPropertySelection, _ns_name
)

from cugraph.experimental.components.scc import EXPERIMENTAL__strong_connected_component

strong_connected_component = experimental_warning_wrapper(
    EXPERIMENTAL__strong_connected_component, _ns_name
)

from cugraph.gnn.data_loading import BulkSampler

BulkSampler = promoted_experimental_warning_wrapper(BulkSampler, _ns_name)
