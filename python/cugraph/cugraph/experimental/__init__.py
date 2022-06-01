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

from cugraph.utilities.api_tools import experimental_warning_wrapper

from cugraph.structure.property_graph import EXPERIMENTAL__PropertyGraph
PropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__PropertyGraph)

from cugraph.structure.property_graph import EXPERIMENTAL__PropertySelection
PropertySelection = experimental_warning_wrapper(EXPERIMENTAL__PropertySelection)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertyGraph
MGPropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__MGPropertyGraph)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertySelection
MGPropertySelection = experimental_warning_wrapper(EXPERIMENTAL__MGPropertySelection)

from cugraph.experimental.community.triangle_count import \
    EXPERIMENTAL__triangle_count
triangle_count = experimental_warning_wrapper(EXPERIMENTAL__triangle_count)