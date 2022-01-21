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

from pylibcugraph.utilities.api_tools import experimental_warning_wrapper

from pylibcugraph._cugraph_c.graphs import EXPERIMENTAL__SGGraph
SGGraph = experimental_warning_wrapper(EXPERIMENTAL__SGGraph)

from pylibcugraph._cugraph_c.resource_handle import EXPERIMENTAL__ResourceHandle
ResourceHandle = experimental_warning_wrapper(EXPERIMENTAL__ResourceHandle)

from pylibcugraph._cugraph_c.graph_properties import EXPERIMENTAL__GraphProperties
GraphProperties = experimental_warning_wrapper(EXPERIMENTAL__GraphProperties)

from pylibcugraph._cugraph_c.pagerank import EXPERIMENTAL__pagerank
pagerank = experimental_warning_wrapper(EXPERIMENTAL__pagerank)
