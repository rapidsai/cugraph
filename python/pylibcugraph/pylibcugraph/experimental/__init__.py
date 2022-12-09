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

"""
The "experimental" package contains packages, functions, classes, etc. that
are ready for use but do not have their API signatures or implementation
finalized yet. This allows users to provide early feedback while still
permitting bigger design changes to take place.

ALL APIS IN EXPERIMENTAL ARE SUBJECT TO CHANGE OR REMOVAL.

Calling experimental objects will raise a PendingDeprecationWarning warning.

If an object is "promoted" to the public API, the experimental namespace will
continue to also have that object present for at least another release.  A
different warning will be output in that case, indicating that the experimental
API has been promoted and will no longer be importable from experimental much
longer.
"""

from pylibcugraph.utilities.api_tools import (
    experimental_warning_wrapper,
    promoted_experimental_warning_wrapper,
)

# experimental_warning_wrapper() wraps the object in a function that provides
# the appropriate warning about using experimental code.

# promoted_experimental_warning_wrapper() is used instead when an object is present
# in both the experimental namespace and its final, public namespace.

# The convention of naming functions with the "EXPERIMENTAL__" prefix
# discourages users from directly importing experimental objects that don't have
# the appropriate warnings, such as what the wrapper and the "experimental"
# namespace name provides.

from pylibcugraph.graphs import SGGraph

SGGraph = promoted_experimental_warning_wrapper(SGGraph)

from pylibcugraph.graphs import MGGraph

MGGraph = promoted_experimental_warning_wrapper(MGGraph)

from pylibcugraph.resource_handle import ResourceHandle

ResourceHandle = promoted_experimental_warning_wrapper(ResourceHandle)

from pylibcugraph.graph_properties import GraphProperties

GraphProperties = promoted_experimental_warning_wrapper(GraphProperties)

from pylibcugraph.pagerank import pagerank

pagerank = promoted_experimental_warning_wrapper(pagerank)

from pylibcugraph.sssp import sssp

sssp = promoted_experimental_warning_wrapper(sssp)

from pylibcugraph.hits import hits

hits = promoted_experimental_warning_wrapper(hits)

from pylibcugraph.node2vec import node2vec

node2vec = promoted_experimental_warning_wrapper(node2vec)

from pylibcugraph.jaccard_coefficients import EXPERIMENTAL__jaccard_coefficients

jaccard_coefficients = experimental_warning_wrapper(EXPERIMENTAL__jaccard_coefficients)

from pylibcugraph.overlap_coefficients import EXPERIMENTAL__overlap_coefficients

overlap_coefficients = experimental_warning_wrapper(EXPERIMENTAL__overlap_coefficients)

from pylibcugraph.sorensen_coefficients import EXPERIMENTAL__sorensen_coefficients

sorensen_coefficients = experimental_warning_wrapper(
    EXPERIMENTAL__sorensen_coefficients
)
