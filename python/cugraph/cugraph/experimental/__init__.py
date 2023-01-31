# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from cugraph.utilities.api_tools import deprecated_warning_wrapper
from cugraph.utilities.api_tools import promoted_experimental_warning_wrapper

from cugraph.structure.property_graph import EXPERIMENTAL__PropertyGraph

PropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__PropertyGraph)

from cugraph.structure.property_graph import EXPERIMENTAL__PropertySelection

PropertySelection = experimental_warning_wrapper(EXPERIMENTAL__PropertySelection)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertyGraph

MGPropertyGraph = experimental_warning_wrapper(EXPERIMENTAL__MGPropertyGraph)

from cugraph.dask.structure.mg_property_graph import EXPERIMENTAL__MGPropertySelection

MGPropertySelection = experimental_warning_wrapper(EXPERIMENTAL__MGPropertySelection)

# FIXME: Remove experimental.triangle_count next release
from cugraph.community.triangle_count import triangle_count

triangle_count = promoted_experimental_warning_wrapper(triangle_count)

from cugraph.experimental.components.scc import EXPERIMENTAL__strong_connected_component

strong_connected_component = experimental_warning_wrapper(
    EXPERIMENTAL__strong_connected_component
)

from cugraph.experimental.structure.bicliques import EXPERIMENTAL__find_bicliques

find_bicliques = deprecated_warning_wrapper(
    experimental_warning_wrapper(EXPERIMENTAL__find_bicliques)
)

from cugraph.experimental.datasets.dataset import Dataset

from cugraph.experimental.link_prediction.jaccard import (
    EXPERIMENTAL__jaccard,
    EXPERIMENTAL__jaccard_coefficient,
)

jaccard = experimental_warning_wrapper(EXPERIMENTAL__jaccard)
jaccard_coefficient = experimental_warning_wrapper(EXPERIMENTAL__jaccard_coefficient)

from cugraph.experimental.link_prediction.sorensen import (
    EXPERIMENTAL__sorensen,
    EXPERIMENTAL__sorensen_coefficient,
)

sorensen = experimental_warning_wrapper(EXPERIMENTAL__sorensen)
sorensen_coefficient = experimental_warning_wrapper(EXPERIMENTAL__sorensen_coefficient)

from cugraph.experimental.link_prediction.overlap import (
    EXPERIMENTAL__overlap,
    EXPERIMENTAL__overlap_coefficient,
)

overlap = experimental_warning_wrapper(EXPERIMENTAL__overlap)
overlap_coefficient = experimental_warning_wrapper(EXPERIMENTAL__overlap_coefficient)

from cugraph.gnn.data_loading import EXPERIMENTAL__BulkSampler

BulkSampler = experimental_warning_wrapper(EXPERIMENTAL__BulkSampler)
