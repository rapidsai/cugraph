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

import warnings

from cugraph_pyg.loader.node_loader import NodeLoader
from cugraph_pyg.loader.neighbor_loader import NeighborLoader

from cugraph_pyg.loader.link_loader import LinkLoader
from cugraph_pyg.loader.link_neighbor_loader import LinkNeighborLoader

from cugraph_pyg.loader.dask_node_loader import DaskNeighborLoader

from cugraph_pyg.loader.dask_node_loader import BulkSampleLoader


def CuGraphNeighborLoader(*args, **kwargs):
    warnings.warn(
        "CuGraphNeighborLoader has been renamed to DaskNeighborLoader", FutureWarning
    )
    return DaskNeighborLoader(*args, **kwargs)
