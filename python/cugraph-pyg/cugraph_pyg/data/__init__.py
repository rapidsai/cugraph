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

import warnings

from cugraph_pyg.data.dask_graph_store import DaskGraphStore
from cugraph_pyg.data.graph_store import GraphStore
from cugraph_pyg.data.feature_store import TensorDictFeatureStore

def CuGraphStore(*args, **kwargs):
    warnings.warn(
        "CuGraphStore has been renamed to DaskGraphStore"
    )
    return DaskGraphStore(*args, **kwargs)
