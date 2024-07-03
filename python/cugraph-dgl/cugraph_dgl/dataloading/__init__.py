# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from cugraph_dgl.dataloading.dataset import (
    HomogenousBulkSamplerDataset,
    HeterogenousBulkSamplerDataset,
)

from cugraph_dgl.dataloading.sampler import Sampler
from cugraph_dgl.dataloading.neighbor_sampler import NeighborSampler

from cugraph_dgl.dataloading.dask_dataloader import DaskDataLoader
from cugraph_dgl.dataloading.dataloader import DataLoader as FutureDataLoader


def DataLoader(*args, **kwargs):
    warnings.warn(
        "DataLoader has been renamed to DaskDataLoader.  "
        "In Release 24.10, cugraph_dgl.dataloading.FutureDataLoader "
        "will take over the DataLoader name.",
        FutureWarning,
    )
    return DaskDataLoader(*args, **kwargs)
