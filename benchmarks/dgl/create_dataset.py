# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

import os
import rmm
from cugraph.generators import rmat

def create_dataset(scale, edgefactor, folder_path):
    _seed = 42
    num_edges = (2**scale) * edgefactor
    seed = _seed
    edgelist_df = rmat(
        scale,
        num_edges,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed,
        clip_and_flip=False,
        scramble_vertex_ids=False,  # FIXME: need to understand relevance of this
        create_using=None,  # None == return edgelist
        mg=False,
    )
    filepath = os.path.join(folder_path, f'rmat_scale_{scale}_edgefactor_{edgefactor}.parquet')
    edgelist_df.to_parquet(filepath)



folder_path = os.path.join(os.getcwd(), 'datasets') 
os.makedirs(folder_path, exist_ok=True)
rmm.reinitialize(managed_memory=True)
for scale in [24,25,26]:
    edgefactor = 16
    create_dataset(scale=scale, edgefactor=edgefactor, folder_path=folder_path)
