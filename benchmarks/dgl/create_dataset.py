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
from dask_cuda import LocalCUDACluster
from distributed import Client
from cugraph.dask.comms import comms as Comms
import numpy as np
_seed = 42

def start_cluster():
    cluster = LocalCUDACluster(protocol='tcp',rmm_managed_memory=True)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    return cluster, client


def create_edgelist_df(scale, edgefactor):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
  
    # Assume strings are names of datasets in the datasets package
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
        mg=True,
    )
    edgelist_df["weight"] = np.float32(1)
    return edgelist_df


if __name__ == "__main__":
    cluster, client = start_cluster()
    folder_path = '/datasets/vjawa/gnn_data/' 
    os.makedirs(folder_path, exist_ok=True)
    for scale in [28]:
        edgefactor = 16
        edgelist_df = create_edgelist_df(scale, edgefactor)
        edgelist_df.to_parquet(folder_path+f'/mg_scale_{scale}_edgefactor_{edgefactor}.parquet')
