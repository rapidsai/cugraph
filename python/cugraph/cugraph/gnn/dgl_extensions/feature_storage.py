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


import cudf
import dask_cudf
import cupy as cp
from cugraph.experimental import MGPropertyGraph


class CuFeatureStorage:
    """
    Storage for node/edge feature data.
    """

    def __init__(
        self, pg, columns, storage_type, backend_lib="torch", indices_offset=0
    ):
        self.pg = pg
        self.columns = columns
        if backend_lib == "torch":
            from torch.utils.dlpack import from_dlpack
        elif backend_lib == "tf":
            from tensorflow.experimental.dlpack import from_dlpack
        elif backend_lib == "cupy":
            from cupy import from_dlpack
        else:
            raise NotImplementedError(
                f"Only PyTorch ('torch'), TensorFlow ('tf'), and CuPy ('cupy') "
                f"backends are currently supported, got {backend_lib=}"
            )
        if storage_type not in ["edge", "node"]:
            raise NotImplementedError("Only edge and node storage is supported")

        self.storage_type = storage_type

        self.from_dlpack = from_dlpack
        self.indices_offset = indices_offset

    def fetch(self, indices, device=None, pin_memory=False, **kwargs):
        """Fetch the features of the given node/edge IDs to the
        given device.

        Parameters
        ----------
        indices : Tensor
            Node or edge IDs.
        device : Device
            Device context.
        pin_memory :

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        # Default implementation uses synchronous fetch.

        indices = cp.asarray(indices)
        if isinstance(self.pg, MGPropertyGraph):
            # dask_cudf loc breaks if we provide cudf series/cupy array
            # https://github.com/rapidsai/cudf/issues/11877
            indices = indices.get()
        else:
            indices = cudf.Series(indices)

        indices = indices + self.indices_offset

        if self.storage_type == "node":
            subset_df = self.pg.get_vertex_data(
                vertex_ids=indices, columns=self.columns
            )
        else:
            subset_df = self.pg.get_edge_data(edge_ids=indices, columns=self.columns)

        subset_df = subset_df[self.columns]

        if isinstance(subset_df, dask_cudf.DataFrame):
            subset_df = subset_df.compute()

        if len(subset_df) == 0:
            raise ValueError(f"indices = {indices} not found in FeatureStorage")
        cap = subset_df.to_dlpack()
        tensor = self.from_dlpack(cap)
        del cap
        if device:
            if not isinstance(tensor, cp.ndarray):
                # Cant transfer to different device for cupy
                tensor = tensor.to(device)
        return tensor
