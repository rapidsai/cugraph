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

import numpy as np


def _get_backend_lib_ar(ar):
    return type(ar).__module__


def _is_vector_feature(col):
    return type(col.dtype).__name__ == "ListDtype"


def _convert_ar_to_numpy(ar):
    if isinstance(ar, list):
        ar = np.asarray(ar)
    else:
        lib_name = _get_backend_lib_ar(ar)
        if lib_name == "torch":
            ar = ar.cpu().numpy()
        elif lib_name == "cupy":
            ar = ar.get()
        elif lib_name == "cudf":
            ar = ar.values.get()
        elif lib_name == "numpy":
            ar = ar
        else:
            raise NotImplementedError(
                f"{lib_name=} not supported yet for conversion to numpy"
            )
    return ar


class CuFeatureStorage:
    """
    Storage for node/edge feature data.
    """

    def __init__(
        self,
        pg,
        column,
        storage_type,
        backend_lib="torch",
        indices_offset=0,
        types_to_fetch=None,
    ):
        self.pg = pg
        self.column = column

        if backend_lib == "torch":
            from torch.utils.dlpack import from_dlpack
        elif backend_lib == "tf":
            from tensorflow.experimental.dlpack import from_dlpack
        elif backend_lib == "cupy":
            from cupy import from_dlpack
        elif backend_lib == "numpy":
            pass
        else:
            raise NotImplementedError(
                f"Only PyTorch ('torch'), TensorFlow ('tf'), and CuPy ('cupy')"
                f"and numpy ('numpy') backends are currently supported, "
                f" got {backend_lib=}"
            )
        if storage_type not in ["edge", "node"]:
            raise NotImplementedError("Only edge and node storage is supported")

        self.storage_type = storage_type

        self.from_dlpack = from_dlpack
        self.indices_offset = indices_offset
        self.types_to_fetch = types_to_fetch

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

        # Handle remote case
        if type(self.pg).__name__ in ["RemotePropertyGraph", "RemoteMGPropertyGraph"]:
            indices = _convert_ar_to_numpy(indices)
            indices = indices + self.indices_offset
            # TODO: Raise Issue
            # We dont support numpy arrays in get_vertex_data, get_edge_data
            # for Remote  Graphs
            indices = indices.tolist()
        else:
            # For local case
            # we rely on cupy to handle various inputs cleanly like  GPU Tensor,
            # cupy array, cudf Series, cpu tensor etc
            import cupy as cp

            indices = cp.asarray(indices)
            indices = indices + self.indices_offset

        if self.storage_type == "node":
            result = self.pg.get_vertex_data(
                vertex_ids=indices, columns=[self.column], types=self.types_to_fetch
            )
        else:
            result = self.pg.get_edge_data(
                edge_ids=indices, columns=[self.column], types=self.types_to_fetch
            )

        if type(result).__name__ == "DataFrame":
            if _is_vector_feature(result[self.column]):
                if self.storage_type == "node":
                    result = self.pg.vertex_vector_property_to_array(
                        result, self.column
                    )
                else:
                    result = self.pg.edge_vector_property_to_array(result, self.column)
                if result.ndim == 2 and result.shape[1] == 1:
                    result = result.squeeze(1)
            else:
                result = result[self.column].values

            if hasattr(result, "compute"):
                result = result.compute()
            if len(result) == 0:
                raise ValueError(f"{indices=} not found in FeatureStorage")
            cap = result.toDlpack()
        else:
            # When backend is not dataframe(pandas, cuDF) we return lists
            result = result[self.column]
            cap = _convert_ar_to_numpy(result)

        if type(cap).__name__ == "PyCapsule":
            tensor = self.from_dlpack(cap)
            del cap
        else:
            tensor = cap
        if device:
            if type(tensor).__module__ == "torch":
                # Can only transfer to different device for pytorch
                tensor = tensor.to(device)
        return tensor
