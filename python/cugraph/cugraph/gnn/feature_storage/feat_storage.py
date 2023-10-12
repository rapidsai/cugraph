# Copyright (c) 2023, NVIDIA CORPORATION.
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
from __future__ import annotations
from collections import defaultdict
from typing import Sequence, Union
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")


class FeatureStore:
    """The feature-store class used to store feature data for GNNs"""

    def __init__(
        self,
        backend: str = "numpy",
        wg_comm: object = None,
        wg_type: str = None,
        wg_location: str = None,
    ):
        """
        Constructs a new FeatureStore object

        Parameters:
        ----------
        backend: str ('numpy', 'torch', 'wholegraph')
            Optional (default='numpy')
            The name of the backend to use.

        wg_comm: WholeMemoryCommunicator
            Optional (default=automatic)
            Only used with the 'wholegraph' backend.
            The communicator to use to store features in WholeGraph.

        wg_type: str ('distributed', 'continuous', 'chunked')
            Optional (default='distributed')
            Only used with the 'wholegraph' backend.
            The memory format (distributed, continuous, or chunked) of
            this FeatureStore.  For more information see the WholeGraph
            documentation.

        wg_location: str ('cpu', 'cuda')
            Optional (default='cuda')
            Only used with the 'wholegraph' backend.
            Where the data is stored (cpu or cuda).
            Defaults to storing on the GPU (cuda).
        """

        self.fd = defaultdict(dict)
        if backend not in ["numpy", "torch", "wholegraph"]:
            raise ValueError(
                f"backend {backend} not supported. "
                "Supported backends are numpy, torch, wholegraph"
            )
        self.backend = backend

        self.__wg_comm = None
        self.__wg_type = None
        self.__wg_location = None

        if backend == "wholegraph":
            self.__wg_comm = (
                wg_comm if wg_comm is not None else wgth.get_local_node_communicator()
            )
            self.__wg_type = wg_type if wg_type is not None else "distributed"
            self.__wg_location = wg_location if wg_location is not None else "cuda"

            if self.__wg_type not in ["distributed", "chunked", "continuous"]:
                raise ValueError(f"invalid memory format {self.__wg_type}")
            if (self.__wg_location != "cuda") and (self.__wg_location != "cpu"):
                raise ValueError(f"invalid location {self.__wg_location}")

    def add_data(
        self, feat_obj: Sequence, type_name: str, feat_name: str, **kwargs
    ) -> None:
        """
        Add the feature data to the feature_storage class
        Parameters:
        ----------
          feat_obj : array_like object
            The feature object to save in feature store
          type_name : str
            The node-type/edge-type of the feature
          feat_name: str
            The name of the feature being stored
        Returns:
        -------
            None
        """
        self.fd[feat_name][type_name] = self._cast_feat_obj_to_backend(
            feat_obj,
            self.backend,
            wg_comm=self.__wg_comm,
            wg_type=self.__wg_type,
            wg_location=self.__wg_location,
            **kwargs,
        )

    def add_data_no_cast(self, feat_obj, type_name: str, feat_name: str) -> None:
        """
        Direct add the feature data to the feature_storage class with no cast
        Parameters:
        ----------
          feat_obj : array_like object
            The feature object to save in feature store
          type_name : str
            The node-type/edge-type of the feature
          feat_name: str
            The name of the feature being stored
        Returns:
        -------
            None
        """
        self.fd[feat_name][type_name] = feat_obj

    def get_data(
        self,
        indices: Union[np.ndarray, torch.Tensor],
        type_name: str,
        feat_name: str,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Retrieve the feature data corresponding to the indices, type and feature name

        Parameters:
        -----------
        indices: np.ndarray or torch.Tensor
            The indices of the values to extract.
        type_name : str
            The node-type/edge-type to store data
        feat_name:
            The feature name to retrieve data for

        Returns:
        --------
        np.ndarray or torch.Tensor
            Array object of the backend type
        """

        if feat_name not in self.fd:
            raise ValueError(
                f"{feat_name} not found in features: {list(self.fd.keys())}"
            )

        if type_name not in self.fd[feat_name]:
            raise ValueError(
                f"type_name {type_name} not found in"
                f" feature: {list(self.fd[feat_name].keys())}"
            )

        feat = self.fd[feat_name][type_name]
        if not isinstance(wgth, MissingModule) and isinstance(
            feat, wgth.WholeMemoryEmbedding
        ):
            indices_tensor = (
                indices
                if isinstance(indices, torch.Tensor)
                else torch.as_tensor(indices, device="cuda")
            )
            return feat.gather(indices_tensor)
        else:
            return feat[indices]

    def get_feature_list(self) -> list[str]:
        return {feat_name: feats.keys() for feat_name, feats in self.fd.items()}

    @staticmethod
    def _cast_feat_obj_to_backend(feat_obj, backend: str, **kwargs):
        if backend == "numpy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_numpy_ar(feat_obj.values, **kwargs)
            else:
                return _cast_to_numpy_ar(feat_obj, **kwargs)
        elif backend == "torch":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_torch_tensor(feat_obj.values, **kwargs)
            else:
                return _cast_to_torch_tensor(feat_obj, **kwargs)
        elif backend == "wholegraph":
            return _get_wg_embedding(feat_obj, **kwargs)


def _get_wg_embedding(feat_obj, wg_comm=None, wg_type=None, wg_location=None, **kwargs):
    wg_comm_obj = wg_comm or wgth.get_local_node_communicator()
    wg_type_str = wg_type or "distributed"
    wg_location_str = wg_location or "cuda"

    if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
        th_tensor = _cast_to_torch_tensor(feat_obj.values)
    else:
        th_tensor = _cast_to_torch_tensor(feat_obj)
    wg_embedding = wgth.create_embedding(
        wg_comm_obj,
        wg_type_str,
        wg_location_str,
        th_tensor.dtype,
        th_tensor.shape,
    )
    (
        local_wg_tensor,
        local_ld_offset,
    ) = wg_embedding.get_embedding_tensor().get_local_tensor()
    local_th_tensor = th_tensor[
        local_ld_offset : local_ld_offset + local_wg_tensor.shape[0]
    ]
    local_wg_tensor.copy_(local_th_tensor)
    wg_comm_obj.barrier()
    return wg_embedding


def _cast_to_torch_tensor(ar, **kwargs):
    if isinstance(ar, cp.ndarray):
        ar = torch.as_tensor(ar, device="cuda")
    elif isinstance(ar, np.ndarray):
        ar = torch.from_numpy(ar)
    else:
        ar = torch.as_tensor(ar)
    return ar


def _cast_to_numpy_ar(ar, **kwargs):
    if isinstance(ar, cp.ndarray):
        ar = ar.get()
    elif type(ar).__name__ == "Tensor":
        ar = ar.numpy()
    else:
        ar = np.asarray(ar)
    return ar
