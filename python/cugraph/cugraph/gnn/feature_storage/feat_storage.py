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
from cugraph.utilities.utils import import_optional

torch = import_optional("torch")


class FeatureStore:
    """The feature-store class used to store feature data for GNNS"""

    def __init__(self, backend="numpy"):
        self.fd = defaultdict(dict)
        if backend not in ["numpy", "torch"]:
            raise ValueError(
                f"backend {backend} not supported. Supported backends are numpy, torch"
            )
        self.backend = backend

    def add_data(self, feat_obj: Sequence, type_name: str, feat_name: str) -> None:
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
            feat_obj, self.backend
        )

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

        return self.fd[feat_name][type_name][indices]

    def get_feature_list(self) -> list[str]:
        return {feat_name: feats.keys() for feat_name, feats in self.fd.items()}

    @staticmethod
    def _cast_feat_obj_to_backend(feat_obj, backend: str):
        if backend == "numpy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_numpy_ar(feat_obj.values)
            else:
                return _cast_to_numpy_ar(feat_obj)
        elif backend == "torch":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_torch_tensor(feat_obj.values)
            else:
                return _cast_to_torch_tensor(feat_obj)


def _cast_to_torch_tensor(ar):
    if isinstance(ar, cp.ndarray):
        ar = torch.as_tensor(ar, device="cuda")
    elif isinstance(ar, np.ndarray):
        ar = torch.from_numpy(ar)
    else:
        ar = torch.as_tensor(ar)
    return ar


def _cast_to_numpy_ar(ar):
    if isinstance(ar, cp.ndarray):
        ar = ar.get()
    elif type(ar).__name__ == "Tensor":
        ar = ar.numpy()
    else:
        ar = np.asarray(ar)
    return ar
