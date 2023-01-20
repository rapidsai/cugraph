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
from cugraph.gnn import FeatureStore
from cugraph.utilities.utils import import_optional

torch = import_optional("torch")


class dgl_FeatureStorage:
    """
    Storage for node/edge feature data.
    """

    def __init__(self, fs: FeatureStore, type_name: str, feat_name: str):
        self.fs = fs
        self.type_name = type_name
        self.feat_name = feat_name

    def fetch(self, indices, device=None, pin_memory=False, **kwargs):
        """Fetch the features of the given node/edge IDs to the
        given device.
        Parameters
        ----------
        indices : Tensor
            Node or edge IDs.
        device : Device
            Device context.
        pin_memory : bool
            Wether to use pin_memory for fetching features
            pin_memory=True is currently not supported

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        if pin_memory:
            raise ValueError("pinned memory not supported in dgl_FeatureStorage")
        if isinstance(indices, torch.Tensor):
            indices = indices.long()
        t = self.fs.get_data(
            indices=indices, type_name=self.type_name, feat_name=self.feat_name
        )
        if device:
            return t.to(device)
        else:
            return t
