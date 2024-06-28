# Copyright (c) 2024, NVIDIA CORPORATION.
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

from typing import List, Union, Tuple
from cugraph.utilities.utils import import_optional

from cugraph_dgl.nn import SparseGraph

import pandas
import numpy
import cupy
import cudf

torch = import_optional("torch")
dgl = import_optional("dgl")

TensorType = Union[
    "torch.Tensor",
    "cupy.ndarray",
    "numpy.ndarray",
    "cudf.Series",
    "pandas.Series",
    List[int],
]

DGLSamplerOutput = Tuple[
    "torch.Tensor",
    "torch.Tensor",
    List[Union["dgl.Block", SparseGraph]],
]
