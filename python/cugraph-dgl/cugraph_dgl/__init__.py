# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

# to prevent rapids context being created when importing cugraph_dgl
os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from cugraph_dgl.cugraph_storage import CuGraphStorage
from cugraph_dgl.convert import cugraph_storage_from_heterograph
import cugraph_dgl.dataloading
import cugraph_dgl.nn
