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

#from cugraph.experimental.datasets import (
#    karate,
#)
from cugraph.experimental.datasets.dataset import (
    Dataset,
    load_all,
#    karate,
#    dolphins,
#    SMALL_DATASETS
)
from cugraph.experimental.datasets import metadata

# SMALL DATASETS
karate = Dataset("metadata/karate.yaml")
dolphins = Dataset("metadata/dolphins.yaml")
polbooks = Dataset("metadata/polbooks.yaml")
netscience = Dataset("metadata/netscience.yaml")
cyber = Dataset("metadata/cyber.yaml")

# MEDIUM DATASETS

# LARGE DATASETS

# GROUPS OF DATASETS
SMALL_DATASETS = [karate, dolphins]