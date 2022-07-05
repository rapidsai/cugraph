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


from cugraph.experimental.datasets.dataset import (
    Dataset,
    load_all,
    set_config,
    set_download_dir,
    get_download_dir
#    karate,
#    dolphins,
#    SMALL_DATASETS
)
from cugraph.experimental.datasets import metadata


karate = Dataset("metadata/karate.yaml")
karate_undirected = Dataset("metadata/karate_undirected.yaml")
karate_asymmetric = Dataset("metadata/karate-asymmetric.yaml")
dolphins = Dataset("metadata/dolphins.yaml")
polbooks = Dataset("metadata/polbooks.yaml")
netscience = Dataset("metadata/netscience.yaml")
cyber = Dataset("metadata/cyber.yaml")
small_line = Dataset("metadata/small_line.yaml")
small_tree = Dataset("metadata/small_tree.yaml")


# LARGE DATASETS
LARGE_DATASETS = [cyber]

# <10,000 lines
MEDIUM_DATASETS = [netscience, polbooks]

# <500 lines
SMALL_DATASETS = [karate, karate_undirected, small_line, small_tree, dolphins]

# ALL
ALL_DATASETS = [karate, karate_undirected, dolphins, netscience, polbooks, cyber,
                small_line, small_tree]