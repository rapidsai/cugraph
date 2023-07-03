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

from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH, RAPIDS_DATASET_ROOT_DIR

#
# Moved Dataset Batches
#

# FIXME: it is hard to keep track of which datasets are included in certain batches and which are not. maybe convert this to a single fixture that let's you import datasets by name and yields a list? eg. get_batch("A", "B", C") => [A, B, C]
# batches

from cugraph.datasets import (
    karate,
    dolphins,
    polbooks,
    netscience,
    small_line,
    small_tree,
    email_Eu_core,
    ktruss_polbooks,
)

DATASETS_UNDIRECTED = [karate, dolphins]
DATASETS_UNDIRECTED_WEIGHTS = [netscience]
DATASETS_SMALL = [karate, dolphins, polbooks]
STRONGDATASETS = [dolphins, netscience, email_Eu_core]
DATASETS_KTRUSS = [(polbooks, ktruss_polbooks)]
MEDIUM_DATASETS = [polbooks]
SMALL_DATASETS = [karate, dolphins, netscience]
RLY_SMALL_DATASETS = [small_line, small_tree]
ALL_DATASETS = [karate, dolphins, netscience, polbooks, small_line, small_tree]
ALL_DATASETS_WGT = [karate, dolphins, netscience, polbooks, small_line, small_tree]
TEST_GROUP = [dolphins, netscience]

# FIXME: removed karate variant. check if unit tests are breaking
DATASETS_UNRENUMBERED = []
DATASETS = [dolphins, netscience]
