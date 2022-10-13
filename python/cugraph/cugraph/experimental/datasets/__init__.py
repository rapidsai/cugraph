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
    get_download_dir,
    default_download_dir,
)
from cugraph.experimental.datasets import metadata
from pathlib import Path


meta_path = Path(__file__).parent / "metadata"

karate = Dataset(meta_path / "karate.yaml")
karate_data = Dataset(meta_path / "karate_data.yaml")
karate_undirected = Dataset(meta_path / "karate_undirected.yaml")
karate_asymmetric = Dataset(meta_path / "karate_asymmetric.yaml")
karate_disjoint = Dataset(meta_path / "karate-disjoint.yaml")
dolphins = Dataset(meta_path / "dolphins.yaml")
polbooks = Dataset(meta_path / "polbooks.yaml")
netscience = Dataset(meta_path / "netscience.yaml")
cyber = Dataset(meta_path / "cyber.yaml")
small_line = Dataset(meta_path / "small_line.yaml")
small_tree = Dataset(meta_path / "small_tree.yaml")
toy_graph = Dataset(meta_path / "toy_graph.yaml")
toy_graph_undirected = Dataset(meta_path / "toy_graph_undirected.yaml")
email_Eu_core = Dataset(meta_path / "email-Eu-core.yaml")
ktruss_polbooks = Dataset(meta_path / "ktruss_polbooks.yaml")

DATASETS_UNDIRECTED = [karate, dolphins]

DATASETS_UNDIRECTED_WEIGHTS = [netscience]

DATASETS_UNRENUMBERED = [karate_disjoint]

DATASETS = [dolphins, netscience, karate_disjoint]

DATASETS_SMALL = [karate, dolphins, polbooks]

STRONGDATASETS = [dolphins, netscience, email_Eu_core]

DATASETS_KTRUSS = [(polbooks, ktruss_polbooks)]

MEDIUM_DATASETS = [polbooks]

SMALL_DATASETS = [karate, dolphins, netscience]

RLY_SMALL_DATASETS = [small_line, small_tree]

ALL_DATASETS = [karate, dolphins, netscience, polbooks, small_line, small_tree]

ALL_DATASETS_WGT = [karate, dolphins, netscience, polbooks, small_line, small_tree]

TEST_GROUP = [dolphins, netscience]
