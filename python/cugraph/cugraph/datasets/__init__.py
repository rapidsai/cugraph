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

from pathlib import Path

# datasets module
from cugraph.datasets.dataset import (
    Dataset,
    download_all,
    set_download_dir,
    get_download_dir,
    default_download_dir,
)
from cugraph.datasets import metadata

# metadata path for .yaml files
meta_path = Path(__file__).parent / "metadata"

cyber = Dataset(meta_path / "cyber.yaml")
dining_prefs = Dataset(meta_path / "dining_prefs.yaml")
dolphins = Dataset(meta_path / "dolphins.yaml")
email_Eu_core = Dataset(meta_path / "email_Eu_core.yaml")
karate = Dataset(meta_path / "karate.yaml")
karate_asymmetric = Dataset(meta_path / "karate_asymmetric.yaml")
karate_disjoint = Dataset(meta_path / "karate_disjoint.yaml")
netscience = Dataset(meta_path / "netscience.yaml")
polbooks = Dataset(meta_path / "polbooks.yaml")
small_line = Dataset(meta_path / "small_line.yaml")
small_tree = Dataset(meta_path / "small_tree.yaml")
toy_graph = Dataset(meta_path / "toy_graph.yaml")
toy_graph_undirected = Dataset(meta_path / "toy_graph_undirected.yaml")

# Benchmarking datasets: be mindful of memory usage
# 250 MB
soc_livejournal = Dataset(meta_path / "soc-livejournal1.yaml")
# 965 MB
cit_patents = Dataset(meta_path / "cit-patents.yaml")
# 1.8 GB
europe_osm = Dataset(meta_path / "europe_osm.yaml")
# 1.5 GB
hollywood = Dataset(meta_path / "hollywood.yaml")
