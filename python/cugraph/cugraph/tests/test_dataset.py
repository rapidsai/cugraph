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

import gc

import pytest
import warnings
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
# from cugraph.testing import utils

from cugraph.experimental.datasets import (karate, dolphins, netscience,
                                           polbooks,
                                           set_config, load_all,
                                           ALL_DATASETS)


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def default_config():
    set_config(Path(__file__).parent.parent /
               "experimental/datasets/datasets_config.yaml")


@pytest.fixture
def create_config():
    config_yaml = """
                  fetch: False
                  force: False
                  download_dir: custom_storage_location
                  """
    c = yaml.safe_load(config_yaml)

    outfile = NamedTemporaryFile()
    with open(outfile.name, 'w') as f:
        yaml.dump(c, f, sort_keys=False)

    yield outfile
    outfile.close()
    default_config()


def test_set_config(create_config):
    set_config(create_config.name)

    assert str(karate.view_config()) == "custom_storage_location"
    assert str(dolphins.view_config()) == "custom_storage_location"
    assert str(netscience.view_config()) == "custom_storage_location"
    assert str(polbooks.view_config()) == "custom_storage_location"


def test_load_all(create_config):
    tmpd = TemporaryDirectory()

    # with open(create_config.name, 'rw') as f:
    #     cfg = yaml.safe_load(f)
    #     print(cfg)
    #     cfg['download_dir'] = tmpd.name
    #     yaml.dump(cfg, f, sort_keys=False)
    #     print(yaml.safe_load(f))

    tmpd.cleanup()

    assert False


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset):
    E = dataset.get_edgelist()

    assert E is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset):
    G = dataset.get_graph()

    assert G is not None
