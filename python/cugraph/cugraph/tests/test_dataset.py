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
import os
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


# Helper function to restore the default cfg
def default_config():
    set_config(Path(__file__).parent.parent /
               "experimental/datasets/datasets_config.yaml")


# We use this to create tempfiles that act as config files when we call
# set_config(). Arguments passed will be handled in _custom_config and
# act as custom download directories
@pytest.fixture()
def create_config(**kwargs):
    # FIXME: remove inner function def
    def _custom_config(**kwargs):
        custom_path = kwargs.pop("name", "custom_storage_location")

        config_yaml = """
                        fetch: False
                        force: False
                        download_dir':
                        """
        c = yaml.safe_load(config_yaml)
        c['download_dir'] = custom_path
        print(c)

        outfile = NamedTemporaryFile()
        with open(outfile.name, 'w') as f:
            yaml.dump(c, f, sort_keys=False)

        return outfile

    return _custom_config


def test_set_config(create_config):
    cfg = create_config()
    set_config(cfg.name)

    assert str(karate.view_config()) == "custom_storage_location"
    assert str(dolphins.view_config()) == "custom_storage_location"
    assert str(netscience.view_config()) == "custom_storage_location"
    assert str(polbooks.view_config()) == "custom_storage_location"
    default_config()


@pytest.skip(msg="wip")
def test_get_path():
    ...


def test_load_all(create_config):
    tmpd = TemporaryDirectory()
    cfg = create_config(name=tmpd.name)
    set_config(cfg.name)
    load_all()

    for data in ALL_DATASETS:
        file_path = Path(tmpd.name) / (data.metadata['name'] +
                                       data.metadata['file_type'])
        print(str(file_path))
        assert os.path.isfile(file_path)

    os.listdir(tmpd.name)
    tmpd.cleanup()
    default_config()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset):
    E = dataset.get_edgelist()

    assert E is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset):
    G = dataset.get_graph()

    assert G is not None
