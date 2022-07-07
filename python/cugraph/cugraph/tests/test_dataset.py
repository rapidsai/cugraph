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

from cugraph.experimental.datasets import (set_config, load_all,
                                           set_download_dir, get_download_dir,
                                           SMALL_DATASETS, ALL_DATASETS)


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(autouse=True)
def cleanup_tests():
    yield
    default_config()
    reset_imports()


# Helper function to restore the default cfg
def default_config():
    set_config(Path(__file__).parent.parent /
               "experimental/datasets/datasets_config.yaml")


def reset_imports():
    for dataset in ALL_DATASETS:
        dataset._edgelist = None
        dataset._graph = None
        dataset.path = None


# We use this to create tempfiles that act as config files when we call
# set_config(). Arguments passed will act as custom download directories
def create_config(custom_path="custom_storage_location"):
    config_yaml = """
                    fetch: False
                    force: False
                    download_dir: None
                    """
    c = yaml.safe_load(config_yaml)
    c['download_dir'] = custom_path

    outfile = NamedTemporaryFile()
    with open(outfile.name, 'w') as f:
        yaml.dump(c, f, sort_keys=False)

    return outfile


# User giving the API a custom config file
def test_set_config():
    cfg = create_config()
    set_config(cfg.name)

    assert str(get_download_dir()).endswith("custom_storage_location")

    cfg.close()


def test_set_download_dir():
    tmpd = TemporaryDirectory()
    set_download_dir(tmpd.name)

    assert str(get_download_dir()).endswith(tmpd.name)
    tmpd.cleanup()


@pytest.mark.skip(reason="wip")
def test_home_directory():
    user_home = Path.home() / ".cugraph/datasets"

    assert get_download_dir() == user_home


def test_load_all():
    tmpd = TemporaryDirectory()
    cfg = create_config(custom_path=tmpd.name)
    set_config(cfg.name)
    load_all()

    for data in ALL_DATASETS:
        file_path = Path(tmpd.name) / (data.metadata['name'] +
                                       data.metadata['file_type'])
        assert os.path.isfile(file_path)

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_fetch(dataset):
    tmpd = TemporaryDirectory()
    cfg = create_config(custom_path=tmpd.name)
    set_config(cfg.name)

    E = dataset.get_edgelist(fetch=True)

    assert E is not None
    assert os.path.isfile(dataset.get_path())

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset):
    tmpd = TemporaryDirectory()
    set_download_dir(tmpd.name)
    E = dataset.get_edgelist(fetch=True)

    assert E is not None

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset):
    tmpd = TemporaryDirectory()
    set_download_dir(tmpd.name)
    G = dataset.get_graph(fetch=True)

    assert G is not None

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_metadata(dataset):
    M = dataset.metadata

    assert M is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_path(dataset):
    tmpd = TemporaryDirectory()
    set_download_dir(tmpd.name)
    dataset.get_edgelist(fetch=True)

    assert os.path.isfile(dataset.get_path())
    tmpd.cleanup()


# Path is None until a dataset initializes its edgelist
@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_path_raises(dataset):
    with pytest.raises(RuntimeError):
        dataset.get_path()
