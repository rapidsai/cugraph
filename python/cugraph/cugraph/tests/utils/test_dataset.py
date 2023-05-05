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


import yaml
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import gc

import pytest
from cudf.testing.testing import assert_frame_equal
import cupy as cp

from cugraph.experimental.datasets import ALL_DATASETS, ALL_DATASETS_WGT, SMALL_DATASETS
from cugraph.structure import Graph
from cugraph.testing import RAPIDS_DATASET_ROOT_DIR_PATH

# Add the sg marker to all tests in this module.
pytestmark = pytest.mark.sg

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


# Use this to simulate a fresh API import
@pytest.fixture
def datasets():
    from cugraph.experimental import datasets

    yield datasets
    del datasets
    clear_locals()


def clear_locals():
    for dataset in ALL_DATASETS:
        dataset._edgelist = None
        dataset._graph = None
        dataset._path = None


# We use this to create tempfiles that act as config files when we call
# set_config(). Arguments passed will act as custom download directories
def create_config(custom_path="custom_storage_location"):
    config_yaml = """
                    fetch: False
                    force: False
                    download_dir: None
                    """
    c = yaml.safe_load(config_yaml)
    c["download_dir"] = custom_path

    outfile = NamedTemporaryFile()
    with open(outfile.name, "w") as f:
        yaml.dump(c, f, sort_keys=False)

    return outfile


# setting download_dir to None effectively re-initialized the default
def test_env_var(datasets):
    os.environ["RAPIDS_DATASET_ROOT_DIR"] = "custom_storage_location"
    datasets.set_download_dir(None)

    expected_path = Path("custom_storage_location").absolute()
    assert datasets.get_download_dir() == expected_path

    del os.environ["RAPIDS_DATASET_ROOT_DIR"]


def test_home_dir(datasets):
    datasets.set_download_dir(None)
    expected_path = Path.home() / ".cugraph/datasets"

    assert datasets.get_download_dir() == expected_path


def test_set_config(datasets):
    cfg = create_config()
    datasets.set_config(cfg.name)

    assert datasets.get_download_dir() == Path("custom_storage_location").absolute()

    cfg.close()


def test_set_download_dir(datasets):
    tmpd = TemporaryDirectory()
    datasets.set_download_dir(tmpd.name)

    assert datasets.get_download_dir() == Path(tmpd.name).absolute()

    tmpd.cleanup()


@pytest.mark.skip(
    reason="Timeout errors; see: https://github.com/rapidsai/cugraph/issues/2810"
)
def test_load_all(datasets):
    tmpd = TemporaryDirectory()
    cfg = create_config(custom_path=tmpd.name)
    datasets.set_config(cfg.name)
    datasets.load_all()

    for data in datasets.ALL_DATASETS:
        file_path = Path(tmpd.name) / (
            data.metadata["name"] + data.metadata["file_type"]
        )
        assert file_path.is_file()

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_fetch(dataset, datasets):
    tmpd = TemporaryDirectory()
    cfg = create_config(custom_path=tmpd.name)
    datasets.set_config(cfg.name)

    E = dataset.get_edgelist(fetch=True)

    assert E is not None
    assert dataset.get_path().is_file()

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset, datasets):
    datasets.set_download_dir(RAPIDS_DATASET_ROOT_DIR_PATH)
    E = dataset.get_edgelist(fetch=True)

    assert E is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset, datasets):
    datasets.set_download_dir(RAPIDS_DATASET_ROOT_DIR_PATH)
    G = dataset.get_graph(fetch=True)

    assert G is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_metadata(dataset):
    M = dataset.metadata

    assert M is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_path(dataset, datasets):
    tmpd = TemporaryDirectory()
    datasets.set_download_dir(tmpd.name)
    dataset.get_edgelist(fetch=True)

    assert dataset.get_path().is_file()
    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS_WGT)
def test_weights(dataset, datasets):
    datasets.set_download_dir(RAPIDS_DATASET_ROOT_DIR_PATH)

    G_w = dataset.get_graph(fetch=True)
    G = dataset.get_graph(fetch=True, ignore_weights=True)

    assert G_w.is_weighted()
    assert not G.is_weighted()


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_create_using(dataset, datasets):
    datasets.set_download_dir(RAPIDS_DATASET_ROOT_DIR_PATH)

    G_d = dataset.get_graph()
    G_t = dataset.get_graph(create_using=Graph)
    G = dataset.get_graph(create_using=Graph(directed=True))

    assert not G_d.is_directed()
    assert not G_t.is_directed()
    assert G.is_directed()


def test_ctor_with_datafile(datasets):
    from cugraph.experimental.datasets import karate

    karate_csv = RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv"

    # test that only a metadata file or csv can be specified, not both
    with pytest.raises(ValueError):
        datasets.Dataset(metadata_yaml_file="metadata_file", csv_file=karate_csv)

    # ensure at least one arg is provided
    with pytest.raises(ValueError):
        datasets.Dataset()

    # ensure csv file has all other required args (col names and col dtypes)
    with pytest.raises(ValueError):
        datasets.Dataset(csv_file=karate_csv)

    with pytest.raises(ValueError):
        datasets.Dataset(csv_file=karate_csv, csv_col_names=["src", "dst", "wgt"])

    # test with file that DNE
    with pytest.raises(FileNotFoundError):
        datasets.Dataset(
            csv_file="/some/file/that/does/not/exist",
            csv_col_names=["src", "dst", "wgt"],
            csv_col_dtypes=["int32", "int32", "float32"],
        )

    expected_karate_edgelist = karate.get_edgelist()

    # test with file path as string, ensure fetch=True does not break
    ds = datasets.Dataset(
        csv_file=karate_csv.as_posix(),
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )
    assert_frame_equal(ds.get_edgelist(fetch=True), expected_karate_edgelist)
    assert str(ds) == "karate"
    assert ds.get_path() == karate_csv

    # test with file path as Path object
    ds = datasets.Dataset(
        csv_file=karate_csv,
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )
    assert_frame_equal(ds.get_edgelist(), expected_karate_edgelist)
    assert str(ds) == "karate"
    assert ds.get_path() == karate_csv


def test_unload(datasets):
    email_csv = RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"

    ds = datasets.Dataset(
        csv_file=email_csv.as_posix(),
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )

    device = cp.cuda.Device(0)

    free_memory_before = device.mem_info[0]
    df = ds.get_edgelist()
    free_memory_after_load = device.mem_info[0]

    assert free_memory_before > free_memory_after_load

    del df
    gc.collect()
    free_memory_after_delete = device.mem_info[0]

    assert free_memory_before > free_memory_after_delete

    ds.unload()
    free_memory_after_unload = device.mem_info[0]

    assert free_memory_after_unload == free_memory_before
