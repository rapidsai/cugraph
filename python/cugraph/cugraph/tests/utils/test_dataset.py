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


import os
from pathlib import Path
from tempfile import TemporaryDirectory
import gc

import pytest

from cugraph.structure import Graph
from cugraph.testing import RAPIDS_DATASET_ROOT_DIR_PATH
from cugraph.experimental.datasets import (
    ALL_DATASETS,
    ALL_DATASETS_WGT,
    SMALL_DATASETS,
)
from cugraph.experimental import datasets

# Add the sg marker to all tests in this module.
pytestmark = pytest.mark.sg


###############################################################################
# Fixtures

# module fixture - called once for this module
@pytest.fixture(scope="module")
def tmpdir():
    """
    Create a tmp dir for downloads, etc., run a test, then cleanup when the
    test is done.
    """
    tmpd = TemporaryDirectory()
    yield tmpd
    # teardown
    tmpd.cleanup()


# function fixture - called once for each function in this module
@pytest.fixture(scope="function", autouse=True)
def setup(tmpdir):
    """
    Fixture used for individual test setup and teardown. This ensures each
    Dataset object starts with the same state and cleans up when the test is
    done.
    """
    # FIXME: this relies on dataset features (unload) which themselves are
    # being tested in this module.
    for dataset in ALL_DATASETS:
        dataset.unload()
    gc.collect()

    datasets.set_download_dir(tmpdir.name)

    yield

    # teardown
    for dataset in ALL_DATASETS:
        dataset.unload()
    gc.collect()


###############################################################################
# Tests

# setting download_dir to None effectively re-initialized the default
def test_env_var():
    os.environ["RAPIDS_DATASET_ROOT_DIR"] = "custom_storage_location"
    datasets.set_download_dir(None)

    expected_path = Path("custom_storage_location").absolute()
    assert datasets.get_download_dir() == expected_path

    del os.environ["RAPIDS_DATASET_ROOT_DIR"]


def test_home_dir():
    datasets.set_download_dir(None)
    expected_path = Path.home() / ".cugraph/datasets"

    assert datasets.get_download_dir() == expected_path


def test_set_download_dir():
    tmpd = TemporaryDirectory()
    datasets.set_download_dir(tmpd.name)

    assert datasets.get_download_dir() == Path(tmpd.name).absolute()

    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_fetch(dataset):
    E = dataset.get_edgelist(fetch=True)

    assert E is not None
    assert dataset.get_path().is_file()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset):
    E = dataset.get_edgelist(fetch=True)
    assert E is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset):
    G = dataset.get_graph(fetch=True)
    assert G is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_metadata(dataset):
    M = dataset.metadata

    assert M is not None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_path(dataset):
    tmpd = TemporaryDirectory()
    datasets.set_download_dir(tmpd.name)
    dataset.get_edgelist(fetch=True)

    assert dataset.get_path().is_file()
    tmpd.cleanup()


@pytest.mark.parametrize("dataset", ALL_DATASETS_WGT)
def test_weights(dataset):
    G = dataset.get_graph(fetch=True)
    assert G.is_weighted()
    G = dataset.get_graph(fetch=True, ignore_weights=True)
    assert not G.is_weighted()


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_create_using(dataset):
    G = dataset.get_graph(fetch=True)
    assert not G.is_directed()
    G = dataset.get_graph(fetch=True, create_using=Graph)
    assert not G.is_directed()
    G = dataset.get_graph(fetch=True, create_using=Graph(directed=True))
    assert G.is_directed()


def test_ctor_with_datafile():
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

    expected_karate_edgelist = karate.get_edgelist(fetch=True)

    # test with file path as string, ensure fetch=True does not break
    ds = datasets.Dataset(
        csv_file=karate_csv.as_posix(),
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )
    # cudf.testing.testing.assert_frame_equal() would be good to use to
    # compare, but for some reason it seems to be holding a reference to a
    # dataframe and gc.collect() does not free everything
    el = ds.get_edgelist()
    assert len(el) == len(expected_karate_edgelist)
    assert str(ds) == "karate"
    assert ds.get_path() == karate_csv

    # test with file path as Path object
    ds = datasets.Dataset(
        csv_file=karate_csv,
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )
    el = ds.get_edgelist()
    assert len(el) == len(expected_karate_edgelist)
    assert str(ds) == "karate"
    assert ds.get_path() == karate_csv


def test_unload():
    email_csv = RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv"

    ds = datasets.Dataset(
        csv_file=email_csv.as_posix(),
        csv_col_names=["src", "dst", "wgt"],
        csv_col_dtypes=["int32", "int32", "float32"],
    )

    # FIXME: another (better?) test would be to check free memory and assert
    # the memory use increases after get_*(), then returns to the pre-get_*()
    # level after unload(). However, that type of test may fail for several
    # reasons (the device being monitored is accidentally also being used by
    # another process, and the use of memory pools to name two). Instead, just
    # test that the internal members get cleared on unload().
    assert ds._edgelist is None

    ds.get_edgelist()
    assert ds._edgelist is not None
    ds.unload()
    assert ds._edgelist is None

    ds.get_graph()
    assert ds._edgelist is not None
    ds.unload()
    assert ds._edgelist is None
