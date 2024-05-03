# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas
import pytest

import cudf
import cugraph
import dask_cudf
from cugraph import datasets
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.datasets import karate
from cugraph.structure import Graph
from cugraph.testing import (
    RAPIDS_DATASET_ROOT_DIR_PATH,
    ALL_DATASETS,
    WEIGHTED_DATASETS,
    SMALL_DATASETS,
    BENCHMARKING_DATASETS,
)


# Add the sg marker to all tests in this module.
pytestmark = pytest.mark.sg


###############################################################################
# Fixtures
###############################################################################


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
# Helpers


# check if there is a row where src == dst
def has_selfloop(dataset):
    if not dataset.metadata["is_directed"]:
        return False
    df = dataset.get_edgelist(download=True)
    df.rename(columns={df.columns[0]: "src", df.columns[1]: "dst"}, inplace=True)
    res = df.where(df["src"] == df["dst"])

    return res.notnull().values.any()


# check if dataset object is symmetric
def is_symmetric(dataset):
    # undirected graphs are symmetric
    if not dataset.metadata["is_directed"]:
        return True
    else:
        df = dataset.get_edgelist(download=True)
        df_a = df.sort_values("src")

        # create df with swapped src/dst columns
        df_b = None
        if "wgt" in df_a.columns:
            df_b = df_a[["dst", "src", "wgt"]]
        else:
            df_b = df_a[["dst", "src"]]
        df_b.rename(columns={"dst": "src", "src": "dst"}, inplace=True)
        # created a df by appending the two
        res = cudf.concat([df_a, df_b])
        # sort/unique
        res = res.drop_duplicates().sort_values("src")

        return len(df_a) == len(res)


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
def test_download(dataset):
    E = dataset.get_edgelist(download=True)

    assert E is not None
    assert dataset.get_path().is_file()


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_download_dask(dask_client, dataset):
    E = dataset.get_dask_edgelist(download=True)

    assert E is not None
    assert dataset.get_path().is_file()


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_reader(dataset):
    # defaults to using cudf
    E = dataset.get_edgelist(download=True)

    assert E is not None
    assert isinstance(E, cudf.DataFrame)
    dataset.unload()

    # using pandas
    E_pd = dataset.get_edgelist(download=True, reader="pandas")

    assert E_pd is not None
    assert isinstance(E_pd, pandas.DataFrame)
    dataset.unload()

    with pytest.raises(ValueError):
        dataset.get_edgelist(reader="fail")
        dataset.get_edgelist(reader=None)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_reader_dask(dask_client, dataset):
    # using dask_cudf
    E = dataset.get_dask_edgelist(download=True)

    assert E is not None
    assert isinstance(E, dask_cudf.core.DataFrame)
    dataset.unload()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_edgelist(dataset):
    E = dataset.get_edgelist(download=True)

    assert E is not None
    assert isinstance(E, cudf.DataFrame)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_dask_edgelist(dask_client, dataset):
    E = dataset.get_dask_edgelist(download=True)

    assert E is not None
    assert isinstance(E, dask_cudf.DataFrame)


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_graph(dataset):
    G = dataset.get_graph(download=True)
    assert G is not None
    assert isinstance(G, cugraph.Graph)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_dask_graph(dask_client, dataset):
    G = dataset.get_dask_graph(download=True)
    assert G is not None
    # TODO Check G is a DistributedGraph


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_metadata(dataset):
    M = dataset.metadata
    assert M is not None
    assert isinstance(M, dict)


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_get_path(dataset):
    tmpd = TemporaryDirectory()
    datasets.set_download_dir(tmpd.name)
    dataset.get_edgelist(download=True)

    assert dataset.get_path().is_file()
    tmpd.cleanup()


@pytest.mark.parametrize("dataset", WEIGHTED_DATASETS)
def test_weights(dataset):
    G = dataset.get_graph(download=True)
    assert G.is_weighted()
    G = dataset.get_graph(download=True, ignore_weights=True)
    assert not G.is_weighted()


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", WEIGHTED_DATASETS)
def test_weights_dask(dask_client, dataset):
    G = dataset.get_dask_graph(download=True)
    assert G.is_weighted()
    G = dataset.get_dask_graph(download=True, ignore_weights=True)
    assert not G.is_weighted()


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_create_using(dataset):
    G = dataset.get_graph(download=True)
    assert not G.is_directed()
    G = dataset.get_graph(download=True, create_using=Graph)
    assert not G.is_directed()
    G = dataset.get_graph(download=True, create_using=Graph(directed=True))
    assert G.is_directed()

    # using a non-Graph type should raise an error
    with pytest.raises(TypeError):
        dataset.get_graph(download=True, create_using=set)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.skip(reason="MG not supported on CI")
@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_create_using_dask(dask_client, dataset):
    G = dataset.get_dask_graph(download=True)
    assert not G.is_directed()
    G = dataset.get_dask_graph(download=True, create_using=Graph)
    assert not G.is_directed()
    G = dataset.get_dask_graph(download=True, create_using=Graph(directed=True))
    assert G.is_directed()

    # using a non-Graph type should raise an error
    with pytest.raises(TypeError):
        dataset.get_dask_graph(download=True, create_using=set)


def test_ctor_with_datafile():
    from cugraph.datasets import karate

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

    expected_karate_edgelist = karate.get_edgelist(download=True)

    # test with file path as string, ensure download=True does not break
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


@pytest.mark.parametrize("dataset", [karate])
def test_unload(dataset):
    assert dataset._edgelist is None

    dataset.get_edgelist()
    assert dataset._edgelist is not None
    dataset.unload()
    assert dataset._edgelist is None

    dataset.get_graph()
    assert dataset._edgelist is not None
    dataset.unload()
    assert dataset._edgelist is None


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_node_and_edge_count(dataset):
    dataset_is_directed = dataset.metadata["is_directed"]
    G = dataset.get_graph(
        download=True, create_using=Graph(directed=dataset_is_directed)
    )

    assert G.number_of_nodes() == dataset.metadata["number_of_nodes"]
    assert G.number_of_edges() == dataset.metadata["number_of_edges"]


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_is_directed(dataset):
    dataset_is_directed = dataset.metadata["is_directed"]
    G = dataset.get_graph(
        download=True, create_using=Graph(directed=dataset_is_directed)
    )

    assert G.is_directed() == dataset.metadata["is_directed"]


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_has_selfloop(dataset):
    assert has_selfloop(dataset) == dataset.metadata["has_loop"]


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_is_symmetric(dataset):
    assert is_symmetric(dataset) == dataset.metadata["is_symmetric"]


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_is_multigraph(dataset):
    G = dataset.get_graph(download=True)

    assert G.is_multigraph() == dataset.metadata["is_multigraph"]


# The datasets used for benchmarks are in their own test, since downloading them
# repeatedly would increase testing overhead significantly
@pytest.mark.parametrize("dataset", BENCHMARKING_DATASETS)
def test_benchmarking_datasets(dataset):
    dataset_is_directed = dataset.metadata["is_directed"]
    G = dataset.get_graph(
        download=True, create_using=Graph(directed=dataset_is_directed)
    )

    assert G.is_directed() == dataset.metadata["is_directed"]
    assert G.number_of_nodes() == dataset.metadata["number_of_nodes"]
    assert G.number_of_edges() == dataset.metadata["number_of_edges"]
    assert has_selfloop(dataset) == dataset.metadata["has_loop"]
    assert is_symmetric(dataset) == dataset.metadata["is_symmetric"]
    assert G.is_multigraph() == dataset.metadata["is_multigraph"]

    dataset.unload()


@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_object_getters(dataset):
    assert dataset.is_directed() == dataset.metadata["is_directed"]
    assert dataset.is_multigraph() == dataset.metadata["is_multigraph"]
    assert dataset.is_symmetric() == dataset.metadata["is_symmetric"]
    assert dataset.number_of_nodes() == dataset.metadata["number_of_nodes"]
    assert dataset.number_of_vertices() == dataset.metadata["number_of_nodes"]
    assert dataset.number_of_edges() == dataset.metadata["number_of_edges"]
