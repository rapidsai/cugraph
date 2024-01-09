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

import cudf
import dask_cudf
import yaml
import os
import pandas as pd
import cugraph.dask as dcg
from pathlib import Path
import urllib.request
from cugraph.structure.graph_classes import Graph


class DefaultDownloadDir:
    """
    Maintains a path to be used as a default download directory.

    All DefaultDownloadDir instances are based on RAPIDS_DATASET_ROOT_DIR if
    set, or _default_base_dir if not set.

    Instances of this class are typically shared by several Dataset instances
    in order to allow for the download directory to be defined and updated by
    a single object.
    """

    _default_base_dir = Path.home() / ".cugraph/datasets"

    def __init__(self, *, subdir=""):
        """
        subdir can be specified to provide a specialized dir under the base dir.
        """
        self._subdir = Path(subdir)
        self.reset()

    @property
    def path(self):
        return self._path.absolute()

    @path.setter
    def path(self, new):
        self._path = Path(new)

    def reset(self):
        self._basedir = Path(
            os.environ.get("RAPIDS_DATASET_ROOT_DIR", self._default_base_dir)
        )
        self._path = self._basedir / self._subdir


default_download_dir = DefaultDownloadDir()


class Dataset:
    """
    A Dataset Object, used to easily import edgelist data and cuGraph.Graph
    instances.

    Parameters
    ----------
    meta_data_file_name : yaml file
        The metadata file for the specific graph dataset, which includes
        information on the name, type, url link, data loading format, graph
        properties
    """

    def __init__(
        self,
        metadata_yaml_file=None,
        csv_file=None,
        csv_header=None,
        csv_delim=" ",
        csv_col_names=None,
        csv_col_dtypes=None,
    ):
        self._metadata_file = None
        self._dl_path = default_download_dir
        self._edgelist = None
        self._path = None

        if metadata_yaml_file is not None and csv_file is not None:
            raise ValueError("cannot specify both metadata_yaml_file and csv_file")

        elif metadata_yaml_file is not None:
            with open(metadata_yaml_file, "r") as file:
                self.metadata = yaml.safe_load(file)
                self._metadata_file = Path(metadata_yaml_file)

        elif csv_file is not None:
            if csv_col_names is None or csv_col_dtypes is None:
                raise ValueError(
                    "csv_col_names and csv_col_dtypes must both be "
                    "not None when csv_file is specified."
                )
            self._path = Path(csv_file)
            if self._path.exists() is False:
                raise FileNotFoundError(csv_file)
            self.metadata = {
                "name": self._path.with_suffix("").name,
                "file_type": ".csv",
                "url": None,
                "header": csv_header,
                "delim": csv_delim,
                "col_names": csv_col_names,
                "col_types": csv_col_dtypes,
            }

        else:
            raise ValueError("must specify either metadata_yaml_file or csv_file")

    def __str__(self):
        """
        Use the basename of the meta_data_file the instance was constructed with,
        without any extension, as the string repr.
        """
        # The metadata file is likely to have a more descriptive file name, so
        # use that one first if present.
        # FIXME: this may need to provide a more unique or descriptive string repr
        if self._metadata_file is not None:
            return self._metadata_file.with_suffix("").name
        else:
            return self.get_path().with_suffix("").name

    def __download_csv(self, url):
        """
        Downloads the .csv file from url to the current download path
        (self._dl_path), updates self._path with the full path to the
        downloaded file, and returns the latest value of self._path.
        """
        self._dl_path.path.mkdir(parents=True, exist_ok=True)

        filename = self.metadata["name"] + self.metadata["file_type"]
        if self._dl_path.path.is_dir():
            self._path = self._dl_path.path / filename
            urllib.request.urlretrieve(url, str(self._path))

        else:
            raise RuntimeError(
                f"The directory {self._dl_path.path.absolute()}" "does not exist"
            )
        return self._path

    def unload(self):
        """
        Remove all saved internal objects, forcing them to be re-created when
        accessed.

        NOTE: This will cause calls to get_*() to re-read the dataset file from
        disk. The caller should ensure the file on disk has not moved/been
        deleted/changed.
        """
        self._edgelist = None

    def get_edgelist(self, download=False, reader="cudf"):
        """
        Return an Edgelist.

        Parameters
        ----------
        download : Boolean (default=False)
            Automatically download the dataset from the 'url' location within
            the YAML file.

        reader : 'cudf' or 'pandas' (default='cudf')
            The library used to read a CSV and return an edgelist DataFrame.
        """
        if self._edgelist is None:
            full_path = self.get_path()
            if not full_path.is_file():
                if download:
                    full_path = self.__download_csv(self.metadata["url"])
                else:
                    raise RuntimeError(
                        f"The datafile {full_path} does not"
                        " exist. Try setting download=True"
                        " to download the datafile"
                    )

            header = None
            if isinstance(self.metadata["header"], int):
                header = self.metadata["header"]

            if reader == "cudf":
                self.__reader = cudf.read_csv
            elif reader == "pandas":
                self.__reader = pd.read_csv
            else:
                raise ValueError(
                    "reader must be a module with a read_csv function compatible with \
                     cudf.read_csv"
                )

            self._edgelist = self.__reader(
                filepath_or_buffer=full_path,
                delimiter=self.metadata["delim"],
                names=self.metadata["col_names"],
                dtype={
                    self.metadata["col_names"][i]: self.metadata["col_types"][i]
                    for i in range(len(self.metadata["col_types"]))
                },
                header=header,
            )

        return self._edgelist.copy()

    def get_dask_edgelist(self, download=False):
        """
        Return a distributed Edgelist.

        Parameters
        ----------
        download : Boolean (default=False)
            Automatically download the dataset from the 'url' location within
            the YAML file.
        """
        if self._edgelist is None:
            full_path = self.get_path()
            if not full_path.is_file():
                if download:
                    full_path = self.__download_csv(self.metadata["url"])
                else:
                    raise RuntimeError(
                        f"The datafile {full_path} does not"
                        " exist. Try setting download=True"
                        " to download the datafile"
                    )

            header = None
            if isinstance(self.metadata["header"], int):
                header = self.metadata["header"]

            blocksize = dcg.get_chunksize(full_path)
            self._edgelist = dask_cudf.read_csv(
                path=full_path,
                blocksize=blocksize,
                delimiter=self.metadata["delim"],
                names=self.metadata["col_names"],
                dtype={
                    self.metadata["col_names"][i]: self.metadata["col_types"][i]
                    for i in range(len(self.metadata["col_types"]))
                },
                header=header,
            )

        return self._edgelist.copy()

    def get_graph(
        self,
        download=False,
        create_using=Graph,
        ignore_weights=False,
        store_transposed=False,
    ):
        """
        Return a Graph object.

        Parameters
        ----------
        download : Boolean (default=False)
            Downloads the dataset from the web.

        create_using: cugraph.Graph (instance or class), optional
        (default=Graph)
            Specify the type of Graph to create. Can pass in an instance to
            create a Graph instance with specified 'directed' attribute.

        ignore_weights : Boolean (default=False)
            Ignores weights in the dataset if True, resulting in an
            unweighted Graph. If False (the default), weights from the
            dataset -if present- will be applied to the Graph. If the
            dataset does not contain weights, the Graph returned will
            be unweighted regardless of ignore_weights.

        store_transposed: Boolean (default=False)
            If True, stores the transpose of the adjacency matrix.  Required
            for certain algorithms, such as pagerank.
        """
        if self._edgelist is None:
            self.get_edgelist(download)

        if create_using is None:
            G = Graph()
        elif isinstance(create_using, Graph):
            # what about BFS if transposed is True
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        elif issubclass(create_using, Graph):
            G = create_using()
        else:
            raise TypeError(
                "create_using must be a cugraph.Graph "
                "(or subclass) type or instance, got: "
                f"{type(create_using)}"
            )
        if len(self.metadata["col_names"]) > 2 and not (ignore_weights):
            G.from_cudf_edgelist(
                self._edgelist,
                source=self.metadata["col_names"][0],
                destination=self.metadata["col_names"][1],
                edge_attr=self.metadata["col_names"][2],
                store_transposed=store_transposed,
            )
        else:
            G.from_cudf_edgelist(
                self._edgelist,
                source=self.metadata["col_names"][0],
                destination=self.metadata["col_names"][1],
                store_transposed=store_transposed,
            )
        return G

    def get_dask_graph(
        self,
        download=False,
        create_using=Graph,
        ignore_weights=False,
        store_transposed=False,
    ):
        """
        Return a distributed Graph object.

        Parameters
        ----------
        download : Boolean (default=False)
            Downloads the dataset from the web.

        create_using: cugraph.Graph (instance or class), optional
        (default=Graph)
            Specify the type of Graph to create. Can pass in an instance to
            create a Graph instance with specified 'directed' attribute.

        ignore_weights : Boolean (default=False)
            Ignores weights in the dataset if True, resulting in an
            unweighted Graph. If False (the default), weights from the
            dataset -if present- will be applied to the Graph. If the
            dataset does not contain weights, the Graph returned will
            be unweighted regardless of ignore_weights.

        store_transposed : bool, optional (default=False)
            If True, stores the transpose of the adjacency matrix.  Required
            for certain algorithms.
        """
        if self._edgelist is None:
            self.get_dask_edgelist(download)

        if create_using is None:
            G = Graph()
        elif isinstance(create_using, Graph):
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        elif issubclass(create_using, Graph):
            G = create_using()
        else:
            raise TypeError(
                "create_using must be a cugraph.Graph "
                "(or subclass) type or instance, got: "
                f"{type(create_using)}"
            )

        if len(self.metadata["col_names"]) > 2 and not (ignore_weights):
            G.from_dask_cudf_edgelist(
                self._edgelist,
                source=self.metadata["col_names"][0],
                destination=self.metadata["col_names"][1],
                edge_attr=self.metadata["col_names"][2],
                store_transposed=store_transposed,
            )
        else:
            G.from_dask_cudf_edgelist(
                self._edgelist,
                source=self.metadata["col_names"][0],
                destination=self.metadata["col_names"][1],
                store_transposed=store_transposed,
            )
        return G

    def get_path(self):
        """
        Returns the location of the stored dataset file.
        """
        if self._path is None:
            self._path = self._dl_path.path / (
                self.metadata["name"] + self.metadata["file_type"]
            )

        return self._path.absolute()

    def is_directed(self):
        """
        Returns True if the graph is a directed graph.
        """
        return self.metadata["is_directed"]

    def is_multigraph(self):
        """
        Returns True if the graph is a multigraph.
        """
        return self.metadata["is_multigraph"]

    def is_symmetric(self):
        """
        Returns True if the graph is symmetric.
        """
        return self.metadata["is_symmetric"]

    def number_of_nodes(self):
        """
        An alias of number_of_vertices()
        """
        return self.number_of_vertices()

    def number_of_vertices(self):
        """
        Get the number of vertices in the graph.
        """
        return self.metadata["number_of_nodes"]

    def number_of_edges(self):
        """
        Get the number of edges in the graph.
        """
        return self.metadata["number_of_edges"]


def download_all(force=False):
    """
    Looks in `metadata` directory and downloads all datafiles from the the URLs
    provided in each YAML file.

    Parameters
    force : Boolean (default=False)
        Overwrite any existing copies of datafiles.
    """
    default_download_dir.path.mkdir(parents=True, exist_ok=True)

    meta_path = Path(__file__).parent.absolute() / "metadata"
    for file in meta_path.iterdir():
        meta = None
        if file.suffix == ".yaml":
            with open(meta_path / file, "r") as metafile:
                meta = yaml.safe_load(metafile)

            if "url" in meta:
                filename = meta["name"] + meta["file_type"]
                save_to = default_download_dir.path / filename
                if not save_to.is_file() or force:
                    urllib.request.urlretrieve(meta["url"], str(save_to))


def set_download_dir(path):
    """
    Set the download location for datasets

    Parameters
    ----------
    path : String
        Location used to store datafiles
    """
    if path is None:
        default_download_dir.reset()
    else:
        default_download_dir.path = path


def get_download_dir():
    return default_download_dir.path
