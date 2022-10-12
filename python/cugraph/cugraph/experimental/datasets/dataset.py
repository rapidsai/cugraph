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

import cudf
import yaml
import os
from pathlib import Path
from cugraph.structure.graph_classes import Graph


class DefaultDownloadDir:
    """
    Maintains the path to the download directory used by Dataset instances.
    Instances of this class are typically shared by several Dataset instances
    in order to allow for the download directory to be defined and updated by
    a single object.
    """

    def __init__(self):
        self._path = Path(
            os.environ.get("RAPIDS_DATASET_ROOT_DIR", Path.home() / ".cugraph/datasets")
        )

    @property
    def path(self):
        """
        If `path` is not set, set it to the environment variable
        RAPIDS_DATASET_ROOT_DIR. If the variable is not set, default to the
        user's home directory.
        """
        if self._path is None:
            self._path = Path(
                os.environ.get(
                    "RAPIDS_DATASET_ROOT_DIR", Path.home() / ".cugraph/datasets"
                )
            )
        return self._path

    @path.setter
    def path(self, new):
        self._path = Path(new)

    def clear(self):
        self._path = None


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

    def __init__(self, meta_data_file_name):
        with open(meta_data_file_name, "r") as file:
            self.metadata = yaml.safe_load(file)

        self._dl_path = default_download_dir
        self._edgelist = None
        self._graph = None
        self._path = None
        """
        self._path = self._dl_path.path / (self.metadata['name'] +
                                           self.metadata['file_type'])
        """

    def __download_csv(self, url):
        self._dl_path.path.mkdir(parents=True, exist_ok=True)

        filename = self.metadata["name"] + self.metadata["file_type"]
        if self._dl_path.path.is_dir():
            df = cudf.read_csv(url)
            df.to_csv(self._dl_path.path / filename, index=False)

        else:
            raise RuntimeError(
                f"The directory {self._dl_path.path.absolute()}" "does not exist"
            )

    def get_edgelist(self, fetch=False):
        """
        Return an Edgelist

        Parameters
        ----------
        fetch : Boolean (default=False)
            Automatically fetch for the dataset from the 'url' location within
            the YAML file.
        """

        if self._edgelist is None:
            full_path = self.get_path()
            if not full_path.is_file():
                if fetch:
                    self.__download_csv(self.metadata["url"])
                else:
                    raise RuntimeError(
                        f"The datafile {full_path} does not"
                        " exist. Try get_edgelist(fetch=True)"
                        " to download the datafile"
                    )
            header = None
            if isinstance(self.metadata["header"], int):
                header = self.metadata["header"]
            self._edgelist = cudf.read_csv(
                full_path,
                delimiter=self.metadata["delim"],
                names=self.metadata["col_names"],
                dtype=self.metadata["col_types"],
                header=header,
            )

        return self._edgelist

    def get_graph(self, fetch=False, create_using=Graph, ignore_weights=False):
        """
        Return a Graph object.

        Parameters
        ----------
        fetch : Boolean (default=False)
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
        """
        if self._edgelist is None:
            self.get_edgelist(fetch)

        if create_using is None:
            self._graph = Graph()
        elif isinstance(create_using, Graph):
            # what about BFS if trnaposed is True
            attrs = {"directed": create_using.is_directed()}
            self._graph = type(create_using)(**attrs)
        elif type(create_using) is type:
            self._graph = create_using()
        else:
            raise TypeError(
                "create_using must be a cugraph.Graph "
                "(or subclass) type or instance, got: "
                f"{type(create_using)}"
            )

        if len(self.metadata["col_names"]) > 2 and not (ignore_weights):
            self._graph.from_cudf_edgelist(
                self._edgelist, source="src", destination="dst", edge_attr="wgt"
            )
        else:
            self._graph.from_cudf_edgelist(
                self._edgelist, source="src", destination="dst"
            )

        return self._graph

    def get_path(self):
        """
        Returns the location of the stored dataset file
        """
        self._path = self._dl_path.path / (
            self.metadata["name"] + self.metadata["file_type"]
        )

        return self._path.absolute()


def load_all(force=False):
    """
    Looks in `metadata` directory and fetches all datafiles from the the URLs
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
                    df = cudf.read_csv(meta["url"])
                    df.to_csv(save_to, index=False)


def set_config(cfgpath):
    """
    Read in a custom config file.

    Parameters
    ----------
    cfgfile : String
        Read the custom config file given its path, and override the default
    """
    with open(Path(cfgpath), "r") as file:
        cfg = yaml.safe_load(file)
        default_download_dir.path = Path(cfg["download_dir"])


def set_download_dir(path):
    """
    Set the download directory for fetching datasets

    Parameters
    ----------
    path : String
        Location used to store datafiles
    """
    if path is None:
        default_download_dir.clear()
    else:
        default_download_dir.path = path


def get_download_dir():
    return default_download_dir.path.absolute()
