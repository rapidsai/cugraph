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

import cugraph
import cudf
import yaml
import os
from pathlib import Path


class DefaultDownloadDir:
    def __init__(self):
        self._dd = Path(os.environ.get("RAPIDS_DATASET_ROOT_DIR",
                                       Path.home() / ".cugraph/datasets"))

    @property
    def dd(self):
        if self._dd is None:
            self._dd = Path(os.environ.get("RAPIDS_DATASET_ROOT_DIR",
                                           Path.home() / ".cugraph/datasets"))
        return self._dd

    @dd.setter
    def dd(self, new):
        self._dd = new

    def clear(self):
        self._dd = None


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
        self.__dir_path = Path(__file__).parent.absolute()

        self._meta_data_file_name = meta_data_file_name
        self.__read_meta_data_file(self._meta_data_file_name)

        self._dd = default_download_dir.dd
        self._edgelist = None
        self._graph = None
        self._path = None

    def __read_meta_data_file(self, meta_data_file):
        metadata_path = self.__dir_path / meta_data_file
        with open(metadata_path, 'r') as file:
            self.metadata = yaml.safe_load(file)
            file.close()

    # check for updates for download_dir location within obj
    def __refresh_dd(self):
        if self._dd != default_download_dir.dd:
            self._dd = default_download_dir.dd

    def __download_csv(self, url):
        if not os.path.isdir(self._dd):
            os.makedirs(self._dd)

        filename = self.metadata['name'] + self.metadata['file_type']
        if os.path.isdir(self._dd):
            df = cudf.read_csv(url)
            df.to_csv(self._dd / filename, index=False)

        else:
            raise RuntimeError("The directory " + str(self._dd.absolute())
                               + " does not exist")

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
            self.__refresh_dd()
            full_path = self._dd / (self.metadata['name'] +
                                    self.metadata['file_type'])

            if not os.path.isfile(full_path):
                if fetch:
                    self.__download_csv(self.metadata['url'])
                else:
                    raise RuntimeError("The datafile does not exist. Try" +
                                       " get_edgelist(fetch=True) to" +
                                       " download the datafile")

            self._edgelist = cudf.read_csv(full_path,
                                           delimiter=self.metadata['delim'],
                                           names=self.metadata['col_names'],
                                           dtype=self.metadata['col_types'])
            self._path = full_path

        return self._edgelist

    def get_graph(self, fetch=False):
        """
        Return a Graph object.

        Parameters
        ----------
        fetch : Boolean (default=False)
            Automatically fetch for the dataset from the 'url' location within
            the YAML file.
        """
        if self._edgelist is None:
            self.get_edgelist(fetch)

        self._graph = cugraph.Graph(directed=self.metadata['is_directed'])
        self._graph.from_cudf_edgelist(self._edgelist, source='src',
                                       destination='dst')

        return self._graph

    def get_path(self):
        """
            Returns the location of the stored dataset file
        """
        if self._path is None:
            raise RuntimeError("Path to datafile has not been set." +
                               " Call get_edgelist or get_graph first")

        return self._path.absolute()


def load_all(force=False):
    """
    Looks in `metadata` directory and fetches all datafiles from the the URLs
    provided in each YAML file.

    Parameters
        ----------
    """
    if not os.path.isdir(default_download_dir.dd):
        os.makedirs(default_download_dir.dd)

    meta_path = Path(__file__).parent.absolute() / "metadata"
    for file in os.listdir(meta_path):
        meta = None
        if file.endswith('.yaml'):
            with open(meta_path / file, 'r') as metafile:
                meta = yaml.safe_load(metafile)
                metafile.close()

            if 'url' in meta:
                filename = meta['name'] + meta['file_type']
                save_to = default_download_dir.dd / filename
                if not os.path.isfile(save_to) or force:
                    df = cudf.read_csv(meta['url'])
                    df.to_csv(save_to, index=False)


def set_config(cfgpath):
    """
    Read in a custom config file.

    Parameters
    ----------
    cfgfile : String
        Read the custom config file given its path, and override the default
    """
    with open(Path(cfgpath), 'r') as file:
        cfg = yaml.safe_load(file)
        default_download_dir.dd = Path(cfg['download_dir'])
        file.close()


def set_download_dir(path):
    """
    Set the download directory for fetching datasets

    Parameters
    ----------
    path : String
        Use as the storage location
    """
    if path is None:
        default_download_dir.dd = None
    else:
        default_download_dir.dd = Path(path)


def get_download_dir():
    return default_download_dir.dd.absolute()
