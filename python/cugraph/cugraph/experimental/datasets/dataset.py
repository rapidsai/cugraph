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


this_dir = Path(os.getenv("this_dir", "cugraph/cugraph/experimental/datasets"))
datasets_dir = this_dir.parent / "datasets"

download_dir = Path("datasets")


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
        self.dir_path = Path(__file__).parent.absolute()
        self.__read_config()
        self.__meta_data_file_name = meta_data_file_name
        self.__read_meta_data_file(self.__meta_data_file_name)
        self.__edgelist = None
        self.__graph = None
        self.path = None

    def __read_meta_data_file(self, meta_data_file):
        metadata_path = self.dir_path / meta_data_file
        with open(metadata_path, 'r') as file:
            self.metadata = yaml.safe_load(file)
            file.close()

    def __read_config(self):
        config_path = self.dir_path / "datasets_config.yaml"
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
            global download_dir
            download_dir = Path(cfg['download_dir'])
            file.close()

    def __download_csv(self, url):
        filename = self.metadata['name'] + self.metadata['file_type']
        if os.path.isdir(download_dir):
            df = cudf.read_csv(url)
            try:
                df.to_csv(download_dir / filename, index=False)
            except RuntimeError:
                print("Error: cannot write files here")
            self.path = download_dir / filename
        else:
            raise RuntimeError("The directory " + str(download_dir.absolute())
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

        if self.__edgelist is None:
            full_path = download_dir / (self.metadata['name'] +
                                        self.metadata['file_type'])

            if not os.path.isfile(full_path):
                if fetch:
                    self.__download_csv(self.metadata['url'])
                else:
                    raise RuntimeError("The datafile does not exist. Try" +
                                       " get_edgelist(fetch=True) to" +
                                       " download the datafile")

            self.__edgelist = cudf.read_csv(full_path,
                                            delimiter=self.metadata['delim'],
                                            names=self.metadata['col_names'],
                                            dtype=self.metadata['col_types'])
            self.path = full_path

        return self.__edgelist

    def get_graph(self, fetch=False):
        """
        Return a Graph object.

        Parameters
        ----------
        fetch : Boolean (default=False)
            Automatically fetch for the dataset from the 'url' location within
            the YAML file.
        """
        if self.__edgelist is None:
            self.get_edgelist(fetch)

        self.__graph = cugraph.Graph(directed=self.metadata['is_directed'])
        self.__graph.from_cudf_edgelist(self.__edgelist, source='src',
                                        destination='dst')

        return self.__graph

    def get_path(self):
        """
            Returns the location of the stored dataset file
        """
        return self.path

    def view_config(self):
        return download_dir


def load_all(force=False):
    """
    Looks in `metadata` directory and fetches all datafiles from the the URLs
    provided in each YAML file.

    Parameters
        ----------
    """
    # FIXME: Remove prints
    if not os.path.isdir(download_dir):
        # print("Creating directory: " + str(download_dir.absolute()))
        os.mkdir(download_dir)

    meta_path = Path(__file__).parent.absolute() / "metadata"
    for file in os.listdir(meta_path):
        meta = None
        if file.endswith('.yaml'):
            with open(meta_path / file, 'r') as metafile:
                meta = yaml.safe_load(metafile)
                metafile.close()

            if 'url' in meta:
                filename = meta['name'] + meta['file_type']
                save_to = download_dir / filename
                if not os.path.isfile(save_to) or force:
                    # print("Downloading dataset from: " + meta['url'])
                    # print("  Saving file to " + str(save_to.absolute()))
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

    global download_dir
    if not os.path.isabs(cfgpath):
        download_dir = Path(cfgpath).absolute()
    else:
        download_dir = this_dir / cfgpath

    with open(download_dir, 'r') as file:
        cfg = yaml.safe_load(file)
        download_dir = Path(cfg['download_dir'])
        file.close()
