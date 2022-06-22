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
# import requests
# import re
import os
# import csv
# import pdb
from pathlib import Path


this_dir = Path(os.getenv("this_dir", "cugraph/cugraph/experimental/datasets"))
datasets_dir = this_dir.parent / "datasets"


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
        self.download_dir = this_dir.parent.parent / "datasets"
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
            self.download_dir = cfg['download_dir']
            file.close()

    def __download_csv(self, url, default_path):
        filename = url.split('/')[-1]
        # Could also be
        # filename = self.metadata['name'] + '.' + metadata['file_type']
        df = cudf.read_csv(url)
        df.to_csv(default_path + filename, index=False)
        self.path = default_path + filename

    def get_edgelist(self, fetch=False):
        """
        Return an Edgelist

        Parameters
        ----------
        fetch : Boolean (default=False)
            Automatically fetch for the dataset from the 'url' location within
            the YAML file.
        """
        # breakpoint()
        if self.__edgelist is None:
            full_path = self.download_dir + self.metadata['name'] \
                            + self.metadata['file_type']
            if not os.path.isfile(full_path):
                if fetch:
                    self.__download_csv(self.metadata['url'],
                                        self.download_dir)
                else:
                    raise RuntimeError("The datafile does not exist. Try \
                                        get_edgelist(fetch=True) to download \
                                        the datafile")

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

    def path(self):
        """
            Print the location of the stored dataset file
        """
        print(self.path)


def load_all(default_path="datasets/", force=False):
    """
    Looks in `metadata` directory and fetches all datafiles from the web.
    """
    meta_path = "python/cugraph/cugraph/experimental/datasets/metadata/"
    for file in os.listdir(meta_path):
        meta = None
        if file.endswith('.yaml'):
            with open(meta_path + file, 'r') as metafile:
                meta = yaml.safe_load(metafile)
                metafile.close()

            if 'url' in meta:
                # filename = meta['url'].split('/')[-1]
                filename = meta['name'] + meta['file_type']
                if not os.path.isfile(default_path + filename) or force:
                    print("Downloading dataset from: " + meta['url'])
                    df = cudf.read_csv(meta['url'])
                    df.to_csv(default_path + filename, index=False)
