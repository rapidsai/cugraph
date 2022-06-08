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
import requests
import os
import pdb

class MetaData:
    def __init__(self, filename):
        # Open JSON file from filepath
        pdb.set_trace()
        dir_path = "python/cugraph/cugraph/experimental/datasets/"
        file = open(dir_path + filename)

        # Convert to Python Dict
        self.meta = yaml.load(file, Loader=yaml.FullLoader)


class Dataset:
    def __init__(self, meta_data_file_name):
        self.__meta_data_file_name = meta_data_file_name    
        self.__edgelist = None
        self.__graph = None

    # FIXME: metadata reading should not be lazy
    def __getattr__(self, attr):
        """
        lazily read meta-data
        """
        if attr == "metadata":
            self.__read_meta_data_file(self.__meta_data_file_name)

    def __read_meta_data_file(self, meta_data_file):
        # MetaData obj reads in JSON
        self.metadata = MetaData(meta_data_file)

    def get_edgelist(self, fetch=False):
        if self.__edgelist is None:
            if fetch:
                # if file exists:
                    # pass
                # else:
                    # call download_csv()
                pass
            else:
                # ... do stuff
                pass
            self.__edgelist = cudf.read_csv(self.metadata.csv_file_name, ...)
        return self.__edgelist
+
    def get_graph(self, fetch=False):
        if self.__graph is None:
            self.__graph = cugraph.from_cudf_edgelist(self.get_edgelist(...), ...)
        return self.__graph

    # def download_csv():
        # fetch from metadata.url
        # metadata.csv = filename


# SMALL DATASETS
karate = Dataset("metadata/karate.yaml")
dolphins = Dataset("metadata/dolphins.yaml")

# MEDIUM DATASETS

# LARGE DATASETS

# GROUPS OF DATASETS
SMALL_DATASETS = [karate, dolphins]
