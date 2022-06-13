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
import re
import os
import csv
import pdb

class Dataset:
    def __init__(self, meta_data_file_name):
        self.__meta_data_file_name = meta_data_file_name
        self.__read_meta_data_file(self.__meta_data_file_name)
        self.__edgelist = None
        self.__graph = None

    def __read_meta_data_file(self, meta_data_file):
        dir_path = "python/cugraph/cugraph/experimental/datasets/"

        with open(dir_path + meta_data_file, 'r') as file:
            self.metadata = yaml.safe_load(file)

    # figure out throwing errors if fetch=False and file doesn't exist...
    def get_edgelist(self, fetch=False):
        """
            Return an Edgelist
        """
        if self.__edgelist is None:
            if not os.path.isfile(self.metadata['path']):
                if fetch:
                    self.__download_csv(self.metadata['url'], "python/")
                else:
                    print("The datafile does not exist. Try running with fetch=True to download the datafile")
                    return
            
            self.__edgelist = cudf.read_csv(self.metadata['path'], delimiter='\t', names=['src', 'dst'], dtype=['int32', 'int32'])

        return self.__edgelist

    def get_graph(self, fetch=False):
        """
            Return a Graph object.
        """
        if self.__edgelist is None:
            self.get_edgelist(fetch)
        
        self.__graph = cugraph.from_cudf_edgelist(self.__edgelist, source='src', destination='dst')

        return self.__graph

    def __download_csv(self, url, default_path):
        # fetch from metadata.url
        #pdb.set_trace()
        filename = self.metadata['url'].split('/')[-1]
        df = cudf.read_csv(self.metadata['url'])
        df.to_csv(default_path+filename, index=False)
        self.metadata['path'] = default_path + filename


# SMALL DATASETS
karate = Dataset("metadata/karate.yaml")
dolphins = Dataset("metadata/dolphins.yaml")

# MEDIUM DATASETS

# LARGE DATASETS

# GROUPS OF DATASETS
SMALL_DATASETS = [karate, dolphins]
