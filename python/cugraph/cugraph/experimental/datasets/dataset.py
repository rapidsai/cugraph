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


class MetaData:
    def __init__(self, meta_data_file_name):
        self.description =
        self.
    pass
​
class Dataset:
    def __init__(self, meta_data_file_name):
        self.__meta_data_file_name = meta_data_file_name
        self.__edgelist = None
        self.__graph = None
​
    def __getattr__(self, attr):
        """
        lazily read meta-data
        """
        if attr == "metadata":
            self.__read_meta_data_file(self.__meta_data_file_name)
​
    @property
    def edgelist(self):
        if self.__edgelist is None:
            self.__edgelist = cudf.read_csv(self.metadata.csv_file_name, ...)
        return self.__edgelist
​
    @property
    def Graph(self):
        if self.__graph is None:
            self.__graph = cugraph.from_cudf_edgelist(self.__edgelist, ...)
        return self.__graph
​
    def __read_meta_data_file(self, meta_data_file):
        self.metadata = MetaData()
​
        self.metadata.description = ...
        self.metadata.url = ...

import cugraph.experimental.datasets as datasets

karate = datasets.karate # Dataset object, no data loaded
# karate.edgelist # load data into cudf edgelist
G = karate.Graph # load data into cugraph Graph