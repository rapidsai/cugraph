# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import warnings
import tarfile

import urllib.request

import cudf
from cugraph.datasets.dataset import (
    DefaultDownloadDir,
    default_download_dir,
)

# results_dir_path = utils.RAPIDS_DATASET_ROOT_DIR_PATH / "tests" / "resultsets"


class Resultset:
    """
    A Resultset Object, used to store golden results to easily run tests that
    need to access said results without the overhead of running an algorithm
    to get the results.

    Parameters
    ----------
    data_dictionary : dict
        The existing algorithm output, expected as a dictionary
    """

    def __init__(self, data_dictionary):
        self._data_dictionary = data_dictionary

    def get_cudf_dataframe(self):
        """
        Converts the existing algorithm output from a dictionary to
        a cudf.DataFrame before writing the DataFrame to output into a csv
        """
        return cudf.DataFrame(self._data_dictionary)


_resultsets = {}


def get_resultset(resultset_name, **kwargs):
    """
    Returns the golden results for a specific test.

    Parameters
    ----------
    resultset_name : String
        Name of the test's module (currently just 'traversal' is supported)

    kwargs :
        All distinct test details regarding the choice of algorithm, dataset,
        and graph
    """
    arg_dict = dict(kwargs)
    arg_dict["resultset_name"] = resultset_name
    # Example:
    # {'a': 1, 'z': 9, 'c': 5, 'b': 2} becomes 'a-1-b-2-c-5-z-9'
    resultset_key = "-".join(
        [
            str(val)
            for arg_dict_pair in sorted(arg_dict.items())
            for val in arg_dict_pair
        ]
    )
    uuid = _resultsets.get(resultset_key)
    if uuid is None:
        raise KeyError(f"results for {arg_dict} not found")

    results_dir_path = default_resultset_download_dir.path
    results_filename = results_dir_path / (uuid + ".csv")
    return cudf.read_csv(results_filename)


default_resultset_download_dir = DefaultDownloadDir(subdir="tests/resultsets")


def load_resultset(resultset_name, resultset_download_url):
    """
    Read a mapping file (<resultset_name>.csv) in the _results_dir and save the
    mappings between each unique set of args/identifiers to UUIDs to the
    _resultsets dictionary. If <resultset_name>.csv does not exist in
    _results_dir, use resultset_download_url to download a file to
    install/unpack/etc. to _results_dir first.
    """
    # curr_resultset_download_dir = get_resultset_download_dir()
    curr_resultset_download_dir = default_resultset_download_dir.path
    # curr_download_dir = path
    curr_download_dir = default_download_dir.path
    mapping_file_path = curr_resultset_download_dir / (resultset_name + "_mappings.csv")
    if not mapping_file_path.exists():
        # Downloads a tar gz from s3 bucket, then unpacks the results files
        compressed_file_dir = curr_download_dir / "tests"
        compressed_file_path = compressed_file_dir / "resultsets.tar.gz"
        if not curr_resultset_download_dir.exists():
            curr_resultset_download_dir.mkdir(parents=True, exist_ok=True)
        if not compressed_file_path.exists():
            urllib.request.urlretrieve(resultset_download_url, compressed_file_path)
        tar = tarfile.open(str(compressed_file_path), "r:gz")
        # TODO: pass filter="fully_trusted" once Python 3.12 is the minimum supported Python version
        #  ref: https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            tar.extractall(str(curr_resultset_download_dir))
        tar.close()

    # FIXME: This assumes separator is " ", but should this be configurable?
    sep = " "
    with open(mapping_file_path) as mapping_file:
        for line in mapping_file.readlines():
            if line.startswith("#"):
                continue

            (uuid, *row_args) = line.split(sep)
            if (len(row_args) % 2) != 0:
                raise ValueError(
                    f'bad row in {mapping_file_path}: "{line}", must '
                    "contain UUID followed by an even number of items"
                )
            row_keys = row_args[::2]
            row_vals = row_args[1::2]
            row_keys = " ".join(row_keys).split()
            row_vals = " ".join(row_vals).split()
            arg_dict = dict(zip(row_keys, row_vals))
            arg_dict["resultset_name"] = resultset_name
            # Create a unique string key for the _resultsets dict based on
            # sorted row_keys. Looking up results based on args will also have
            # to sort, but this will ensure results can looked up without
            # requiring maintaining a specific order. Example:
            # {'a': 1, 'z': 9, 'c': 5, 'b': 2} becomes 'a-1-b-2-c-5-z-9'
            resultset_key = "-".join(
                [
                    str(val)
                    for arg_dict_pair in sorted(arg_dict.items())
                    for val in arg_dict_pair
                ]
            )

            _resultsets[resultset_key] = uuid
