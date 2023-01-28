# Copyright (c) 2023, NVIDIA CORPORATION.
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

import argparse
import pandas
import numpy as np
import os
import ast

from cugraph.gnn import FeatureStore

def arg_to_list(s: str):
    mylist = ast.literal_eval(s)
    if type(mylist) is not list:
        raise TypeError("Input is not in list format")
    return mylist

def read_edge_type(input_dir, can_edge_type):
    df_edges = pandas.read_parquet(
        os.path.join(input_dir, '__'.join(can_edge_type))
    )
    return [
        df_edges.src.to_numpy(),
        df_edges.dst.to_numpy()
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputDir",
        type=str,
        default=None,
        help="Directory where the input graph is stored",
        required=True,
    )
    parser.add_argument(
        "--nodesWithFeatures",
        type=arg_to_list,
        default=None,
        help="List of node types that have features",
        required=True,
    )
    parser.add_argument(
        "--nodesWithLabels",
        type=arg_to_list,
        default=None,
        help="List of node types that have labels",
        required=True,
    )

    args = parser.parse_args()
    print('Reading From:', args.inputDir)

    can_edge_types = [
        ('author','affiliated_with', 'institution'),
        ('author','writes','paper'),
        ('paper','cites','paper')
    ]

    # Read Edges
    G = {
        can_edge_type: read_edge_type(args.inputDir, can_edge_type)
        for can_edge_type in can_edge_types
    }

    # Read features
    F = FeatureStore(backend='numpy')
    for node_type in args.nodesWithLabels:
        node_data = np.load(
            os.path.join(
                os.path.join(args.inputDir, 'paper'),
                'node-feat.npy'
            )
        )
        F.add_data(
            node_data,
            node_type,
            'x'
        )


if __name__ == '__main__':
    main()