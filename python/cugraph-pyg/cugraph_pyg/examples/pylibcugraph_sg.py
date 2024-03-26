# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This example shows how to use cuGraph and pylibcuGraph to run a
# single-GPU workflow.  Most users of the GNN packages will not interact
# with cuGraph directly.  This example is intented for users who want
# to extend cuGraph within a PyTorch workflow.

import pandas
import numpy as np

import cudf

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties, degrees

from ogb.nodeproppred import NodePropPredDataset


def calc_degree(edgelist):
    src = cudf.Series(edgelist[0])
    dst = cudf.Series(edgelist[1])

    seeds = cudf.Series(np.arange(256))

    print("constructing graph")
    G = SGGraph(
        ResourceHandle(),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        src,
        dst,
    )
    print("graph constructed")

    print("calculating degrees")
    vertices, in_deg, out_deg = degrees(
        ResourceHandle(), G, seeds, do_expensive_check=False
    )
    print("degrees calculated")

    print("constructing dataframe")
    df = pandas.DataFrame(
        {"v": vertices.get(), "in": in_deg.get(), "out": out_deg.get()}
    )
    print(df)

    print("done")


def main():
    dataset = NodePropPredDataset("ogbn-products")
    el = dataset[0][0]["edge_index"].astype("int64")
    calc_degree(el)


if __name__ == "__main__":
    main()
