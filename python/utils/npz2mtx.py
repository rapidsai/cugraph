# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
import os
import time
import scipy.io
import scipy.sparse
import argparse


parser = argparse.ArgumentParser(
    description="Convert the sparsity pattern \
                                 of a NPZ file into a MatrixMarket file. \
                                 Each directed edge is explicitely stored, \
                                 edges are unsorted, IDs are 0-based."
)
parser.add_argument(
    "file", type=argparse.FileType(), help="Path to the MatrixMarket file"
)
parser.add_argument(
    "--symmetry",
    type=str,
    default="general",
    choices=["general", "symmetric"],
    help="Pattern, either general or symmetric",
)
args = parser.parse_args()

# Read
print("Reading " + str(args.file.name) + "...")
t1 = time.time()
M = scipy.sparse.load_npz(args.file.name).tocoo()
read_time = time.time() - t1
print("Time (s) : " + str(round(read_time, 3)))

print("V =" + str(M.shape[0]) + ", E = " + str(M.nnz))

# Write
print(
    "Writing mtx file: "
    + os.path.splitext(os.path.basename(args.file.name))[0]
    + ".csv ..."
)
t1 = time.time()
scipy.io.mmwrite(
    os.path.splitext(os.path.basename(args.file.name))[0] + ".mtx",
    M,
    symmetry=args.symmetry,
)
write_time = time.time() - t1
print("Time (s) : " + str(round(write_time, 3)))
