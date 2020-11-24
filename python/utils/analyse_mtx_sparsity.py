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

# Input:    <matrix.mtx>

# Output:   <mmFile,rows, cols, nnz, sparcity (%), empty rows (%),
#               sparsity the largest row (%),
#           sparsity at Q1 (%), sparsity at med (%), sparsity at Q3 (%),
#               Gini coeff>
#           <mmFile>_row_lengths_histogram.png (please comment plt.*
#               at the end of the script if not needed)

import numpy as np
import sys
from scipy.io import mmread
import scipy.sparse
import networkx as nx
import matplotlib.pyplot as plt


def gini(v):
    # zero denotes total equality between rows,
    # and one denote the dominance of a single row.
    # v = np.sort(v) #values must be sorted
    index = np.arange(1, v.shape[0] + 1)  # index per v element
    n = v.shape[0]
    return (np.sum((2 * index - n - 1) * v)) / (n * np.sum(v))


def count_consecutive(v):
    # accumulate each time v[i] = v[i-1]+1
    return np.sum((v[1:] - v[:-1]) == 1)


def consecutive_entries_per_row(M):
    # count the number of consecutive column indicies
    # for each row of a saprse CSR matrix sparse CSR.
    # not to be mixed with the longest sequence or the number of sequences
    v = [0] * M.shape[0]
    for i in range(M.shape[0]):
        v[i] = count_consecutive(M.indices[M.indptr[i]:M.indptr[i + 1]])
    return np.array(v)


# Command line arguments
argc = len(sys.argv)
if argc <= 1:
    print("Error: usage is : python analyse_mtx_sparsity.py matrix.mtx")
    sys.exit()
mmFile = sys.argv[1]

# Read
M_in = mmread(mmFile)
if M_in is None:
    raise TypeError("Could not read the input")
    M = scipy.sparse.csr_matrix(M_in)

if not M.has_sorted_indices:
    M.sort_indices()

# M = M.transpose()
M.sort_indices()

if M is None:
    raise TypeError("Could not convert to csr")

# General properties
row = M.shape[0]
col = M.shape[1]
nnz = M.nnz
real_nnz = M.count_nonzero()
nnz_per_row = M.getnnz(1)

# Distribution info
nnz_per_row.sort()
row_max = nnz_per_row.max()
quartile1 = nnz_per_row[round(row / 4)]
median = nnz_per_row[round(row / 2)]
quartile3 = nnz_per_row[round(3 * (row / 4))]
empty_rows = row - np.count_nonzero(nnz_per_row)
gini_coef = gini(nnz_per_row)
G = nx.from_scipy_sparse_matrix(M)
print(nx.number_connected_components(G))
# Extras:
# row_min = nnz_per_row.min()
# cepr = consecutive_entries_per_row(M)
# pairs = np.sum(cepr) # consecutive elements (pairs)
# max_pairs = cepr.max()

# print (CSV)
print(
    str(mmFile)
    + ","
    + str(row)
    + ","
    + str(col)
    + ","
    + str(nnz)
    + ","
    + str(round((1.0 - (nnz / (row * col))) * 100.0, 2))
    + ","
    + str(round((empty_rows / row) * 100.0, 2))
    + ","
    + str(round((1.0 - row_max / col) * 100.0, 2))
    + ","
    + str(round((1.0 - quartile1 / col) * 100.0, 2))
    + ","
    + str(round((1.0 - median / col) * 100.0, 2))
    + ","
    + str(round((1.0 - quartile3 / col) * 100.0, 2))
    + ","
    + str(round(gini_coef, 2))
)
# Extras:
#       str(round(((2*pairs)/nnz)*100,2))   )
#       str(round(nnz/row,2)) +','+
#       str(real_nnz) +','+
#       str(empty_rows)+','+
#       str(row_min) +','+
#       str(row_max) +','+
#       str(quartile1) +','+
#       str(median) +','+
#       str(quartile3) +','+
#       str(max_pairs) +','+
#       str(round((1.0-(real_nnz/(row*col)))* 100.0,2)) +','+
#       str(round((1.0-row_min/col)*100.0,2)) +','+

# historgam
plt.xlabel("Row lengths")
plt.ylabel("Occurences")
plt.hist(nnz_per_row, log=True)
plt.savefig(str(mmFile) + "_transposed_row_lengths_histogram.png")
plt.clf()
