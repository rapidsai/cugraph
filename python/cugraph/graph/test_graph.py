# Copyright (c) 2019, NVIDIA CORPORATION.
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
import pytest
import numpy as np
from scipy.io import mmread

def ReadMtxFile(mmFile):
    print('Reading '+ str(mmFile) + '...')
    return mmread(mmFile).asfptype()

def compare_series(series_1, series_2):
    if (len(series_1) != len(series_2)):
        print("Series do not match in length")
        return 0
    for i in range(len(series_1)):
        if(series_1[i] != series_2[i]):
            print("Series[" + str(i) + "] does not match, " + str(series_1[i]) + ", " + str(series_2[i]))
            return 0
    return 1

def compareOffsets(cu, np):
    if not (len(cu) <= len(np)):
        print("Mismatched length: " + str(len(cu)) + " != " + str(len(np)))
        return False
    for i in range(len(cu)):
        if cu[i] != np[i]:
            print("Series[" + str(i) + "]: " + str(cu[i]) + " != " + str(np[i]))
            return False
    return True

datasets = ['/datasets/networks/karate.mtx', 
            '/datasets/networks/dolphins.mtx', 
            '/datasets/networks/netscience.mtx']

@pytest.mark.parametrize('graph_file', datasets)
def test_add_edge_list_to_adj_list(graph_file):

    M = ReadMtxFile(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    M = M.tocsr()
    if M is None :  
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')
	
    offsets_exp = M.indptr
    indices_exp = M.indices

    # cugraph add_egde_list to_adj_list call
    G = cugraph.Graph()
    G.add_edge_list(sources,destinations, None)
    offsets_cu, indices_cu = G.view_adj_list()
    assert compareOffsets(offsets_cu, offsets_exp)
    assert compare_series(indices_cu, indices_exp)

@pytest.mark.parametrize('graph_file', datasets)
def test_add_adj_list_to_edge_list(graph_file):
    M = ReadMtxFile(graph_file)
    M = M.tocsr()
    if M is None :  
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')
            
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)

    M = M.tocoo()
    sources_exp = cudf.Series(M.row)
    destinations_exp = cudf.Series(M.col)

    # cugraph add_adj_list to_edge_list call
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    sources, destinations = G.view_edge_list()
    sources_cu = np.array(sources)
    destinations_cu = np.array(destinations)
    assert compare_series(sources_cu, sources_exp)
    assert compare_series(destinations_cu, destinations_exp)
   
@pytest.mark.parametrize('graph_file', datasets)
def test_transpose_from_adj_list(graph_file): 
    M = ReadMtxFile(graph_file)
    M = M.tocsr()
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    G.add_transpose()
    Mt = M.transpose().tocsr()
    toff, tind = G.view_transpose_adj_list()
    assert compare_series(Mt.indices, tind)
    assert compareOffsets(toff, Mt.indptr)
    
@pytest.mark.parametrize('graph_file', datasets)
def test_view_edge_list_from_adj_list(graph_file):
    M = ReadMtxFile(graph_file)
    M = M.tocsr()
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    G = cugraph.Graph()
    G.add_adj_list(offsets, indices, None)
    src2, dst2 = G.view_edge_list()
    M = M.tocoo()
    src1 = M.row
    dst1 = M.col
    assert compare_series(src1, src2)
    assert compare_series(dst1, dst2)
       
@pytest.mark.parametrize('graph_file', datasets)
def test_delete_edge_list_delete_adj_list(graph_file):
    M = ReadMtxFile(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    M = M.tocsr()
    if M is None :  
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')
        
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)

    # cugraph delete_adj_list delete_edge_list call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    G.delete_edge_list()
    with pytest.raises(cudf.bindings.GDFError.GDFError) as excinfo:
        G.view_adj_list()
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'

    G.add_adj_list(offsets, indices, None)
    G.delete_adj_list()
    with pytest.raises(cudf.bindings.GDFError.GDFError) as excinfo:
        G.view_edge_list()
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'


