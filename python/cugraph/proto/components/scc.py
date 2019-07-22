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


import cudf
import cugraph
import numpy as np



#
# TRIM Process:
#   - removed single vertex componenets
#   - select vertex with highest out degree
#   - forwards BFS
#   - backward BFS
#   - compute intersection = components
#   - remove component
#   - repeat


def strong_connected_component(source, destination) :
    """
    Generate the strongly connected components (using the TRIM approach)

    Parameters
    ----------
    source : cudf.Seriers
	A cudf seriers that contains the source side of an edge list

    destination : cudf.Seriers
	A cudf seriers that contains the destination side of an edge list
     

    Returns
    -------
    df : cudf.DataFrame
      df['labels'][i] gives the label id of the i'th vertex

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)

   >>> components, single_components, count = 
    """
  
    max_value = np.iinfo(np.int32).max

    # create a copy of the data that can be manipulated
    coo = cudf.DataFrame()
    coo['src'] = source
    coo['dst'] = destination
    
    # create space for the answers
    single_components = cudf.DataFrame()
    components        = cudf.DataFrame()
    count             = 0
    
    G_fw = cugraph.Graph()
    G_bw = cugraph.Graph()

    G_fw.add_edge_list(coo['src'], coo['dst'])        
    G_bw.add_edge_list(coo['dst'], coo['src'])        

    # get a list of vertices and sort the list on out_degree
    d = G_fw.degrees()
    d = d.sort_values(by='out_degree', ascending=False) 
    
    bad = d.query('in_degree == 0 or out_degree == 0')
    
    if len(bad) : 
        single_components = _update_singletons(single_components, bad)
        d                 = _filter_list(d, bad)
    
    #----- Start processing -----
    while len(d) > 0 : 
     
        v = d['vertex'][0]        
 
        # compute the forward BFS
        bfs_fw = cugraph.bfs(G_fw, v)  
        bfs_fw = bfs_fw.query("distance != @max_value")
           
        # Now backwards
        bfs_bw = cugraph.bfs(G_bw, v)
        bfs_bw = bfs_bw.query("distance != @max_value")    

        # intersection
        common = bfs_fw.merge(bfs_bw, on='vertex', how='inner')

        if len(common) > 1 :     
            components = _update_components(components, common, count)
            d          = _filter_list(d, common)
            count      =  count + 1

            del common

        else :
            # v is an isolated vertex
            vdf = cudf.DataFrame()
            vdf['vertex'] = v
            single_components = _update_singletons(single_components, vdf)
            d                 = _filter_list(d, vdf)


    # loop until coo == 0

    return components, single_components, count






def _update_singletons(df, sv) :
    
    _d = cudf.DataFrame()
    _d['vertex'] = sv['vertex']
    _d['id']     = sv['vertex']
    
    if len(df) > 0 :
        _d = cudf.concat([df, _d])
    
    return _d




def _update_components(df, bfs_common, vert) :
  
    _d = cudf.DataFrame()
    _d['vertex'] = bfs_common['vertex'].copy()
    _d['id']     = vert    
               
    if len(df) > 0 :  
        
        _d = cudf.concat([df, _d])
            
    return _d




def _filter_list(vert_list, drop_list) :
    t = cudf.DataFrame()
    t['vertex'] = drop_list['vertex']
    t['d'] = 0

    df = vert_list.merge(t, on='vertex', how="left" )

    df['d'] = df['d'].fillna(1)
    df = df.query('d == 1')
    df.drop_column('d')    
    
    return df

