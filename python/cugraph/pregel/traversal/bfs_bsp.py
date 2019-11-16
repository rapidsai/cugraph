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
import numpy as np
from collections import OrderedDict


def pregel_bfs(G, start) :
    """
    This function executes an unwieghted Breadth-First-Search (BFS) traversal to find the distances and predecessors 
    from a sepcified starting vertex in the graph. 
    
    Parameters
    ----------
     G : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        
    start
        The index of the graph vertex from which the traversal begins

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex
        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal
    
    Examples
    --------
    >>> data_df = cudf.read_csv('datasets/karate.csv', delimiter=' ', dtype=['int32', 'int32', 'float32'], header=None)
    >>> df = cugraph.pregel_bfs(data_df, 1)    
    
    """    

    ###  TBD
    
    
    

def bfs_df(df, start, src_col='src', dst_col='dst', copy_data=True) :
    """
    This function executes an unwieghted Breadth-First-Search (BFS) traversal to find the distances and predecessors 
    from a sepcified starting vertex in the graph. 
    
    Parameters
    ----------
    df : cudf.dataframe
        a dataframe containing the source and destination edge list

    start : Integer
        The index of the graph vertex from which the traversal begins
        
    src : string
        the source column name
        
    dst : string
        the destination column name    
        
    copy_data : Bool
        whether we can manipulate the dataframe or if a copy should be made



    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex
        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal
    
    Examples
    --------
    >>> data_df = cudf.read_csv('datasets/karate.csv', delimiter=' ', dtype=['int32', 'int32', 'float32'], header=None)
    >>> df = cugraph.pregel_bfs(data_df, 1, '0', '1')    
    
    """

    # extract the source and destination columns into a new dataframe that can be modified
    if copy_data : 
        coo_data = cudf.DataFrame()
        coo_data['src'] = df[src_col]
        coo_data['dst'] = df[dst_col]
    else :
        coo_data = df
        coo_data.rename(columns={src_col:'src', dst_col:'dst'}, inplace=True)
       
    
    # convert the "start" vertex into a series
    start_v = cudf.Series(start)
    ddf = _create_vertex_list(start_v)

    # create the answer DF
    answer = cudf.DataFrame()
    answer['vertex'] = start
    answer['distance'] = 0
    answer['predecessor'] = -1
    
    # init some variables
    distance = 0    
    done = False

    while not done :
        #---------------------------------      
        # update the distance and add it to the dataframe
        distance = distance + 1
        ddf['distance'] = distance

        #-----------------------------------
        # Removed all instances of the vertices in 'ddf' from 'dst' side of coo
        # we do not want to hop to a vertex that has already been seen 
        coo_data = coo_data.merge(ddf, on=['dst'], how='left') 
        coo_data = coo_data[coo_data.distance.isnull()]
        coo_data = coo_data.drop('distance')  

        # now update column names for finding source vertices
        ddf.rename(columns={'dst':'src'}, inplace=True)

        #---------------------------------      
        # merge the list of vertices and distances with the COO list
        hop_df = coo_data.merge(ddf, on=['src'], how='left')

        #---------------------------------  
        # get the data that was one hop out 
        one_hop = hop_df.query("distance == @distance")    

        #---------------------------------    
        # get all the edges that where not touched - also drop all reverse edges 
        coo_data = hop_df[hop_df.distance.isnull()]
        coo_data = coo_data.drop('distance')    

        #---------------------------------  
        # create a new dataframe of vertices to hop out from
        ddf = _create_vertex_list(one_hop['dst'])

        #---------------------------------      
        # update the answer  
        one_hop.rename(columns={'dst':'vertex', 'src':'predecessor'}, inplace=True)  

        # could contain a number of 
        aggsOut = OrderedDict()
        aggsOut['predecessor'] = 'min'   
        aggsOut['distance'] = 'min'   
 
        _a = one_hop.groupby(['vertex'], as_index=False).agg(aggsOut)      

        answer = cudf.concat([answer,_a])     
        answer = answer.groupby(['vertex'], as_index=False).agg(aggsOut)      

        if len(coo_data) == 0 :
            done = True    
    
        if not done and len(ddf) == 0:
            # remaining vertices are unreachable
            ddf.rename(columns={'dst':'vertex'}, inplace=True)
            ddf['predecessor'] = -1
            ddf['distance'] = np.iinfo(np.int32).max
            answer = cudf.concat([answer, ddf])  
            done = True
    
    # all done, return the answer
    return answer


# create a dataframe from a cudf.Series.
def _create_vertex_list(v_df) :
    _df = cudf.DataFrame()
    _df['dst'] = v_df   

    return _df  