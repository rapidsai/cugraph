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


def bfs_df(df, start, src_col='src', dst_col='dst', copy_data=True) :
    """
    This function executes an unwieghted Breadth-First-Search (BFS) traversal to find the distances and predecessors 
    from a sepcified starting vertex in the graph. 
    
    Parameters
    ----------
    df : cudf.dataframe
        a dataframe containing the source and destination edge list

    start : same type as 'src' and 'dst' 
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
        coo_data = df[[src_col, dst_col]]
    else :
        coo_data = df
    
    coo_data.rename(columns={src_col:'src', dst_col:'dst'}, inplace=True)
       
    
    # convert the "start" vertex into a series
    frontier = cudf.Series(start).to_frame('dst')
    #frontier = start_v.to_frame('dst')
    
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
        frontier['distance'] = distance

        #-----------------------------------
        # Removed all instances of the frontier vertices from 'dst' side of coo_data
        # we do not want to hop to a vertex that has already been seen 
        coo_data = coo_data.merge(frontier, on=['dst'], how='left') 
        coo_data = coo_data[coo_data.distance.isnull()]
        coo_data.drop_column('distance')  

        # now update column names for finding source vertices
        frontier.rename(columns={'dst':'src'}, inplace=True)

        #---------------------------------      
        # merge the list of vertices and distances with the COO list
        # there are two sets of results that we get from the "hop_df" nerge
        # (A) the set of edges that start with a vertice in the frontier set 
        #     - this goes into the answer set
        #     - this also forms the next frontier set
        # (B) the set of edges that did not start with a frontier vertex - this form the new set of coo_data        
        hop_df = coo_data.merge(frontier, on=['src'], how='left')
        
        #---------------------------------  
        # (A) get the data where the 'src' was in the frontier list (e.g. there is a distance score) 
        # create a new dataframe of vertices to hop out from (the 'dst')
        one_hop = hop_df.query("distance == @distance")    
        frontier = one_hop['dst'].to_frame('dst')

        #---------------------------------    
        # (B) get all the edges that where not touched 
        coo_data = hop_df[hop_df.distance.isnull()]
        coo_data.drop_column('distance')    

        #---------------------------------      
        # update the answer  
        one_hop.rename(columns={'dst':'vertex', 'src':'predecessor'}, inplace=True)  

        answer = cudf.concat([answer,one_hop])     

        if len(coo_data) == 0 :
            done = True    
    
        if not done and len(frontier) == 0:
            # remaining vertices are unreachable
            frontier.rename(columns={'dst':'vertex'}, inplace=True)
            frontier['predecessor'] = -1
            frontier['distance'] = np.iinfo(np.int32).max
            answer = cudf.concat([answer, frontier])  
            done = True
    
    # all done, return the answer
    return answer
 
 