import cugraph
import cudf
import random
from cugraph.tests import utils
"""
gdf = cudf.read_csv('./datasets/karate.csv', delimiter=' ',
                  dtype=['int32', 'int32', 'float32'], header=None)
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='0', destination='1', edge_attr='2')
df = cudf.DataFrame()
df["s_v_1"] = [0, 4]
df["s_v_2"] = [3, 9]
df["s_v_1"] = df["s_v_2"] + 1000  
print(df)
#rw, offset = cugraph.random_walks(G, df, 3)
#print(rw, "\n\n")
#print(offset, "\n\n")
#print(df)
"""


df_G = utils.read_csv_file('./datasets/karate.csv')
df_G.rename(columns={"0": "src", "1": "dst", "2": "weight"}, inplace=True)
df_G['src_0'] = df_G['src'] + 1000
df_G['dst_0'] = df_G['dst'] + 1000
#df_G['weight_0'] = 1.0
#print(df_G)
G = cugraph.Graph()
G.from_cudf_edgelist(df_G, source=['src', 'src_0'], destination=['dst', 'dst_0'])
k = random.randint(1, 10)
start_vertices = random.sample(G.nodes().to_array().tolist(), k)
seeds = cudf.DataFrame()
seeds['v'] = start_vertices
seeds['v_0'] = seeds['v'] + 1000
#df = cudf.Series(df_G['src'])
#print(df)
#Series = G.lookup_internal_vertex_id(df_G['src'])
"""
df_1 = G.add_internal_vertex_id(df_G, 'id', ['src', 'src_0'])
"""
#df_1 = G.add_internal_vertex_id(df_G, 'id_2', 'dst')


#print(df_G)
#df, offsets = cugraph.random_walks(G, 5, 3)

start_vertices =cudf.DataFrame()
start_vertices['v1']=[1, 4, 5]
start_vertices['v2']=start_vertices['v1']+1000
print(start_vertices)
start_vertices = G.lookup_internal_vertex_id(start_vertices['v1'])

