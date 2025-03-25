import pylibcugraph, cupy, numpy



from cugraph.testing import utils
import pylibcugraph, cupy, numpy
import cugraph


dataset_path="/home/nfs/jnke/remove_deprecated_api/cugraph_25.04/datasets/karate.csv"

df = utils.read_csv_file(dataset_path)

df = df.rename(columns={"0": "src", "1": "dst"})

print("df = \n", df)


G = cugraph.Graph(directed=True)

G.from_cudf_edgelist(
    df,
    source="src",
    destination="dst",
    edge_attr=None,
    renumber=True,
)

import cudf


df_result = cugraph.homogeneous_uniform_neighbor_sample(
    G,
    #start_list = cudf.Series([2, 5, 1], dtype="int32"),
    start_list = cudf.Series([0, 2], dtype="int32"),
    #starting_vertex_label_offsets= cudf.Series([0, 2], dtype="int32"),
    starting_vertex_label_offsets= [0, 2],
    #fanout_vals = [2]
    #start_list = cupy.asarray([2, 5, 1]).astype(numpy.int32),
    #starting_vertex_label_offsets = cupy.asarray([0, 2, 3]),
    #starting_vertex_label_offsets = cupy.asarray([0, 2]),
    fanout_vals = numpy.array([2, 1]).astype(numpy.int32),
    renumber = False,
    return_offsets = True,

    #prior_sources_behavior = "exclude",
    #compression = "CSR"
    )

if isinstance(df_result, tuple):
    #print("df = \n", df_result)
    df, offsets = df_result
    print("df = \n", df)
    print("offsets = \n", offsets)
    #print("batch_id = \n", batch_id)
else:
    print("df = \n", df_result)

#print("df = \n", df_result)
