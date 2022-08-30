import random
import time
from pathlib import Path

from gaas_client import GaasClient


_data_dir = (Path(__file__).parent)/"data"

edgelist_csv_data = {
    "karate": {"csv_file_name":
               (_data_dir/"karate.csv").absolute().as_posix(),
               "dtypes": ["int32", "int32", "float32"],
               "num_edges": 156,
               },
}

client = GaasClient()

test_data = edgelist_csv_data["karate"]
client.load_csv_as_edge_data(test_data["csv_file_name"],
                             dtypes=test_data["dtypes"],
                             vertex_col_names=["0", "1"],
                             type_name="")
time.sleep(10)
n = int(random.random() * 1000)

#print(f"---> starting {n}", flush=True)

for i in range(1000000):
    extracted_gid = client.extract_subgraph(allow_multi_edges=False)
    #client.delete_graph(extracted_gid)
    #print(f"---> {n}: extracted {extracted_gid}", flush=True)

#print(f"---> done {n}", flush=True)
