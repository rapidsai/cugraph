import time
import random
from pathlib import Path

from gaas_client import GaasClient

client = GaasClient()

time.sleep(10)
n = int(random.random() * 1000)

#print(f"---> starting {n}", flush=True)

for i in range(1000000):
    extracted_gid = client.extract_subgraph(allow_multi_edges=False)
    #client.delete_graph(extracted_gid)
    #print(f"---> {n}: extracted {extracted_gid}", flush=True)

#print(f"---> done {n}", flush=True)
