# 1. Import

import os
import time

from dask.distributed import Client

import dask_cudf
import dask_cugraph as dcg


# 2. Set the Number of GPU Devices and File Paths

number_of_devices = 2
scheduler_file_path = r"/home/seunghwak/cluster.json"
input_data_path = r"/home/seunghwak/TMP/Input-small/edges"


# 3. Define Utility Functions

def set_visible(i, n):
    all_devices = list(range(n))
    visible_devices = ",".join(map(str, all_devices[i:] + all_devices[:i]))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


# 4. Create a Client

print("Initializing.")

start_time = time.time()  # start timing from here

client = Client(scheduler_file="/home/seunghwak/cluster.json",
                direct_to_workers=True)

# 5. Map One Worker to One GPU

devices = list(range(number_of_devices))
device_workers = list(client.has_what().keys())
assert len(devices) == len(device_workers)

[client.submit(set_visible, device, len(devices), workers=[worker])
    for device, worker in zip(devices, device_workers)]

# 6. Read Input Data

print("Read Input Data.")

dgdf = dask_cudf.read_csv(input_data_path + r"/part-*",
                          delimiter='\t', names=['src', 'dst'],
                          dtype=['int32', 'int32'])
dgdf = client.persist(dgdf)

# 7. Sort Data

print("Sort Input Data.")

dgdf = dgdf.sort_values_binned(by='dst')
dgdf = client.persist(dgdf)

# 8. Compute PageRank

print("Compute PageRank.")

pagerank = dcg.mg_pagerank(dgdf)
print(pagerank)
res_df = pagerank.compute()
print(res_df)

# 9. Close the Client and Report Execution Time

print("Terminating.")

client.close()

end_time = time.time()
print((end_time - start_time), "seconds")
