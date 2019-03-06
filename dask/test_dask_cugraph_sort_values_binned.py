# 1. Import

import os

from dask.distributed import Client

import dask_cudf


# 2. Define Utility Functions

def set_visible(i, n):
    all_devices = list(range(n))
    visible_devices = ",".join(map(str, all_devices[i:] + all_devices[:i]))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


# 3. Create a Client

print("Initializing.")

client = Client(scheduler_file="/home/seunghwak/cluster.json",
                direct_to_workers=True)

# 4. Map One Worker to One GPU

devices = [0, 1]
device_workers = list(client.has_what().keys())
assert len(devices) == len(device_workers)

[client.submit(set_visible, device, len(devices), workers=[worker])
    for device, worker in zip(devices, device_workers)]

# 5. Read Input Data

print("Read Input Data.")

dgdf = dask_cudf.read_csv("/home/seunghwak/TMP/Input-small/edges/part-*",
                          delimiter='\t', names=['src', 'dst'],
                          dtype=['int32', 'int32'])
dgdf = client.persist(dgdf)

# 6. Sort Data

print("Sort Input Data.")

dgdf = dgdf.sort_values_binned(by='dst')
dgdf = client.persist(dgdf)

# 7. Validate Sorting Output

print("Validate Sorted Data.")

prev_src = -1
prev_dst = -1
for p in range(dgdf.npartitions):
    print("Validating partition ", p, "/", dgdf.npartitions)
    gdf = dgdf.get_partition(p).compute()
    pdf = gdf.to_pandas()
    for r in range(len(pdf)):
        row = pdf.loc[r]
        cur_src = row[1]
        cur_dst = row[2]
        if cur_dst < prev_dst:
            print("Validation error (primary): p=", p, " r=", r, " prev=(",
                  prev_src, ",", prev_dst, ") cur=(", cur_src, ",", cur_dst,
                  ")")
        assert cur_dst >= prev_dst
        # if cur_dst == prev_dst:
        #     if cur_src < prev_src:
        #         print("Validation error (secondary): p=", p, " r=", r,
        #               " prev=(", prev_src, ",", prev_dst, ") cur=(", cur_src,
        #               ",", cur_dst, ")")
        if r == 0:
            if cur_dst == prev_dst:
                print("Validation error (binning): p=", p, " r=", r,
                      " prev=(", prev_src, ",", prev_dst, ") cur=(", cur_src,
                      ",", cur_dst, ")")
        prev_src = cur_src
        prev_dst = cur_dst

# 8. Close the Client

print("Terminating.")

client.close()
