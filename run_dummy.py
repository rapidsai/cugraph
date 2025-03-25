import cudf
import cupy



#batch_ids =  [0]
#num_hops =  1
#label_hop_offsets =  [0 4]

num_hops = 1
batch_ids = cupy.array([0])
label_hop_offsets = cupy.array([0, 4])
hop_ids_r = cudf.Series(cupy.arange(num_hops))
print("hop_ids_r = \n", hop_ids_r)
hop_ids_r = cudf.concat([hop_ids_r] * len(batch_ids), ignore_index=True)

print("hop_ids_r = \n", hop_ids_r)

# generate the hop column



hop_ids_r = (
    cudf.Series(hop_ids_r, name="hop_id")
    .repeat(cupy.diff(label_hop_offsets))
    .reset_index(drop=True)
)

print("hop_ids_r = \n", hop_ids_r)







# *****************************