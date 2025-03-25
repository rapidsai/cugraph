import pylibcugraph, cupy, numpy
srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5], dtype=numpy.int32)
dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4], dtype=numpy.int32)
weights = cupy.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1, 3.1,
                         4.1, 7.2, 3.2], dtype=numpy.float32)
start_vertices = cupy.asarray([2, 5]).astype(numpy.int32)
h_fan_out = numpy.array([2]).astype(numpy.int32)
resource_handle = pylibcugraph.ResourceHandle()
graph_props = pylibcugraph.GraphProperties(
    is_symmetric=False, is_multigraph=False)
G = pylibcugraph.SGGraph(
    resource_handle, graph_props, srcs, dsts, weight_array=weights,
    store_transposed=True, renumber=False, do_expensive_check=False)
sampling_results = pylibcugraph.homogeneous_uniform_neighbor_sample(
        resource_handle, G, start_vertices, None, h_fan_out, False, True)
