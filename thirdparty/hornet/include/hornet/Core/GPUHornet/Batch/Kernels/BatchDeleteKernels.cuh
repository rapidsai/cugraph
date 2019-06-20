/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
namespace hornets_nest {
namespace gpu {

template<typename T>
__global__
void deleteSortedKernel() {

}

template<typename T>
__global__
void genererateDegreeTmpKernel() {

}

template<int BLOCK_SIZE, typename HornetDevice>
__global__
void locateEdges(HornetDevice              hornet,
                 const int*   __restrict__ d_prefixsum,
                 const int*   __restrict__ d_batch_offsets,//offset in batch
                 const vid_t* __restrict__ d_batch_unique_src,
                 const vid_t* __restrict__ d_batch_dst,
                 int                       batch_size,
                 bool*                     d_flags,
                 int*         __restrict__ d_locations) {

    const auto& lambda = [&] (int pos, degree_t offset) {
                    auto     vertex = hornet.vertex(d_batch_unique_src[pos]);
                    assert(offset < vertex.degree());
                    auto        dst = vertex.edge(offset).dst_id();
                    int start = d_batch_offsets[pos];
                    int end   = d_batch_offsets[pos + 1];
                    int found = xlib::lower_bound_left(
                            d_batch_dst + start,
                            end - start,
                            dst);
                    if ((found >= 0) && (dst == d_batch_dst[start + found])) {
                        d_locations[start + found] = offset;
                        d_flags[start + found] = true;
                    }


                };
    xlib::simpleBinarySearchLB<BLOCK_SIZE>(d_prefixsum, batch_size, nullptr, lambda);
}

// === overwriteDeletedEdges logic ===
// Adjacency list of a vertex :
// d0, d1, d2, d3, d4, d5, d6
// Locations to be deleted 1, 5, 6
//
// New degree = 4
// Work on d4, d5, d6
// Is destination still valid? If yes, scatter to invalidated position
// (d4, d5, d6) -> (1, 5, 6)
// In this case, only d4 is valid in the working set
// so, adj[1] = d4
//
// Updated Adjacency list :
// d0, d4, d2, d3, d4, d5, d6
//
// Subsequent function (fixInternalRepresentation) will reduce the degree
// so that the new adjacency list is
// d0, d4, d2, d3
template<int BLOCK_SIZE, typename HornetDevice>
__global__
void overwriteDeletedEdges(
        HornetDevice              hornet,
        const int*   __restrict__ d_batch_offsets,
        const int*   __restrict__ d_locations,//offset in batch
        const int*   __restrict__ d_counts,//number of deletions
        const vid_t* __restrict__ d_batch_unique_src,
        const vid_t* __restrict__ d_batch_dst,
        int                       batch_size,
        int*         __restrict__ d_counter) {

    const auto& lambda = [&] (int pos, degree_t offset) {
        auto     vertex = hornet.vertex(d_batch_unique_src[pos]);
        offset += vertex.degree() - d_counts[pos];
        assert(offset < vertex.degree());
        auto        dst = vertex.edge(offset).dst_id();
        int start = d_batch_offsets[pos];
        int end   = d_batch_offsets[pos + 1];
        int found = xlib::lower_bound_left(
                d_batch_dst + start,
                end - start,
                dst);
        if (found < 0 || (dst != d_batch_dst[start + found])) {//revisit, can we avoid if?
            int loc = atomicAdd(d_counter + pos, 1);
            auto deleted_dst_offset = d_locations[start + loc];
            vertex.neighbor_ptr()[deleted_dst_offset] = vertex.neighbor_ptr()[offset];
        }

    };
    xlib::simpleBinarySearchLB<BLOCK_SIZE>(d_batch_offsets, batch_size, nullptr, lambda);
}

} // namespace gpu
} // namespace hornets_nest
