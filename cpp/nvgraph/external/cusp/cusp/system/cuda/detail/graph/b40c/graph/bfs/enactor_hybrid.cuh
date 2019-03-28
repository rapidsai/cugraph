/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Hybrid BFS enactor
 ******************************************************************************/

#pragma once

#include "../../util/kernel_runtime_stats.cuh"

#include "../../graph/bfs/enactor_base.cuh"
#include "../../graph/bfs/problem_type.cuh"

#include "../../graph/bfs/contract_expand_atomic/kernel.cuh"
#include "../../graph/bfs/contract_expand_atomic/kernel_policy.cuh"
#include "../../graph/bfs/two_phase/expand_atomic/kernel.cuh"
#include "../../graph/bfs/two_phase/expand_atomic/kernel_policy.cuh"
#include "../../graph/bfs/two_phase/filter_atomic/kernel.cuh"
#include "../../graph/bfs/two_phase/filter_atomic/kernel_policy.cuh"
#include "../../graph/bfs/two_phase/contract_atomic/kernel.cuh"
#include "../../graph/bfs/two_phase/contract_atomic/kernel_policy.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Hybrid BFS enactor.
 *
 * Combines functionality of contract-expand and two-phase enactors,
 * running contract-expand (only global edge frontier) for small-sized
 * BFS iterations and two-phase (global edge and vertex frontiers) for
 * large-sized BFS iterations.
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorHybrid : public EnactorBase
{

    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------

protected:

    /**
     * Mechanism for implementing software global barriers from within
     * a fused-iteration kernel invocation
     */
    util::GlobalBarrierLifetime global_barrier;

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime 	fused_kernel_stats;
    util::KernelRuntimeStatsLifetime 	expand_kernel_stats;
    util::KernelRuntimeStatsLifetime 	filter_kernel_stats;
    util::KernelRuntimeStatsLifetime 	contract_kernel_stats;
    unsigned long long 					total_runtimes;			// Total time "worked" by each cta
    unsigned long long 					total_lifetimes;		// Total time elapsed by each cta
    unsigned long long 					total_queued;

    /**
     * Throttle state.  We want the host to have an additional BFS h_iteration
     * of kernel launches queued up for for pipeline efficiency (particularly on
     * Windows), so we keep a pinned, mapped word that the traversal kernels will
     * signal when done.
     */
    volatile int 		*done;
    int 				*d_done;
    cudaEvent_t			throttle_event;

    /**
     * Iteration output (from fused-iterations)
     */
    long long 			*d_iteration;
    long long 			h_iteration;


    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

protected:

    /**
     * Prepare enactor for search.  Must be called prior to each search.
     */
    template <typename CsrProblem>
    cudaError_t Setup(
        CsrProblem &csr_problem,
        int fused_grid_size,
        int expand_grid_size,
        int filter_grid_size,
        int contract_grid_size)
    {
        typedef typename CsrProblem::SizeT 			SizeT;
        typedef typename CsrProblem::VertexId 		VertexId;
        typedef typename CsrProblem::VisitedMask 	VisitedMask;

        cudaError_t retval = cudaSuccess;
        do {

            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::B40CPerror<0>(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
                                              "EnactorHybrid cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::B40CPerror<0>(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
                                              "EnactorHybrid cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::B40CPerror<0>(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                                              "EnactorHybrid cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;

                // Allocate gpu memory for d_iteration
                if (retval = util::B40CPerror<0>(cudaMalloc((void**) &d_iteration, sizeof(long long)),
                                              "EnactorHybrid cudaMalloc d_iteration failed", __FILE__, __LINE__)) break;
            }

            // Make sure our runtime stats are good
            if (retval = fused_kernel_stats.Setup(fused_grid_size)) break;
            if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
            if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;
            if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;

            // Make sure barriers are initialized
            if (retval = global_barrier.Setup(fused_grid_size)) break;

            // Reset statistics
            done[0] 			= -1;
            total_runtimes 		= 0;
            total_lifetimes 	= 0;
            total_queued 		= 0;

            // Single-gpu graph slice
            typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

            // Bind bitmask texture
            int bytes = (graph_slice->nodes + 8 - 1) / 8;
            cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<VisitedMask>();
            if (retval = util::B40CPerror<0>(cudaBindTexture(
                                              0,
                                              two_phase::contract_atomic::BitmaskTex<VisitedMask>::ref,
                                              graph_slice->d_visited_mask,
                                              bitmask_desc,
                                              bytes),
                                          "EnactorHybrid cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind row-offsets texture
            cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::B40CPerror<0>(cudaBindTexture(
                                              0,
                                              two_phase::expand_atomic::RowOffsetTex<SizeT>::ref,
                                              graph_slice->d_row_offsets,
                                              row_offsets_desc,
                                              (graph_slice->nodes + 1) * sizeof(SizeT)),
                                          "EnactorHybrid cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind bitmask texture
            if (retval = util::B40CPerror<0>(cudaBindTexture(
                                              0,
                                              contract_expand_atomic::BitmaskTex<VisitedMask>::ref,
                                              graph_slice->d_visited_mask,
                                              bitmask_desc,
                                              bytes),
                                          "EnactorHybrid cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

            // Bind row-offsets texture
            if (retval = util::B40CPerror<0>(cudaBindTexture(
                                              0,
                                              contract_expand_atomic::RowOffsetTex<SizeT>::ref,
                                              graph_slice->d_row_offsets,
                                              row_offsets_desc,
                                              (graph_slice->nodes + 1) * sizeof(SizeT)),
                                          "EnactorHybrid cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;


        } while (0);

        return retval;
    }

public:

    /**
     * Constructor
     */
    EnactorHybrid(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        d_iteration(NULL),
        h_iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}


    /**
     * Destructor
     */
    virtual ~EnactorHybrid()
    {
        if (done) {
            util::B40CPerror<0>(cudaFreeHost((void *) done), "EnactorHybrid cudaFreeHost done failed", __FILE__, __LINE__);
            util::B40CPerror<0>(cudaEventDestroy(throttle_event), "EnactorHybrid cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
        if (d_iteration) {
            util::B40CPerror<0>(cudaFree((void *) d_iteration), "EnactorHybrid cudaFree d_iteration failed", __FILE__, __LINE__);
        }
    }


    /**
     * Obtain statistics about the last BFS search enacted
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = h_iteration - 1;

        avg_duty = (total_lifetimes > 0) ?
                   double(total_runtimes) / total_lifetimes :
                   0.0;
    }


    /**
     * Enacts a breadth-first-search on the specified graph problem.
     *
     * @return cudaSuccess on success, error enumeration otherwise
     */
    template <
    typename OnePhasePolicy,
             typename ExpandPolicy,
             typename FilterPolicy,
             typename ContractPolicy,
             typename CsrProblem>
    cudaError_t EnactSearch(
        CsrProblem 						&csr_problem,
        typename CsrProblem::VertexId 	src,
        int 							max_grid_size = 0)
    {
        typedef typename CsrProblem::SizeT 			SizeT;
        typedef typename CsrProblem::VertexId 		VertexId;
        typedef typename CsrProblem::VisitedMask 	VisitedMask;

        cudaError_t retval = cudaSuccess;

        do {

            // Determine grid size(s)
            int fused_min_occupancy 	= OnePhasePolicy::CTA_OCCUPANCY;
            int fused_grid_size 		= MaxGridSize(fused_min_occupancy, max_grid_size);

            int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
            int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);

            int filter_occupancy			= FilterPolicy::CTA_OCCUPANCY;
            int filter_grid_size 			= MaxGridSize(filter_occupancy, max_grid_size);

            int contract_min_occupancy		= ContractPolicy::CTA_OCCUPANCY;
            int contract_grid_size 			= MaxGridSize(contract_min_occupancy, max_grid_size);

            if (DEBUG) {
                printf("BFS fused min occupancy %d, level-grid size %d\n",
                       fused_min_occupancy, fused_grid_size);
                printf("BFS expand min occupancy %d, level-grid size %d\n",
                       expand_min_occupancy, expand_grid_size);
                printf("BFS filter occupancy %d, level-grid size %d\n",
                       filter_occupancy, filter_grid_size);
                printf("BFS contract min occupancy %d, level-grid size %d\n",
                       contract_min_occupancy, contract_grid_size);

                printf("Iteration, Filter queue, Contraction queue, Expansion queue\n");
                printf("0, 0, 0, 1\n");
            }

            // Lazy initialization
            if (retval = Setup(
                             csr_problem,
                             fused_grid_size,
                             expand_grid_size,
                             filter_grid_size,
                             contract_grid_size)) break;

            // Single-gpu graph slice
            typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

            VertexId iteration 				= 0;
            VertexId queue_index 			= 0;	// Work stealing/queue index
            SizeT queue_length 				= 0;
            int selector 					= 0;

            while (done[0] != 0) {

                VertexId one_phase_iteration = iteration;

                // Run fused contract-expand kernel
                contract_expand_atomic::KernelGlobalBarrier<OnePhasePolicy>
                <<<fused_grid_size, OnePhasePolicy::THREADS>>>(
                    iteration,
                    queue_index,
                    queue_index,												// also serves as steal_index
                    src,
                    graph_slice->frontier_queues.d_keys[selector],				// in edge frontier
                    graph_slice->frontier_queues.d_keys[selector ^ 1],			// out edge frontier
                    graph_slice->frontier_queues.d_values[selector],			// in predecessors
                    graph_slice->frontier_queues.d_values[selector ^ 1],		// out predecessors
                    graph_slice->d_column_indices,
                    graph_slice->d_row_offsets,
                    graph_slice->d_labels,
                    graph_slice->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[0],							// max frontier vertices (all queues should be the same size)
                    global_barrier,
                    fused_kernel_stats,
                    (VertexId *) d_iteration);

                if (DEBUG && (retval = util::B40CPerror<0>(cudaThreadSynchronize(), "contract_expand_atomic::KernelGlobalBarrier failed ", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                // Retrieve output iteration
                if (retval = util::B40CPerror<0>(cudaMemcpy(
                                                  &iteration,
                                                  (VertexId *) d_iteration,
                                                  sizeof(VertexId),
                                                  cudaMemcpyDeviceToHost),
                                              "EnactorHybrid cudaMemcpy d_iteration failed", __FILE__, __LINE__)) break;

                // Check if done or just saturated
                if (iteration < 0) {
                    iteration *= -1;			// saturated
                    done[0] = -1;
                } else {
                    break;						// done
                }

                if ((iteration - one_phase_iteration) & 1) {
                    // An odd number of iterations passed: update selector
                    selector ^= 1;
                }
                // Update queue index by the number of elapsed iterations
                queue_index += (iteration - one_phase_iteration);

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf("\n%lld, , , %lld", (long long) iteration, (long long) queue_length);
                }
                if (INSTRUMENT) {
                    if (retval = fused_kernel_stats.Accumulate(
                                     fused_grid_size,
                                     total_runtimes,
                                     total_lifetimes,
                                     total_queued)) break;
                }

                // Run two-phase until done is not -1
                VertexId two_phase_iteration = iteration;
                /* while (done[0] < 0) { */
                // TODO: Two-phase expansion incorrect on sm_30+
                while (0) {

                    if (DEBUG) printf("\n%lld", (long long) iteration);

                    //
                    // Filter
                    //

                    two_phase::filter_atomic::Kernel<FilterPolicy>
                    <<<filter_grid_size, FilterPolicy::THREADS>>>(
                        queue_index,											// queue counter index
                        queue_index,											// steal counter index
                        d_done,
                        graph_slice->frontier_queues.d_keys[selector],			// edge frontier in
                        graph_slice->frontier_queues.d_keys[selector ^ 1],		// vertex frontier out
                        graph_slice->frontier_queues.d_values[selector],		// predecessor in
                        graph_slice->frontier_queues.d_values[selector ^ 1],	// predecessor out
                        graph_slice->d_visited_mask,
                        this->work_progress,
                        graph_slice->frontier_elements[selector],				// max edge frontier vertices
                        graph_slice->frontier_elements[selector ^ 1],			// max vertex frontier vertices
                        this->filter_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror<0>(cudaThreadSynchronize(), "filter_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;

                    if (DEBUG) {
                        if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                        printf(", %lld", (long long) queue_length);
                    }
                    if (INSTRUMENT) {
                        if (retval = filter_kernel_stats.Accumulate(
                                         filter_grid_size,
                                         total_runtimes,
                                         total_lifetimes)) break;
                    }

                    // Check if done
                    if (done[0] == 0) break;

                    //
                    // Contraction
                    //

                    two_phase::contract_atomic::Kernel<ContractPolicy>
                    <<<contract_grid_size, ContractPolicy::THREADS>>>(
                        src,
                        iteration,
                        0,														// num_elements (unused: we obtain this from device-side counters instead)
                        queue_index,
                        queue_index,											// also serves as steal_index
                        1,														// number of GPUs
                        d_done,
                        graph_slice->frontier_queues.d_keys[selector],			// in edge frontier
                        graph_slice->frontier_queues.d_keys[selector ^ 1],		// out vertex frontier
                        graph_slice->frontier_queues.d_values[selector],		// in predecessors
                        graph_slice->d_labels,
                        graph_slice->d_visited_mask,
                        work_progress,
                        graph_slice->frontier_elements[selector],				// max in vertices
                        graph_slice->frontier_elements[selector ^ 1],			// max out vertices
                        contract_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror<0>(cudaThreadSynchronize(), "contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;

                    if (DEBUG) {
                        if (work_progress.GetQueueLength(queue_index, queue_length)) break;
                        printf(", %lld", (long long) queue_length);
                    }
                    if (INSTRUMENT) {
                        if (contract_kernel_stats.Accumulate(
                                    contract_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }

                    // Throttle
                    if ((iteration - two_phase_iteration) & 1) {
                        if (util::B40CPerror<0>(cudaEventSynchronize(throttle_event),
                                             "LevelGridBfsEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                    } else {
                        if (util::B40CPerror<0>(cudaEventRecord(throttle_event),
                                             "LevelGridBfsEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                    }
                    // Check if done
                    if (done[0] == 0) break;

                    //
                    // Expansion
                    //

                    two_phase::expand_atomic::Kernel<ExpandPolicy>
                    <<<expand_grid_size, ExpandPolicy::THREADS>>>(
                        queue_index,											// queue counter index
                        queue_index,											// steal counter index
                        1,														// number of GPUs
                        d_done,
                        graph_slice->frontier_queues.d_keys[selector],			// in vertex frontier
                        graph_slice->frontier_queues.d_keys[selector ^ 1],		// out edge frontier
                        graph_slice->frontier_queues.d_values[selector ^ 1],	// out predecessors
                        graph_slice->d_column_indices,
                        graph_slice->d_row_offsets,
                        work_progress,
                        graph_slice->frontier_elements[selector],			// max in vertices
                        graph_slice->frontier_elements[selector ^ 1],		// max out vertices
                        expand_kernel_stats);

                    if (DEBUG && (retval = util::B40CPerror<0>(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;
                    cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

                    queue_index++;
                    selector ^= 1;
                    iteration++;

                    if (INSTRUMENT || DEBUG) {
                        if (work_progress.GetQueueLength(queue_index, queue_length)) break;
                        total_queued += queue_length;
                        if (DEBUG) printf(", %lld", (long long) queue_length);
                        if (INSTRUMENT) {
                            expand_kernel_stats.Accumulate(
                                expand_grid_size,
                                total_runtimes,
                                total_lifetimes);
                        }
                    }
                }
            }
            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::B40CPerror<0>(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
                break;
            }

            h_iteration = iteration;

        } while (0);

        if (DEBUG) printf("\n");

        return retval;
    }


    /**
     * Enacts a breadth-first-search on the specified graph problem.
     *
     * @return cudaSuccess on success, error enumeration otherwise
     */
    template <typename CsrProblem>
    cudaError_t EnactSearch(
        CsrProblem 						&csr_problem,
        typename CsrProblem::VertexId 	src,
        int 							max_grid_size = 0)
    {
        typedef typename CsrProblem::VertexId 	VertexId;
        typedef typename CsrProblem::SizeT 		SizeT;

        // GF100
        if (cuda_props.device_sm_version >= 200) {

            const int SATURATION_QUIT = 4 * 128;

            // Fused-iteration contract-expand kernel config
            typedef contract_expand_atomic::KernelPolicy<
            typename CsrProblem::ProblemType,
                     200,
                     INSTRUMENT, 			// INSTRUMENT
                     SATURATION_QUIT,		// SATURATION_QUIT
                     (sizeof(VertexId) > 4) ? 7 : 8,		// CTA_OCCUPANCY
                     7,						// LOG_THREADS
                     0,						// LOG_LOAD_VEC_SIZE
                     0,						// LOG_LOADS_PER_TILE
                     5,						// LOG_RAKING_THREADS
                     util::io::ld::cg,		// QUEUE_READ_MODIFIER,
                     util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
                     util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
                     util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                     util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
                     false,					// WORK_STEALING
                     32,						// WARP_GATHER_THRESHOLD
                     128 * 4, 				// CTA_GATHER_THRESHOLD,
                     5,						// END_BITMASK_CULL
                     6> 						// LOG_SCHEDULE_GRANULARITY
                     OnePhasePolicy;

            // Expansion kernel config
            typedef two_phase::expand_atomic::KernelPolicy<
            typename CsrProblem::ProblemType,
                     200,
                     INSTRUMENT, 			// INSTRUMENT
                     8,						// CTA_OCCUPANCY
                     7,						// LOG_THREADS
                     0,						// LOG_LOAD_VEC_SIZE
                     0,						// LOG_LOADS_PER_TILE
                     5,						// LOG_RAKING_THREADS
                     util::io::ld::cg,		// QUEUE_READ_MODIFIER,
                     util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
                     util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
                     util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
                     util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
                     true,					// WORK_STEALING
                     32,						// WARP_GATHER_THRESHOLD
                     128 * 4, 				// CTA_GATHER_THRESHOLD,
                     7> 						// LOG_SCHEDULE_GRANULARITY
                     ExpandPolicy;

            // Filter kernel config
            typedef two_phase::filter_atomic::KernelPolicy<
            typename CsrProblem::ProblemType,
                     200,					// CUDA_ARCH
                     INSTRUMENT, 			// INSTRUMENT
                     SATURATION_QUIT, 		// SATURATION_QUIT
                     8,						// CTA_OCCUPANCY
                     7,						// LOG_THREADS
                     1,						// LOG_LOAD_VEC_SIZE
                     1,						// LOG_LOADS_PER_TILE
                     5,						// LOG_RAKING_THREADS
                     util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
                     util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
                     false,					// WORK_STEALING
                     9> 						// LOG_SCHEDULE_GRANULARITY
                     FilterPolicy;

            // Contraction kernel config
            typedef two_phase::contract_atomic::KernelPolicy<
            typename CsrProblem::ProblemType,
                     200,
                     INSTRUMENT, 			// INSTRUMENT
                     0, 						// SATURATION_QUIT
                     true, 					// DEQUEUE_PROBLEM_SIZE
                     8,						// CTA_OCCUPANCY
                     7,						// LOG_THREADS
                     1,						// LOG_LOAD_VEC_SIZE
                     0,						// LOG_LOADS_PER_TILE
                     5,						// LOG_RAKING_THREADS
                     util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
                     util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
                     false,					// WORK_STEALING
                     0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
                     8> 						// LOG_SCHEDULE_GRANULARITY
                     ContractPolicy;

            return EnactSearch<OnePhasePolicy, ExpandPolicy, FilterPolicy, ContractPolicy>(
                       csr_problem, src, max_grid_size);
        }

        /* Commented out to reduce compile time. Uncomment for GT200

            	// GT200
            	if (cuda_props.device_sm_version >= 130) {

                	const int SATURATION_QUIT = 4 * 128;

                	// Fused-iteration contract-expand kernel config
        			typedef contract_expand_atomic::KernelPolicy<
        				typename CsrProblem::ProblemType,
        				130,
        				INSTRUMENT, 			// INSTRUMENT
        				SATURATION_QUIT,		// SATURATION_QUIT
        				1,						// CTA_OCCUPANCY
        				8,						// LOG_THREADS
        				0,						// LOG_LOAD_VEC_SIZE
        				1,						// LOG_LOADS_PER_TILE
        				5,						// LOG_RAKING_THREADS
        				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
        				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
        				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
        				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
        				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
        				false,					// WORK_STEALING
        				32,						// WARP_GATHER_THRESHOLD
        				128 * 4, 				// CTA_GATHER_THRESHOLD,
        				5,						// END_BITMASK_CULL
        				6> 						// LOG_SCHEDULE_GRANULARITY
        					OnePhasePolicy;

        			// Expansion kernel config
        			typedef two_phase::expand_atomic::KernelPolicy<
        				typename CsrProblem::ProblemType,
        				130,
        				INSTRUMENT, 			// INSTRUMENT
        				1,						// CTA_OCCUPANCY
        				8,						// LOG_THREADS
        				0,						// LOG_LOAD_VEC_SIZE
        				1,						// LOG_LOADS_PER_TILE
        				5,						// LOG_RAKING_THREADS
        				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
        				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
        				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
        				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
        				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
        				false,					// WORK_STEALING
        				32,						// WARP_GATHER_THRESHOLD
        				128 * 4, 				// CTA_GATHER_THRESHOLD,
        				7> 						// LOG_SCHEDULE_GRANULARITY
        					ExpandPolicy;

        			// Filter kernel config
        			typedef two_phase::filter_atomic::KernelPolicy<
        				typename CsrProblem::ProblemType,
        				130,					// CUDA_ARCH
        				INSTRUMENT, 			// INSTRUMENT
        				0, 						// SATURATION_QUIT
        				8,						// CTA_OCCUPANCY
        				7,						// LOG_THREADS
        				1,						// LOG_LOAD_VEC_SIZE
        				0,						// LOG_LOADS_PER_TILE
        				5,						// LOG_RAKING_THREADS
        				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
        				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
        				false,					// WORK_STEALING
        				8> 						// LOG_SCHEDULE_GRANULARITY
        					FilterPolicy;

        			// Contraction kernel config
        			typedef two_phase::contract_atomic::KernelPolicy<
        				typename CsrProblem::ProblemType,
        				130,
        				INSTRUMENT, 			// INSTRUMENT
        				SATURATION_QUIT, 		// SATURATION_QUIT
        				true, 					// DEQUEUE_PROBLEM_SIZE
        				1,						// CTA_OCCUPANCY
        				8,						// LOG_THREADS
        				1,						// LOG_LOAD_VEC_SIZE
        				1,						// LOG_LOADS_PER_TILE
        				6,						// LOG_RAKING_THREADS
        				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
        				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
        				false,					// WORK_STEALING
        				0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
        				10> 					// LOG_SCHEDULE_GRANULARITY
        					ContractPolicy;

        			return EnactSearch<OnePhasePolicy, ExpandPolicy, FilterPolicy, ContractPolicy>(
        				csr_problem, src, max_grid_size);
        	    }
        */

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

};


} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

