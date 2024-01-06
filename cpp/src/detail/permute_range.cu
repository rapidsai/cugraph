

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
// #include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

namespace cugraph {

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> permute_range(raft::handle_t const& handle,
                                            raft::random::RngState& rng_state,
                                            vertex_t local_range_size,
                                            vertex_t local_range_start,
                                            bool multi_gpu)
{
  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto global_start =
      cugraph::host_scalar_bcast(handle.get_comms(), local_range_start, 0, handle.get_stream());
    auto sub_range_sizes =
      cugraph::host_scalar_allgather(handle.get_comms(), local_range_size, handle.get_stream());
    std::exclusive_scan(
      sub_range_sizes.begin(), sub_range_sizes.end(), sub_range_sizes.begin(), global_start);
    CUGRAPH_EXPECTS(
      sub_range_sizes[comm_rank] == local_range_start,
      "Invalid input arguments: a rage must have contiguous and non-overlapping values");
  }
  rmm::device_uvector<vertex_t> random_cluster_assignments(local_range_size, handle.get_stream());

  // generate as many number as #local_vertices on each GPU
  detail::sequence_fill(handle.get_stream(),
                        random_cluster_assignments.begin(),
                        random_cluster_assignments.size(),
                        local_range_start);

  // shuffle/permute locally
  rmm::device_uvector<float> random_numbers(random_cluster_assignments.size(), handle.get_stream());

  cugraph::detail::uniform_random_fill(handle.get_stream(),
                                       random_numbers.data(),
                                       random_numbers.size(),
                                       float{0.0},
                                       float{1.0},
                                       rng_state);
  thrust::sort_by_key(handle.get_thrust_policy(),
                      random_numbers.begin(),
                      random_numbers.end(),
                      random_cluster_assignments.begin());

  if (multi_gpu) {
    // distribute shuffled/permuted numbers to other GPUs
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    std::vector<size_t> tx_value_counts(comm_size);
    std::fill(tx_value_counts.begin(),
              tx_value_counts.end(),
              random_cluster_assignments.size() / comm_size);

    std::vector<vertex_t> h_random_gpu_ranks;
    {
      rmm::device_uvector<vertex_t> d_random_numbers(random_cluster_assignments.size() % comm_size,
                                                     handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           d_random_numbers.data(),
                                           d_random_numbers.size(),
                                           vertex_t{0},
                                           vertex_t{comm_size},
                                           rng_state);

      h_random_gpu_ranks.resize(d_random_numbers.size());

      raft::update_host(h_random_gpu_ranks.data(),
                        d_random_numbers.data(),
                        d_random_numbers.size(),
                        handle.get_stream());
    }

    for (int i = 0; i < static_cast<int>(random_cluster_assignments.size() % comm_size); i++) {
      tx_value_counts[h_random_gpu_ranks[i]]++;
    }

    std::tie(random_cluster_assignments, std::ignore) = cugraph::shuffle_values(
      handle.get_comms(), random_cluster_assignments.begin(), tx_value_counts, handle.get_stream());

    // shuffle/permute locally again
    random_numbers.resize(random_cluster_assignments.size(), handle.get_stream());

    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         random_numbers.data(),
                                         random_numbers.size(),
                                         float{0.0},
                                         float{1.0},
                                         rng_state);
    thrust::sort_by_key(handle.get_thrust_policy(),
                        random_numbers.begin(),
                        random_numbers.end(),
                        random_cluster_assignments.begin());

    // take care of deficits and extras numbers

    int nr_extras =
      static_cast<int>(random_cluster_assignments.size()) - static_cast<int>(local_range_size);
    int nr_deficits = nr_extras >= 0 ? 0 : -nr_extras;

    auto extra_cluster_ids = cugraph::detail::device_allgatherv(
      handle,
      comm,
      raft::device_span<vertex_t const>(random_cluster_assignments.data() + local_range_size,
                                        nr_extras > 0 ? nr_extras : 0));

    random_cluster_assignments.resize(local_range_size, handle.get_stream());
    auto deficits =
      cugraph::host_scalar_allgather(handle.get_comms(), nr_deficits, handle.get_stream());

    std::exclusive_scan(deficits.begin(), deficits.end(), deficits.begin(), vertex_t{0});

    raft::copy(random_cluster_assignments.data() + local_range_size - nr_deficits,
               extra_cluster_ids.begin() + deficits[comm_rank],
               nr_deficits,
               handle.get_stream());
  }

  assert(random_cluster_assignments.size() == local_range_size);
  return random_cluster_assignments;
}

template rmm::device_uvector<int32_t> permute_range(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    int32_t local_range_size,
                                                    int32_t local_range_start,
                                                    bool multi_gpu);

template rmm::device_uvector<int64_t> permute_range(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    int64_t local_range_size,
                                                    int64_t local_range_start,
                                                    bool multi_gpu);

}  // namespace detail
}  // namespace cugraph
