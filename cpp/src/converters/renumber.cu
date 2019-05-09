/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Renumber vertices
// Author: Chuck Hastings charlesh@nvidia.com

#include "renumber.cuh"

gdf_error gdf_renumber_vertices(const gdf_column *src, const gdf_column *dst,
				gdf_column *src_renumbered, gdf_column *dst_renumbered,
				gdf_column *numbering_map) {

  GDF_REQUIRE( src->size == dst->size, GDF_COLUMN_SIZE_MISMATCH );
  GDF_REQUIRE( src->dtype == dst->dtype, GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((src->dtype == GDF_INT32) || (src->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src->size > 0, GDF_DATASET_EMPTY );

  //
  //  TODO: we're currently renumbering without using valid.  We need to
  //        worry about that at some point, but for now we'll just
  //        copy the valid pointers to the new columns and go from there.
  //
  cudaStream_t stream{nullptr};

  size_t src_size = src->size;
  size_t new_size;

  //
  // TODO:  I assume int64_t for output.  A few thoughts:
  //
  //    * I could match src->dtype - since if the raw values fit in an int32_t,
  //      then the renumbered values must fit within an int32_t
  //    * If new_size < (2^31 - 1) then I could allocate 32-bit integers
  //      and copy them in order to make the final footprint smaller.
  //
  //
  //  NOTE:  Forcing match right now - it appears that cugraph is artficially
  //         forcing the type to be 32
  if (src->dtype == GDF_INT32) {
    int32_t *tmp;

    ALLOC_MANAGED_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
    gdf_column_view(src_renumbered, tmp, src->valid, src->size, src->dtype);

    ALLOC_MANAGED_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
    gdf_column_view(dst_renumbered, tmp, dst->valid, dst->size, dst->dtype);

    gdf_error err = cugraph::renumber_vertices(src_size,
					       (const int32_t *) src->data,
					       (const int32_t *) dst->data,
					       (int32_t *) src_renumbered->data,
					       (int32_t *) dst_renumbered->data,
					       &new_size, &tmp);
    if (err != GDF_SUCCESS)
      return err;

    gdf_column_view(numbering_map, tmp, nullptr, new_size, src->dtype);
  } else if (src->dtype == GDF_INT64) {

    //
    //  NOTE: At the moment, we force the renumbered graph to use
    //        32-bit integer ids.  Since renumbering is going to make
    //        the vertex range dense, this limits us to 2 billion
    //        vertices.
    //
    //        The renumbering code supports 64-bit integer generation
    //        so we can run this with int64_t output if desired...
    //        but none of the algorithms support that.
    //
    int64_t *tmp;
    ALLOC_MANAGED_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
    gdf_column_view(src_renumbered, tmp, src->valid, src->size, GDF_INT32);

    ALLOC_MANAGED_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
    gdf_column_view(dst_renumbered, tmp, dst->valid, dst->size, GDF_INT32);

    gdf_error err = cugraph::renumber_vertices(src_size,
					       (const int64_t *) src->data,
					       (const int64_t *) dst->data,
					       (int32_t *) src_renumbered->data,
					       (int32_t *) dst_renumbered->data,
					       &new_size, &tmp);
    if (err != GDF_SUCCESS)
      return err;

    //
    //  If there are too many vertices then the renumbering overflows so we'll
    //  return an error.
    //
    if (new_size > 0x7fffffff) {
      ALLOC_FREE_TRY(src_renumbered, stream);
      ALLOC_FREE_TRY(dst_renumbered, stream);
      return GDF_COLUMN_SIZE_TOO_BIG;
    }

    gdf_column_view(numbering_map, tmp, nullptr, new_size, src->dtype);
  } else {
    return GDF_UNSUPPORTED_DTYPE;
  }

  return GDF_SUCCESS;
}

