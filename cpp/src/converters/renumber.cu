// -*-c++-*-

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

  //
  //  Added this back in.  Below I added support for strings, however the 
  //  cudf python interface doesn't fully support strings yet, so the below
  //  code can't be debugged.  Rather than remove the code, this error check
  //  will prevent code from being executed.  Once cudf fully support string
  //  columns we can eliminate this check and debug the GDF_STRING case below.
  //
  GDF_REQUIRE( ((src->dtype == GDF_INT32) || (src->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src->size > 0, GDF_DATASET_EMPTY );

  //
  //  TODO: we're currently renumbering without using valid.  We need to
  //        worry about that at some point, but for now we'll just
  //        copy the valid pointers to the new columns and go from there.
  //
  cudaStream_t stream{nullptr};

  //
  //  For now, let's just specify a default value of the hash size.
  //  This should be configurable.
  //
  int hash_size = 8191;

  //
  // TODO:  I assume int32_t for output.  At the moment, the rest of
  //        cugraph assumes int32_t for vertex ids.  Until that assumption
  //        changes, we can just hardcode to int32_t.
  //
  //    * I could match src->dtype - since if the raw values fit in an int32_t,
  //      then the renumbered values must fit within an int32_t
  //    * If input is 64-bit, I could compute with 64 bit integers and check
  //      check if new_size < (2^31 - 1) then I could allocate 32-bit integers
  //      and copy them in order to make the final footprint smaller.
  //    * I could make the user suggest a desired behavior.
  //

  //
  //  Renumbering is different based upon the column types.  Note
  //  that we required src and dst data types to match above.
  //
  switch (src->dtype) {
  case GDF_INT32:
    {
      size_t new_size;
      int32_t *tmp;

      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(src_renumbered, tmp, src->valid, src->size, src->dtype);

      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(dst_renumbered, tmp, dst->valid, dst->size, dst->dtype);

      gdf_error err = cugraph::renumber_vertices(src->size,
                                                 static_cast<const int32_t *>(src->data),
                                                 static_cast<const int32_t *>(dst->data),
                                                 static_cast<int32_t *>(src_renumbered->data),
                                                 static_cast<int32_t *>(dst_renumbered->data),
                                                 &new_size,
                                                 &tmp,
                                                 cugraph::HashFunctionObjectInt(hash_size),
                                                 thrust::less<int32_t>()
                                                 );
      if (err != GDF_SUCCESS)
        return err;

      gdf_column_view(numbering_map, tmp, nullptr, new_size, src->dtype);
      break;
    }

  case GDF_INT64:
    {
      size_t new_size;

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
      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(src_renumbered, tmp, src->valid, src->size, GDF_INT32);

      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(dst_renumbered, tmp, dst->valid, dst->size, GDF_INT32);

      gdf_error err = cugraph::renumber_vertices(src->size,
                                                 static_cast<const int64_t *>(src->data),
                                                 static_cast<const int64_t *>(dst->data),
                                                 static_cast<int32_t *>(src_renumbered->data),
                                                 static_cast<int32_t *>(dst_renumbered->data),
                                                 &new_size,
                                                 &tmp,
                                                 cugraph::HashFunctionObjectInt(hash_size),
                                                 thrust::less<int64_t>()
                                                 );
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

      //
      //  Note that the numbering_map column matches the type of the input
      //  columns (we want the numbering map to take us back to the
      //  original data, so it must match that column type).
      //
      gdf_column_view(numbering_map, tmp, nullptr, new_size, src->dtype);
      break;
    }

  case GDF_STRING:
    {
      size_t new_size;

      int32_t *tmp;
      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(src_renumbered, tmp, src->valid, src->size, GDF_INT32);

      ALLOC_TRY((void**) &tmp, sizeof(int32_t) * src->size, stream);
      gdf_column_view(dst_renumbered, tmp, dst->valid, dst->size, GDF_INT32);

      NVStrings *srcList = reinterpret_cast<NVStrings*>(src->data);
      NVStrings *dstList = reinterpret_cast<NVStrings*>(dst->data);

      std::pair<const char *, size_t> *srcs;
      std::pair<const char *, size_t> *dsts;
      std::pair<const char *, size_t> *output_map;

      ALLOC_TRY((void**) &srcs, sizeof(std::pair<const char *, size_t>) * src->size, stream);
      ALLOC_TRY((void**) &dsts, sizeof(std::pair<const char *, size_t>) * dst->size, stream);

      srcList->create_index(srcs, true);
      dstList->create_index(dsts, true);
      
      gdf_error err = cugraph::renumber_vertices(src->size,
                                                 srcs,
                                                 dsts,
                                                 static_cast<int32_t *>(src_renumbered->data),
                                                 static_cast<int32_t *>(dst_renumbered->data),
                                                 &new_size,
                                                 &output_map,
                                                 cugraph::HashFunctionObjectString(hash_size),
                                                 cugraph::CompareString()
                                                 );
      if (err != GDF_SUCCESS)
        return err;

      //
      //  We're done with srcs and dsts
      //
      ALLOC_FREE_TRY(srcs, stream);
      ALLOC_FREE_TRY(dsts, stream);

      //
      //  If there are too many vertices then the renumbering overflows so we'll
      //  return an error.
      //
      if (new_size > 0x7fffffff) {
        ALLOC_FREE_TRY(src_renumbered, stream);
        ALLOC_FREE_TRY(dst_renumbered, stream);
        return GDF_COLUMN_SIZE_TOO_BIG;
      }

      //
      //  Note that the numbering_map column matches the type of the input
      //  columns (we want the numbering map to take us back to the
      //  original data, so it must match that column type).
      //
      gdf_column_view(numbering_map, output_map, nullptr, new_size, src->dtype);
      break;
    }

  default:
    return GDF_UNSUPPORTED_DTYPE;
  }

  return GDF_SUCCESS;
}
