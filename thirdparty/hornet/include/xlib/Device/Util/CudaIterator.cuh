/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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
 *
 * @file
 */
#pragma once

namespace xlib {

/**
 * offset iterator
 */
template<typename T, unsigned BlockSize>
struct cuda_iterator {
protected:
    cusize_t offset;
    const cusize_t stride = BlockSize * gridDim.x;
public:
    __device__ __forceinline__
    cuda_iterator(cusize_t _offset) : offset(_offset) {}

    /*__device__ __forceinline__
    cuda_iterator(const cuda_iterator<T, BlockSize>& obj) = delete;*/

    __device__ __forceinline__
    cusize_t operator*() const {
        return offset;
    }

    __device__ __forceinline__
    cuda_iterator& operator++() {
        offset += stride;
        return *this;
    }

    __device__ __forceinline__
    bool operator!=(const cuda_iterator& obj) const {
        return offset < obj.offset;
    }
};


/**
 * pointer iterator, safe for warp/block (all threads enter in the main loop)
 */
template<unsigned BlockSize, typename iterator_node_t>
struct cuda_safe_iterator : cuda_iterator<cusize_t, BlockSize> {
protected:
    cusize_t max_size;
public:
    iterator_node_t& node;
    using cuda_iterator<cusize_t, BlockSize>::offset;

    __device__ __forceinline__
    cuda_safe_iterator(cusize_t _offset,
                       cusize_t _max_size,
                       iterator_node_t& _node) :
        cuda_iterator<cusize_t, BlockSize>(_offset),
        max_size(_max_size),
        node(_node) {}

    /*__device__ __forceinline__
    cuda_safe_iterator(const cuda_safe_iterator
                            <BlockSize, iterator_node_t>& obj) = delete;*/

    __device__ __forceinline__
    iterator_node_t& operator*() {
        node.eval(offset, max_size);
        return node;
    }
};

//------------------------------------------------------------------------------

/**
 * pointer + offset iterator
 */
template<typename T, unsigned BlockSize>
struct cuda_forward_iterator {
protected:
    T* const ptr = nullptr;
public:
    cusize_t offset;
    const cusize_t stride = BlockSize * gridDim.x;

    __device__ __forceinline__
    cuda_forward_iterator() : offset(0) {}

    __device__ __forceinline__
    cuda_forward_iterator(T* const _ptr, cusize_t _offset) :
        ptr(_ptr),
        offset(_offset) {}

    /*__device__ __forceinline__
    cuda_forward_iterator(const cuda_forward_iterator<T, BlockSize>& obj)
        = delete;*/

    __device__ __forceinline__
    cuda_forward_iterator& operator++() {
        offset += stride;
        return *this;
    }

    __device__ __forceinline__
    T& operator*() const {
        return *(ptr + offset);
    }

    __device__ __forceinline__
    bool operator!=(const cuda_forward_iterator& obj) const {
        return obj.offset > offset;
    }
};

//==============================================================================

/**
 * pointer + offset data structure
 */
template<typename T, unsigned BlockSize, unsigned VW_SIZE = 1>
struct cu_array {
private:
    T* const array;
    const cusize_t size;
    static const unsigned STRIDE = BlockSize / VW_SIZE;
public:
    __device__ __forceinline__
    cu_array(T* _array, T _size) : array(_array), size(_size) {}

    __device__ __forceinline__
    cuda_forward_iterator<T, STRIDE> begin() const {
        const unsigned global_id = (blockIdx.x * BlockSize + threadIdx.x)
                                   / VW_SIZE;
        return cuda_forward_iterator<T, STRIDE>(array, global_id);
    }

    __device__ __forceinline__
    cuda_forward_iterator<T, STRIDE> end() const {
        return cuda_forward_iterator<T, STRIDE>(array, size);
    }
};

template<typename T, unsigned BlockSize, unsigned VW_SIZE = 1>
struct cuda_range_loop {
private:
    const cusize_t size;
    static const unsigned STRIDE = BlockSize / VW_SIZE;
public:
    __device__ __forceinline__
    cuda_range_loop(T _size) : size(_size) {}

    __device__ __forceinline__
    cuda_iterator<T, STRIDE> begin() const {
        const unsigned global_id = (blockIdx.x * BlockSize + threadIdx.x)
                                   / VW_SIZE;
        return cuda_iterator<T, STRIDE>(global_id);
    }

    __device__ __forceinline__
    cuda_iterator<T, STRIDE> end() const {
        return cuda_iterator<T, STRIDE>(size);
    }
};

} // namespace xlib
