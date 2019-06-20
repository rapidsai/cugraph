/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @brief Improved CUDA APIs
 * @details Advatages:                                                      <br>
 *   - **clear semantic**: input, then output (google style)
 *   - **type checking**:
 *      - input and output must have the same type T
 *      - const checking for inputs
 *      - device symbols must be references
 *   - **no byte object sizes**: the number of bytes is  determined by looking
 *       the parameter type T
 *   - **fast debugging**:
 *      - in case of error the macro provides the file name, the line, the
 *        name of the function where it is called, and the API name that fail
 *      - assertion to check null pointers and num_items == 0
 *      - assertion to check every CUDA API errors
 *      - additional info: cudaMalloc fail -> what is the available memory?
 *   - **direct argument passing** of constant values. E.g.                 <br>
 *       \code{.cu}
 *        cuMemcpyToSymbol(false, d_symbol); //d_symbol must be bool
 *       \endcode
 *   - much **less verbose**
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
 */
#pragma once

#include "Device/Util/CudaUtil.cuh" //__cudaErrorHandler
#include <cassert>                  //assert

#if defined(NEVER_DEFINED)
    #include "SafeFunctionsAsync_.cuh"
#endif

///@cond
#define cuMemcpyDevToDevAsync(...)                                             \
    xlib::detail::cuMemcpyDevToDevAsyncAux(__FILE__, __LINE__,__func__,        \
                                           __VA_ARGS__)                        \

#define cuMemcpyToDeviceAsync(...)                                             \
    xlib::detail::cuMemcpyToDeviceAsyncAux(__FILE__, __LINE__,__func__,        \
                                           __VA_ARGS__)                        \

#define cuMemcpyToHostAsync(...)                                               \
    xlib::detail::cuMemcpyToHostAsyncAux(__FILE__, __LINE__, __func__,         \
                                         __VA_ARGS__)                          \

//------------------------------------------------------------------------------

#define cuMemcpyToSymbolAsync(...)                                             \
    xlib::detail::cuMemcpyToSymbolAsyncAux(__FILE__, __LINE__,__func__,        \
                                           __VA_ARGS__)                        \

#define cuMemcpyFromSymbolAsync(...)                                           \
    xlib::detail::cuMemcpyFromSymbolAsyncAux(__FILE__, __LINE__,__func__       \
                                             __VA_ARGS__)                      \

//------------------------------------------------------------------------------

#define cuMemset0x00Async(...)                                                 \
    xlib::detail::cuMemset0x00AsyncAux(__FILE__, __LINE__, __func__,           \
                                       __VA_ARGS__)                            \

#define cuMemset0xFFAsync(...)                                                 \
    xlib::detail::cuMemset0xFFAsyncAux(__FILE__, __LINE__, __func__,           \
                                       __VA_ARGS__)                            \

#define cuMemsetAsync(...)                                                     \
    xlib::detail::cuMemsetAsyncAux(__FILE__, __LINE__, __func__, __VA_ARGS__)  \

//------------------------------------------------------------------------------

#define cuMemcpy2DToDeviceAsync(...)                                           \
    xlib::detail::cuMemcpy2DToDeviceAsyncAux(__FILE__,  __LINE__, __func__,    \
                                             __VA_ARGS__)                      \

#define cuMemcpy2DToHostAsync(...)                                             \
    xlib::detail::cuMemcpy2DToHostAsyncAux(__FILE__,  __LINE__, __func__,      \
                                           __VA_ARGS__)                        \

#define cuMemcpy2DDevToDevAsync(...)                                           \
    xlib::detail::cuMemcpy2DDevToDevAsyncAux(__FILE__,  __LINE__, __func__,    \
                                             __VA_ARGS__)                      \

//==============================================================================
//==============================================================================

namespace xlib {
namespace detail {

//==============================================================================
////////////////
//  cuMemset  //
////////////////

template<typename T>
void cuMemsetAsyncGenericAux(const char* file, int line, const char* func_name,
                             T* ptr, size_t num_items, unsigned char mask,
                             cudaStream_t stream = 0) noexcept {
    assert(num_items > 0 && ptr != nullptr);
    char api_name[] = "cudaMemset(0x__)";
    char value1 = static_cast<char>(mask / (0xF));
    char value2 = static_cast<char>(mask % (0xF));
    api_name[13] = (value1 <= '9') ? '0' + value1 : 'A' + value1 - 10;
    api_name[14] = (value2 <= '9') ? '0' + value2 : 'A' + value2 - 10;
    cudaErrorHandler(cudaMemsetAsync(ptr, mask, num_items * sizeof(T), stream),
                     api_name, file, line, func_name);
}

template<typename T>
void cuMemset0x00AsyncAux(const char* file, int line, const char* func_name,
                          T* ptr, size_t num_items = 1, cudaStream_t stream = 0)
                          noexcept {
    cuMemsetAsyncGenericAux(file, line, func_name, ptr, num_items, 0x00,
                            stream);
}

#if 0//asynchronous functions should have Async in their function name, these functions either conflict with synchrnous versions (if both SafeCudaAPIAsync.cuh and SafeCudaAPISync.cuh are included) or can unintentionally bind to synchronous or asynchronous versions based on included header files
template<typename T>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T& symbol, cudaStream_t stream = 0) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, 1, 0x00,
                            stream);
}

template<typename T, int SIZE>
void cuMemset0x00Aux(const char* file, int line, const char* func_name,
                     T (&symbol)[SIZE], cudaStream_t stream = 0) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, SIZE, 0x00,
                            stream);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T* ptr, size_t num_items = 1, cudaStream_t stream = 0)
                     noexcept {
    cuMemsetAsyncGenericAux(file, line, func_name, ptr, num_items, 0xFF,
                            stream);
}

template<typename T>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T& symbol, cudaStream_t stream = 0) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, 1, 0xFF,
                            stream);
}

template<typename T, int SIZE>
void cuMemset0xFFAux(const char* file, int line, const char* func_name,
                     T (&symbol)[SIZE], cudaStream_t stream = 0) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, SIZE, 0xFF,
                            stream);
}

template<typename T>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T* ptr, size_t num_items, unsigned char mask,
                 cudaStream_t stream = 0) noexcept {
    cuMemsetAsyncGenericAux(file, line, func_name, ptr, num_items, mask,
                            stream);
}

template<typename T>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T& symbol, unsigned char mask, cudaStream_t stream = 0)
                 noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, 1, mask,
                            stream);
}

template<typename T, int SIZE>
void cuMemsetAux(const char* file, int line, const char* func_name,
                 T (&symbol)[SIZE], unsigned char mask,
                 cudaStream_t stream = 0) noexcept {
    T* symbol_address;
    SAFE_CALL( cudaGetSymbolAddress(symbol_address, symbol) )
    cuMemsetAsyncGenericAux(file, line, func_name, symbol_address, SIZE, mask,
                            stream);
}
#endif

//==============================================================================
//////////////////
//  cuMemset2D  //
//////////////////

template<typename T>
void cuMemset2DAsyncGenericAux(const char* file, int line, const char* func_name,
                          T* ptr, size_t rows, size_t cols, size_t pitch,
                          unsigned char mask, cudaStream_t stream) noexcept {
    assert(ptr != nullptr && rows > 0 && cols > 0 && pitch >= cols);
    char api_name[] = "cudaMemset2DAsync(0x__)";
    char value1 = static_cast<char>(mask / (0xF));
    char value2 = static_cast<char>(mask % (0xF));
    api_name[13] = (value1 <= '9') ? '0' + value1 : 'A' + value1 - 10;
    api_name[14] = (value2 <= '9') ? '0' + value2 : 'A' + value2 - 10;
    cudaErrorHandler(cudaMemset2DAsync(ptr, pitch * sizeof(T), mask,
                                       cols * sizeof(T), rows, stream),
                     api_name, file, line, func_name);
}

template<typename T>
void cuMemset2DAsync0x00Aux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols, size_t pitch,
                       cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, pitch,
                              0x00, stream);
}

template<typename T>
void cuMemset2DAsync0x00Aux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols,
                       cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, cols,
                              0x00, stream);
}

template<typename T>
void cuMemset2DAsync0xFFAux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols,
                       cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, cols,
                              0xFF, stream);
}

template<typename T>
void cuMemset2DAsync0xFFAux(const char* file, int line, const char* func_name,
                       T* ptr, size_t rows, size_t cols, size_t pitch,
                       cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, pitch,
                              0xFF, stream);
}

template<typename T>
void cuMemset2DAsyncAux(const char* file, int line, const char* func_name,
                   T* ptr, size_t rows, size_t cols, unsigned char mask,
                   cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, cols,
                              mask, stream);
}

template<typename T>
void cuMemset2DAsyncAux(const char* file, int line, const char* func_name,
                        T* ptr, size_t rows, size_t cols, size_t pitch,
                        unsigned char mask, cudaStream_t stream = 0) noexcept {
    cuMemset2DAsyncGenericAux(file, line, func_name, ptr, rows, cols, pitch,
                              mask, stream);
}

//==============================================================================
////////////////
//  cuMemcpy  //
////////////////

template<typename T>
void cuMemcpyAsyncGenericAux(const char* file, int line, const char* func_name,
                             const T* input, size_t num_items, T* output,
                             cudaMemcpyKind cuda_memcpy_kind,
                             cudaStream_t stream = 0) noexcept {
    assert(input != nullptr && output != nullptr);
    if (num_items == 0)
        return;
    const char* api_name[] = { "", "cudaMemcpyAsync(ToDevice)",
                               "cudaMemcpyAsync(ToHost)",
                               "cudaMemcpyAsync(DeviceToDevice)", "" };
    const auto& selected = api_name[static_cast<int>(cuda_memcpy_kind)];
    cudaErrorHandler(cudaMemcpyAsync(output, input, num_items * sizeof(T),
                                     cuda_memcpy_kind, stream),
                     selected, file, line, func_name);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t num_items, T* output,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, input, num_items, output,
                            cudaMemcpyHostToDevice, stream);
}

template<typename T>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T& input, T* output,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, &input, 1, output,
                            cudaMemcpyHostToDevice, stream);
}

//Fixed Array to Pointer
template<typename T, int SIZE>
void cuMemcpyToDeviceAsyncAux(const char* file, int line, const char* func_name,
                              const T (&input)[SIZE], T* output,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, &input, SIZE, output,
                            cudaMemcpyHostToDevice, stream);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyToHostAsyncAux(const char* file, int line, const char* func_name,
                            const T* input, size_t num_items, T* output,
                            cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, input, num_items, output,
                            cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void cuMemcpyToHostAsyncAux(const char* file, int line, const char* func_name,
                            const T* input, T& output,
                            cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, input, 1, &output,
                            cudaMemcpyDeviceToHost, stream);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpyDevToDevAsyncAux(const char* file, int line, const char* func_name,
                         const T* input, size_t num_items, T* output,
                         cudaStream_t stream = 0) noexcept {
    cuMemcpyAsyncGenericAux(file, line, func_name, input, num_items, output,
                            cudaMemcpyDeviceToDevice, stream);
}

//==============================================================================
//////////////////
//  cuMemcpy2D  //
//////////////////

template<typename T>
void cuMemcpy2DAsyncGeneric(const char* file, int line, const char* func_name,
                            const T* input, size_t rows, size_t cols,
                            size_t src_pitch, T* output, size_t dst_pitch,
                            cudaMemcpyKind cuda_memcpy_kind,
                            cudaStream_t stream = 0) noexcept {
    assert(input != nullptr && output != nullptr && rows > 0 && cols > 0 &&
           src_pitch >= cols && dst_pitch >= cols);
    const char* api_name[] = { "", "cuda2DMemcpy(ToDevice)",
                               "cuda2DMemcpy(ToHost)",
                               "cuda2DMemcpy(DeviceToDevice)", "" };
    const auto& selected = api_name[static_cast<int>(cuda_memcpy_kind)];
    cudaErrorHandler(cudaMemcpy2DAsync(output, dst_pitch * sizeof(T),
                                       input, src_pitch * sizeof(T),
                                       cols * sizeof(T), rows,
                                       cuda_memcpy_kind, stream),
                     selected, file, line, func_name);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DToDeviceAsyncAux(const char* file, int line,
                                const char* func_name,
                                const T* input, size_t rows, size_t cols,
                                size_t src_pitch, T* output, size_t dst_pitch,
                                cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, dst_pitch, cudaMemcpyHostToDevice, stream);
}

template<typename T>
void cuMemcpy2DToDeviceAsyncAux(const char* file, int line,
                                const char* func_name,
                                const T* input, size_t rows, size_t cols,
                                T* output, size_t dst_pitch,
                                cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, dst_pitch, cudaMemcpyHostToDevice, stream);
}

template<typename T>
void cuMemcpy2DToDeviceAsyncAux(const char* file, int line,
                                const char* func_name, const T* input,
                                size_t rows, size_t cols, size_t src_pitch,
                                T* output, cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, cols, cudaMemcpyHostToDevice, stream);
}

template<typename T>
void cuMemcpy2DToDeviceAsyncAux(const char* file, int line, const char* func_name,
                                const T* input, size_t rows, size_t cols,
                                T* output, cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, cols, cudaMemcpyHostToDevice, stream);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DToHostAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t rows, size_t cols,
                              size_t src_pitch, T* output, size_t dst_pitch,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, dst_pitch, cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void cuMemcpy2DToHostAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t rows, size_t cols,
                              T* output, size_t dst_pitch,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, dst_pitch, cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void cuMemcpy2DToHostAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t rows, size_t cols,
                              size_t src_pitch, T* output,
                              cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, cols, cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void cuMemcpy2DToHostAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t rows, size_t cols,
                              T* output, cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, cols, cudaMemcpyDeviceToHost, stream);
}

//------------------------------------------------------------------------------

template<typename T>
void cuMemcpy2DDevToDevAsyncAux(const char* file, int line,
                                const char* func_name, const T* input,
                                size_t rows, size_t cols, size_t src_pitch,
                                T* output, size_t dst_pitch,
                                cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, dst_pitch, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
void cuMemcpy2DDevToDevAsyncAux(const char* file, int line,
                                const char* func_name, const T* input,
                                size_t rows, size_t cols, T* output,
                                size_t dst_pitch, cudaStream_t stream = 0)
                                noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, dst_pitch, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
void cuMemcpy2DDevToDevAsyncAux(const char* file, int line,
                                const char* func_name, const T* input,
                                size_t rows, size_t cols, size_t src_pitch,
                                T* output, cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, src_pitch,
                           output, cols, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
void cuMemcpy2DDevToDevAsyncAux(const char* file, int line,
                                const char* func_name, const T* input,
                                size_t rows, size_t cols, T* output,
                                cudaStream_t stream = 0) noexcept {
    cuMemcpy2DAsyncGeneric(file, line, func_name, input, rows, cols, cols,
                           output, cols, cudaMemcpyDeviceToDevice, stream);
}

//==============================================================================
////////////////////////
//  cuMemcpyToSymbol  //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyToSymbolAsyncAux(const char* file, int line, const char* func_name,
                              const T& input, T& symbol,
                              cudaStream_t stream = 0) noexcept {
    cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, &input, sizeof(T),
                                             0, cudaMemcpyHostToDevice,
                                             stream),
                    "cudaMemcpyToSymbol", file, line, func_name);
}

template<typename T, int SIZE>
void cuMemcpyToSymbolAsyncAux(const char* file, int line, const char* func_name,
                              const T& input, T (&symbol)[SIZE],
                              cudaStream_t stream = 0) noexcept {
    cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, &input, sizeof(T),
                                             0, cudaMemcpyHostToDevice,
                                             stream),
                    "cudaMemcpyToSymbol", file, line, func_name);
}

//Pointer To Fixed Array
template<typename T, int SIZE>
void cuMemcpyToSymbolAsyncAux(const char* file, int line, const char* func_name,
                              const T* input, size_t num_items,
                              T (&symbol)[SIZE], size_t item_offset = 0,
                              cudaStream_t stream = 0) noexcept {
    assert(num_items + item_offset <= SIZE && input != nullptr);
    cudaErrorHandler(cudaMemcpyToSymbolAsync(symbol, input,
                                             num_items * sizeof(T),
                                             item_offset * sizeof(T), 0,
                                             cudaMemcpyHostToDevice, stream),
                     "cudaMemcpyToSymbol", file, line, func_name);
}

//==============================================================================
////////////////////////
// cuMemcpyFromSymbol //
////////////////////////

//Reference To Reference
template<typename T>
void cuMemcpyFromSymbolAsyncAux(const char* file, int line, const char* func_name,
                                const T& symbol, T& output,
                                cudaStream_t stream = 0) noexcept {
    cudaErrorHandler(cudaMemcpyFromSymbolAsync(&output, symbol, sizeof(T), 0,
                                               cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyFromSymbol", file, line, func_name);
}

template<typename T, int SIZE1, int SIZE2>
void cuMemcpyFromSymbolAsyncAux(const char* file, int line, const char* func_name,
                                const T (&symbol)[SIZE1], T (&output)[SIZE2],
                                cudaStream_t stream = 0) noexcept {
    assert(SIZE1 < SIZE2);
    cudaErrorHandler(cudaMemcpyFromSymbolAsync(&output, symbol,
                                               SIZE1 * sizeof(T), 0,
                                               cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyFromSymbol", file, line, func_name);
}


template<typename T, int SIZE1>
void cuMemcpyFromSymbolAsyncAux(const char* file, int line, const char* func_name,
                                const T (&symbol)[SIZE1], T* output,
                                cudaStream_t stream = 0) noexcept {
    assert(output != nullptr);
    cudaErrorHandler(cudaMemcpyFromSymbolAsync(output, symbol,
                                               SIZE1 * sizeof(T), 0,
                                               cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyFromSymbol", file, line, func_name);
}

///@endcond

} // namespace detail
} // namespace xlib
