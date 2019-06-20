/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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
 *   - **fast debbuging**:
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

/**
 * @brief Allocate device memory
 * @tparam T Input type
 * @param[out] d_pointer device pointer to allocate
 * @param[in]  num_items number of items of size T to allocate
 * @warning `num_items` > 0
 */
template<typename T>
void cuMalloc(T*& d_pointer, size_t num_items = 1);

/**
 * @brief Free device memory
 * @tparam TArgs Input types list
 * @param[in] d_pointers list of device pointers to deallocate
 */
template<typename... TArgs>
void cuFree(TArgs... d_pointers);

/**
 * @brief Set a given number of items of the device pointer to zero
 * @tparam T Input type
 * @param[out] d_pointer device pointer to write
 * @param[in]  num_items number of items to write
 * @warning `num_items` > 0 and `d_pointer` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemset0x00(T* d_pointer, size_t num_items);

/**
 * @brief Set a given number of items of the device pointer to all ones
 * @tparam T Input type
 * @param[out] d_pointer device pointer to write
 * @param[in]  num_items number of items to write
 * @warning `num_items` > 0 and `d_pointer` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemset0xFF(T* d_pointer, size_t num_items);
//------------------------------------------------------------------------------

/**
 * @brief Copy host memory (pointer) to device global memory
 * @tparam T Input/Output type
 * @param[in] h_input host pointer to read
 * @param[in] num_items number of items to copy
 * @param[out] d_output device pointer to write
 * @warning `num_items` > 0 and `h_input` \f$\ne\f$ `nullptr` and
 *          `d_output` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemcpyToDevice(const T* h_input, size_t num_items, T* d_output);

/**
 * @brief Asynchronous copy host memory (pointer) to device global memory
 * @tparam T Input/Output type
 * @param[in] h_input host pointer to read
 * @param[in] num_items number of items to copy
 * @param[out] d_output device pointer to write
 * @warning  `num_items` > 0 and `h_input` \f$\ne\f$ `nullptr` and
 *           `d_output` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemcpyToDeviceAsync(const T* h_input, size_t num_items, T* d_output);

/**
 * @brief Copy device global memory (pointer) to host memory (pointer)
 * @tparam T Input/Output type
 * @param[in] d_input device pointer to read
 * @param[in] num_items number of items to copy
 * @param[out] h_output host pointer to write
 * @warning  `num_items` > 0 and `d_input` \f$\ne\f$ `nullptr` and
 *           `h_output` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemcpyToHost(const T* d_input, size_t num_items, T* h_output);
//------------------------------------------------------------------------------

/**
 * @brief Copy host memory (reference) to device symbol
 * @tparam T Input/Output type
 * @param[in] h_input host reference to read
 * @param[out] symbol device symbol to write
 */
template<typename T>
void cuMemcpyToSymbol(const T& h_input, T& symbol);

/**
 * @brief Copy host memory (pointer) to device symbol
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] h_input host pointer to read
 * @param[in] num_items number of items to copy
 * @param[out] symbol device symbol to write
 * @param[in] item_offset offset from start of symbol
 * @warning  `num_items` > 0 and `d_input` \f$\ne\f$ `nullptr` and
 *           `num_items + item_offset` \f$\le\f$ `SIZE`
 */
template<typename T, unsigned SIZE>
void cuMemcpyToSymbol(const T* h_input, size_t num_items, T (&symbol)[SIZE],
                      size_t item_offset = 0);

/**
 * @brief Copy host memory (pointer) to device symbol
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] h_input host pointer to read
 * @param[out] symbol device symbol to write
 * @warning  `d_input` \f$\ne\f$ `nullptr`
 */
template<typename T, unsigned SIZE>
void cuMemcpyToSymbol(const T* h_input, T (&symbol)[SIZE]);

/**
 * @brief Asynchronous copy host memory (reference) to device symbol
 * @tparam T Input/Output type
 * @param[in] h_input host reference to read
 * @param[out] symbol device symbol to write
 */
template<typename T>
void cuMemcpyToSymbolAsync(const T& h_input, T& symbol);


/**
 * @brief Asynchronous copy host memory (pointer) to device symbol
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] h_input host pointer to read
 * @param[in] num_items number of items to copy
 * @param[out] symbol device symbol to write
 * @param[in] item_offset offset from start of symbol
 * @warning  `num_items` > 0 and `d_input` \f$\ne\f$ `nullptr` and
 *           `num_items + item_offset` \f$\le\f$ `SIZE`
 */
template<typename T>
void cuMemcpyToSymbolAsync(const T* h_input, size_t num_items,
                           T (&symbol)[SIZE], size_t item_offset = 0);

/**
 * @brief Copy device symbol to device symbol host memory (reference)
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] symbol device symbol to read
 * @param[out] h_output host reference to write
 * @param[in] item_offset offset from start of symbol
 */
template<typename T>
void cuMemcpyFromSymbol(const T(&symbol)[SIZE], T& h_output,
                        size_t item_offset = 0);

/**
 * @brief Copy device symbol to device symbol host memory (pointer)
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] symbol device symbol to read
 * @param[in] num_items number of items to copy
 * @param[out] h_output host pointer to write
 * @param[in] item_offset offset from start of symbol
 * @warning  `num_items` > 0 and `h_output` \f$\ne\f$ `nullptr` and
 *           `num_items + item_offset` \f$\le\f$ `SIZE`
 */
template<typename T>
void cuMemcpyFromSymbol(const T(&symbol)[SIZE], size_t num_items, T* h_output,
                        size_t item_offset = 0);

/**
 * @brief Copy device symbol to device symbol host memory (pointer)
 * @tparam T Input/Output type
 * @tparam SIZE device symbol size
 * @param[in] symbol device symbol to read
 * @param[out] h_output host pointer to write
 * @warning `h_output` \f$\ne\f$ `nullptr`
 */
template<typename T>
void cuMemcpyFromSymbol(const T(&symbol)[SIZE], T* h_output);
