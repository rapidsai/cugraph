/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#include <cmath>        //std::round
#include <iomanip>
#include <locale>
#include <sstream>
#include <type_traits>  //std::is_floating_point

namespace xlib {

template<typename T>
std::string format(T num, unsigned precision) noexcept {
    T round_num = !std::is_floating_point<T>::value ? num :
                  std::round(num * static_cast<T>(100)) / static_cast<T>(100);
    std::string title = std::to_string(round_num);
    auto     find_p = title.find('.');
    auto       init = find_p == std::string::npos ? title.size() : find_p;

    for (int i = static_cast<int>(init) - 3; i > 0; i -= 3)
        title.insert(static_cast<unsigned>(i), 1, ',');

    auto find_r = title.find('.');
    if (find_r != std::string::npos)
        title.erase(find_r + precision + 1);
    return title;
}

template<typename T, size_t SIZE>
void printArray(T (&array)[SIZE], const std::string& title,
                const std::string& sep) noexcept {
    printArray(array, SIZE, title, sep);
}

template<typename T>
void printArray(const T* array, size_t size, const std::string& title,
                const std::string& sep) noexcept {
    std::cout << title;
    if (size == 0)
        std::cout << "<empty>";
    for (size_t i = 0; i < size; i++)
        std::cout << array[i] << sep;
    std::cout << "\n" << std::endl;
}

template<>
void printArray<char>(const char* array, size_t size, const std::string& title,
                      const std::string& sep) noexcept;

template<>
void printArray<unsigned char>(const unsigned char* array, size_t size,
                               const std::string& title,
                               const std::string& sep) noexcept;

//------------------------------------------------------------------------------

template<typename T>
void printMatrix(T* const* matrix, size_t rows, size_t cols,
                 const std::string& title) noexcept {
    std::cout << title;
    for (int i = 0; i < rows; i++)
        printArray(matrix[i * cols], cols, "\n", '\t');
    std::cout << std::endl;
}

//==============================================================================

namespace detail {

template<typename T, typename Lambda>
void printMatrixAux(const T* matrix, size_t rows, size_t cols, size_t ld,
                    const std::string& title, const Lambda& indexing) noexcept {
    xlib::IosFlagSaver tmp;
    if (title != "")
        std::cout << title << "\n";

    auto max_width = new int[cols]();
    std::stringstream ss;
    ss.setf(std::cout.flags());
    ss.precision(std::cout.precision());
    ss.imbue(std::cout.getloc());

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ss << matrix[indexing(i, j, ld)];
            max_width[j] = std::max(max_width[j],
                                    static_cast<int>(ss.str().size()));
            ss.str("");
        }
    }
    std::cout << std::right;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << std::setw(max_width[j] + 2)
                      << matrix[indexing(i, j, ld)];
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] max_width;
}

} // namespace detail

template<typename T>
void printMatrix(const T* matrix, size_t rows, size_t cols,
                 const std::string& title) noexcept {
    detail::printMatrixAux(matrix, rows, cols, cols, title,
                     [](size_t i, size_t j, size_t ld) { return i * ld + j; });
}

template<typename T>
void printMatrix(const T* matrix, size_t rows, size_t cols, size_t ld,
                 const std::string& title) noexcept {
    detail::printMatrixAux(matrix, rows, cols, ld, title,
                      [](size_t i, size_t j, size_t ld) { return i * ld + j; });
}

template<typename T>
void printMatrixCM(const T* matrix, size_t rows, size_t cols,
                  const std::string& title) noexcept {
    detail::printMatrixAux(matrix, rows, cols, rows, title,
                 [](size_t i, size_t j, size_t ld) { return j * ld + i; });
}

template<typename T>
void printMatrixCM(const T* matrix, size_t rows, size_t cols, size_t ld,
                  const std::string& title) noexcept {
    detail::printMatrixAux(matrix, rows, cols, ld, title,
                   [](size_t i, size_t j, size_t ld) { return j * ld + i; });
}

//==============================================================================

template<typename T>
HOST_DEVICE
void printBits(T* array, int size) {
    const auto T_SIZE = static_cast<int>( sizeof(T) * 8u );
    using R = typename std::conditional<std::is_same<T, float>::value, unsigned,
              typename std::conditional<
                std::is_same<T, double>::value, uint64_t, T>::type>::type;

    for (int i = 0; i < size; i += T_SIZE) {
        for (int j = i; j < i + T_SIZE && j < size; j++) {
            auto array_value = reinterpret_cast<R&>( array[j / T_SIZE] );
            auto        mask = static_cast<R>( 1 << (j % T_SIZE) );
            int        value = ( array_value & mask ) ? 1 : 0;
            printf("%d", value);
        }
        printf(" ");
    }
    printf("\n");
}

template<typename T>
HOST_DEVICE void
printBits(const T& value) {
    const auto T_SIZE = sizeof(T) * 8;
    using R = typename std::conditional<sizeof(T) == sizeof(unsigned),
                                        unsigned,
              typename std::conditional<sizeof(T) == sizeof(uint64_t),
                                        uint64_t, T>::type>::type;

    char bits[T_SIZE + 1] = {};
    for (int i = 0; i < T_SIZE; i++) {
        bits[i] = reinterpret_cast<R>(value) & static_cast<R>(1 << i) ? '1' : '0';
    }
    printf("%s\n", bits);
}

} // namespace xlib
