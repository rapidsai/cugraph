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
#include "Host/Basic.hpp"
#include "Host/Numeric.hpp" //xlib::per_cent
#include <cassert>                  //assert
#include <cmath>                    //std::round
#include <iomanip>                  //std::setw

#if defined(__linux__)
    #include <fcntl.h>          //::open
    #include <sys/mman.h>       //::mmap
    #include <sys/stat.h>       //::open
    #include <sys/types.h>      //::open
    #include <unistd.h>         //::lseek
#endif

namespace xlib {

inline Progress::Progress(size_t total) noexcept :
                     _float_chunk(static_cast<double>(total - 1) / 100.0),
                     _total(total),
                     _next_chunk(static_cast<size_t>(_float_chunk)) {}

inline void Progress::next(size_t progress) noexcept {
    if (progress == 0 || _first) {
        std::cout << ((_next_chunk == 0) ? "   100%\n" : "     0%")
                  << std::flush;
        _first = false;
    }
    else if (progress == _next_chunk) {
        std::cout << "\b\b\b\b\b\b\b" << std::setw(6) << _to_print++
                  << "%" << std::flush;
        _next_chunk = static_cast<size_t>(
                                 static_cast<double>(_to_print) * _float_chunk);
        if (_to_print == 101)
            std::cout << std::endl;
    }
}

inline void Progress::per_cent(size_t progress) const noexcept {
    if (progress == 0 || _first) {
        std::cout << "     0%" << std::flush;
        _first = false;
        return;
    }
    std::cout << "\b\b\b\b\b\b\b" << std::setw(6)
           << std::round(xlib::per_cent(progress, _total)) << "%" << std::flush;
    if (progress == _total)
        std::cout << std::endl;
}

//==============================================================================
//==============================================================================

#if defined(__linux__)

inline MemoryMapped::MemoryMapped(const char* filename, size_t file_size,
                                  Enum mode, bool print) noexcept :
                                    _progress(file_size),
                                    _file_size(file_size),
                                    _print(print) {

    _fd = mode == READ ? ::open(filename, O_RDONLY, S_IRUSR) :
                ::open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (_fd == -1) ERROR("::open")

    if (::lseek(_fd, static_cast<off_t>(file_size - 1), SEEK_SET) == -1)
        ERROR("::lseek")
    if (mode == WRITE && ::write(_fd, "", 1) != 1)
        ERROR("::write")

    _mmap_ptr = static_cast<char*>(::mmap(nullptr, file_size,
                                          mode == READ ? PROT_READ : PROT_WRITE,
                                          MAP_SHARED, _fd, 0));
    if (_mmap_ptr == MAP_FAILED) ERROR("::mmap");
    if (::madvise(_mmap_ptr, file_size, MADV_SEQUENTIAL) == -1)
        ERROR("::madvise");
}

inline MemoryMapped::~MemoryMapped() noexcept {
    if (::munmap(_mmap_ptr, _file_size) == -1)
        ERROR("::munmap");
    if (::close(_fd) == -1)
        ERROR("::close");
    if (_partial != _file_size)
        ERROR("MemoryMapped: file partially read/write");
}

template<typename T, typename... Ts>
void MemoryMapped::read(T* data, size_t size, Ts... args) {
    if (_print)
        _progress.per_cent(_partial);
    std::copy(reinterpret_cast<T*>(_mmap_ptr + _partial),               //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial) + size, data); //NOLINT
    _partial += size * sizeof(T);
    assert(_partial <= _file_size);
    read(args...);
}

inline void MemoryMapped::read() const noexcept {
    if (_print)
        _progress.per_cent(_partial);
}

template<typename T, typename... Ts>
void MemoryMapped::read_noprint(T* data, size_t size, Ts... args) {
    std::copy(reinterpret_cast<T*>(_mmap_ptr + _partial),               //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial) + size, data); //NOLINT
    _partial += size * sizeof(T);
    assert(_partial <= _file_size);
    read_noprint(args...);
}

inline void MemoryMapped::read_noprint() const noexcept {}

//------------------------------------------------------------------------------

template<typename T, typename... Ts>
void MemoryMapped::write(const T* data, size_t size, Ts... args) {
    if (_print)
        _progress.per_cent(_partial);
    std::copy(data, data + size,                                        //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial));              //NOLINT
    _partial += size * sizeof(T);
    assert(_partial <= _file_size);
    write(args...);
}

inline void MemoryMapped::write() const noexcept {
    if (_print)
        _progress.per_cent(_partial);
}

template<typename T, typename... Ts>
void MemoryMapped::write_noprint(const T* data, size_t size, Ts... args) {
    std::copy(data, data + size,                                        //NOLINT
              reinterpret_cast<T*>(_mmap_ptr + _partial));              //NOLINT
    _partial += size * sizeof(T);
    assert(_partial <= _file_size);
    write_noprint(args...);
}

inline void MemoryMapped::write_noprint() const noexcept {}

#endif

} // namespace xlib
