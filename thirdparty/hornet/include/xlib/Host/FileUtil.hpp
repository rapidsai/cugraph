/**
 * @internal
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
 *
 * @file
 */
#pragma once

#include <cstddef>  //size_t
#include <fstream>  //std::ifstream
#include <string>   //std::string

namespace xlib {

class Progress {
public:
    explicit Progress(size_t total)    noexcept;
    void     next    (size_t progress) noexcept;
    void     per_cent(size_t progress) const noexcept;
private:
    const double _float_chunk;
    const size_t _total;
    size_t       _next_chunk;
    int          _to_print   { 1 };
    mutable bool _first      { true };

};

#if defined(__linux__)

class MemoryMapped {
public:
    enum Enum { READ, WRITE };
    explicit MemoryMapped(const char* filename, size_t file_size, Enum mode,
                          bool print = false) noexcept;
    ~MemoryMapped() noexcept;

    template<typename T, typename... Ts>
    void read(T* data, size_t size, Ts... args);

    template<typename T, typename... Ts>
    void read_noprint(T* data, size_t size, Ts... args);

    template<typename T, typename... Ts>
    void write(const T* data, size_t size, Ts... args);

    template<typename T, typename... Ts>
    void write_noprint(const T* data, size_t size, Ts... args);

private:
    void read() const noexcept;
    void read_noprint() const noexcept;

    void write() const noexcept;
    void write_noprint() const noexcept;

    Progress _progress;
    char*    _mmap_ptr   { nullptr };
    size_t   _partial    { 0 };
    size_t   _file_size  { 0 };
    int      _fd         { 0 };
    bool     _print      { false };
};

#endif

void        check_regular_file(const char* filename);
void        check_regular_file(std::ifstream& fin, const char* filename = "");
size_t      file_size(const char* filename);
size_t      file_size(std::ifstream& fin);

std::string extract_filename            (const std::string& str) noexcept;
std::string extract_file_extension      (const std::string& str) noexcept;
std::string extract_filepath            (const std::string& str) noexcept;
std::string extract_filepath_noextension(const std::string& str) noexcept;
void        skip_lines(std::istream& fin, int num_lines = 1);
void        skip_words(std::istream& fin, int num_words = 1);

} // namespace xlib

#include "impl/FileUtil.i.hpp"
