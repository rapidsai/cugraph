/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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
 */

namespace std {

inline char2 numeric_limits<char2>::max() noexcept {
    return make_char2(std::numeric_limits<char>::max(),
                      std::numeric_limits<char>::max());
}

inline char2 numeric_limits<char2>::min() noexcept {
    return make_char2(std::numeric_limits<char>::min(),
                      std::numeric_limits<char>::min());
}

inline char2 numeric_limits<char2>::lowest() noexcept {
    return make_char2(std::numeric_limits<char>::lowest(),
                      std::numeric_limits<char>::lowest());
}

//------------------------------------------------------------------------------

inline uchar2 numeric_limits<uchar2>::max() noexcept {
    return make_uchar2(std::numeric_limits<unsigned char>::max(),
                       std::numeric_limits<unsigned char>::max());
}

inline uchar2 numeric_limits<uchar2>::min() noexcept {
    return make_uchar2(std::numeric_limits<unsigned char>::min(),
                       std::numeric_limits<unsigned char>::min());
}

inline uchar2 numeric_limits<uchar2>::lowest() noexcept {
    return make_uchar2(std::numeric_limits<unsigned char>::lowest(),
                       std::numeric_limits<unsigned char>::lowest());
}

//------------------------------------------------------------------------------

inline char4 numeric_limits<char4>::max() noexcept {
    return make_char4(std::numeric_limits<char>::max(),
                      std::numeric_limits<char>::max(),
                      std::numeric_limits<char>::max(),
                      std::numeric_limits<char>::max());
}

inline char4 numeric_limits<char4>::min() noexcept {
    return make_char4(std::numeric_limits<char>::min(),
                      std::numeric_limits<char>::min(),
                      std::numeric_limits<char>::min(),
                      std::numeric_limits<char>::min());
}

inline char4 numeric_limits<char4>::lowest() noexcept {
    return make_char4(std::numeric_limits<char>::lowest(),
                      std::numeric_limits<char>::lowest(),
                      std::numeric_limits<char>::lowest(),
                      std::numeric_limits<char>::lowest());
}

//------------------------------------------------------------------------------

inline uchar4 numeric_limits<uchar4>::max() noexcept {
    return make_uchar4(std::numeric_limits<unsigned char>::max(),
                       std::numeric_limits<unsigned char>::max(),
                       std::numeric_limits<unsigned char>::max(),
                       std::numeric_limits<unsigned char>::max());
}

inline uchar4 numeric_limits<uchar4>::min() noexcept {
    return make_uchar4(std::numeric_limits<unsigned char>::min(),
                       std::numeric_limits<unsigned char>::min(),
                       std::numeric_limits<unsigned char>::min(),
                       std::numeric_limits<unsigned char>::min());
}

inline uchar4 numeric_limits<uchar4>::lowest() noexcept {
    return make_uchar4(std::numeric_limits<unsigned char>::lowest(),
                       std::numeric_limits<unsigned char>::lowest(),
                       std::numeric_limits<unsigned char>::lowest(),
                       std::numeric_limits<unsigned char>::lowest());
}

//------------------------------------------------------------------------------

inline short2 numeric_limits<short2>::max() noexcept {
    return make_short2(std::numeric_limits<short>::max(),
                       std::numeric_limits<short>::max());
}

inline short2 numeric_limits<short2>::min() noexcept {
    return make_short2(std::numeric_limits<short>::min(),
                       std::numeric_limits<short>::min());
}

inline short2 numeric_limits<short2>::lowest() noexcept {
    return make_short2(std::numeric_limits<short>::lowest(),
                       std::numeric_limits<short>::lowest());
}

//------------------------------------------------------------------------------

inline ushort2 numeric_limits<ushort2>::max() noexcept {
    return make_ushort2(std::numeric_limits<unsigned short>::max(),
                        std::numeric_limits<unsigned short>::max());
}

inline ushort2 numeric_limits<ushort2>::min() noexcept {
    return make_ushort2(std::numeric_limits<unsigned short>::min(),
                        std::numeric_limits<unsigned short>::min());
}

inline ushort2 numeric_limits<ushort2>::lowest() noexcept {
    return make_ushort2(std::numeric_limits<unsigned short>::lowest(),
                        std::numeric_limits<unsigned short>::lowest());
}

//------------------------------------------------------------------------------

inline short4 numeric_limits<short4>::max() noexcept {
    return make_short4(std::numeric_limits<short>::max(),
                       std::numeric_limits<short>::max(),
                       std::numeric_limits<short>::max(),
                       std::numeric_limits<short>::max());
}

inline short4 numeric_limits<short4>::min() noexcept {
    return make_short4(std::numeric_limits<short>::min(),
                       std::numeric_limits<short>::min(),
                       std::numeric_limits<short>::min(),
                       std::numeric_limits<short>::min());
}

inline short4 numeric_limits<short4>::lowest() noexcept {
    return make_short4(std::numeric_limits<short>::lowest(),
                       std::numeric_limits<short>::lowest(),
                       std::numeric_limits<short>::lowest(),
                       std::numeric_limits<short>::lowest());
}

//------------------------------------------------------------------------------

inline ushort4 numeric_limits<ushort4>::max() noexcept {
    return make_ushort4(std::numeric_limits<unsigned short>::max(),
                        std::numeric_limits<unsigned short>::max(),
                        std::numeric_limits<unsigned short>::max(),
                        std::numeric_limits<unsigned short>::max());
}

inline ushort4 numeric_limits<ushort4>::min() noexcept {
    return make_ushort4(std::numeric_limits<unsigned short>::min(),
                        std::numeric_limits<unsigned short>::min(),
                        std::numeric_limits<unsigned short>::min(),
                        std::numeric_limits<unsigned short>::min());
}

inline ushort4 numeric_limits<ushort4>::lowest() noexcept {
    return make_ushort4(std::numeric_limits<unsigned short>::lowest(),
                        std::numeric_limits<unsigned short>::lowest(),
                        std::numeric_limits<unsigned short>::lowest(),
                        std::numeric_limits<unsigned short>::lowest());
}

//------------------------------------------------------------------------------

inline int2 numeric_limits<int2>::max() noexcept {
    return make_int2(std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max());
}

inline int2 numeric_limits<int2>::min() noexcept {
    return make_int2(std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::min());
}

inline int2 numeric_limits<int2>::lowest() noexcept {
    return make_int2(std::numeric_limits<int>::lowest(),
                     std::numeric_limits<int>::lowest());
}

//------------------------------------------------------------------------------

inline uint2 numeric_limits<uint2>::max() noexcept {
    return make_uint2(std::numeric_limits<unsigned>::max(),
                      std::numeric_limits<unsigned>::max());
}

inline uint2 numeric_limits<uint2>::min() noexcept {
    return make_uint2(std::numeric_limits<unsigned>::min(),
                      std::numeric_limits<unsigned>::min());
}

inline uint2 numeric_limits<uint2>::lowest() noexcept {
    return make_uint2(std::numeric_limits<unsigned>::lowest(),
                      std::numeric_limits<unsigned>::lowest());
}

//------------------------------------------------------------------------------

inline int4 numeric_limits<int4>::max() noexcept {
    return make_int4(std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max());
}

inline int4 numeric_limits<int4>::min() noexcept {
    return make_int4(std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::min());
}

inline int4 numeric_limits<int4>::lowest() noexcept {
    return make_int4(std::numeric_limits<int>::lowest(),
                     std::numeric_limits<int>::lowest(),
                     std::numeric_limits<int>::lowest(),
                     std::numeric_limits<int>::lowest());
}

//------------------------------------------------------------------------------

inline uint4 numeric_limits<uint4>::max() noexcept {
    return make_uint4(std::numeric_limits<unsigned>::max(),
                      std::numeric_limits<unsigned>::max(),
                      std::numeric_limits<unsigned>::max(),
                      std::numeric_limits<unsigned>::max());
}

inline uint4 numeric_limits<uint4>::min() noexcept {
    return make_uint4(std::numeric_limits<unsigned>::min(),
                      std::numeric_limits<unsigned>::min(),
                      std::numeric_limits<unsigned>::min(),
                      std::numeric_limits<unsigned>::min());
}

inline uint4 numeric_limits<uint4>::lowest() noexcept {
    return make_uint4(std::numeric_limits<unsigned>::lowest(),
                      std::numeric_limits<unsigned>::lowest(),
                      std::numeric_limits<unsigned>::lowest(),
                      std::numeric_limits<unsigned>::lowest());
}

//------------------------------------------------------------------------------

inline longlong2 numeric_limits<longlong2>::max() noexcept {
    return make_longlong2(std::numeric_limits<long long>::max(),
                          std::numeric_limits<long long>::max());
}

inline longlong2 numeric_limits<longlong2>::min() noexcept {
    return make_longlong2(std::numeric_limits<long long>::min(),
                          std::numeric_limits<long long>::min());
}

inline longlong2 numeric_limits<longlong2>::lowest() noexcept {
    return make_longlong2(std::numeric_limits<long long>::lowest(),
                          std::numeric_limits<long long>::lowest());
}

//------------------------------------------------------------------------------

inline ulonglong2 numeric_limits<ulonglong2>::max() noexcept {
    return make_ulonglong2(std::numeric_limits<long long unsigned>::max(),
                           std::numeric_limits<long long unsigned>::max());
}

inline ulonglong2 numeric_limits<ulonglong2>::min() noexcept {
    return make_ulonglong2(std::numeric_limits<long long unsigned>::min(),
                           std::numeric_limits<long long unsigned>::min());
}

inline ulonglong2 numeric_limits<ulonglong2>::lowest() noexcept {
    return make_ulonglong2(std::numeric_limits<long long unsigned>::lowest(),
                           std::numeric_limits<long long unsigned>::lowest());
}

//------------------------------------------------------------------------------

inline float2 numeric_limits<float2>::max() noexcept {
    return make_float2(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
}

inline float2 numeric_limits<float2>::min() noexcept {
    return make_float2(std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min());
}

inline float2 numeric_limits<float2>::lowest() noexcept {
    return make_float2(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());
}

//------------------------------------------------------------------------------

inline float4 numeric_limits<float4>::max() noexcept {
    return make_float4(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
}

inline float4 numeric_limits<float4>::min() noexcept {
    return make_float4(std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::min());
}

inline float4 numeric_limits<float4>::lowest() noexcept {
    return make_float4(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());
}

//------------------------------------------------------------------------------

inline double2 numeric_limits<double2>::max() noexcept {
    return make_double2(std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max());
}

inline double2 numeric_limits<double2>::min() noexcept {
    return make_double2(std::numeric_limits<double>::min(),
                        std::numeric_limits<double>::min());
}

inline double2 numeric_limits<double2>::lowest() noexcept {
    return make_double2(std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::lowest());
}

} // namespace std

//==============================================================================

inline std::ostream& operator<< (std::ostream& out, const char2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const uchar2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const char4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const uchar4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const short2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const ushort2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const short4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const ushort4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const int2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const uint2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const int4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const uint4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const longlong2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulonglong2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator << (std::ostream& out, const float2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const float4& value) {
    out << "(" << value.x << "," << value.y << "," << value.z << ","
        << value.w << ")";
    return out;
}

inline std::ostream& operator << (std::ostream& out, const double2& value) {
    out << "(" << value.x << "," << value.y << ")";
    return out;
}

//==============================================================================

HOST_DEVICE bool operator== (const char2& A, const char2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const char2& A, const char2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const char2& A, const char2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const char2& A, const char2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const char2& A, const char2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const char2& A, const char2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const uchar2& A, const uchar2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const uchar2& A, const uchar2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const uchar2& A, const uchar2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const uchar2& A, const uchar2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const uchar2& A, const uchar2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const uchar2& A, const uchar2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const short2& A, const short2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const short2& A, const short2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const short2& A, const short2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const short2& A, const short2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const short2& A, const short2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const short2& A, const short2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const ushort2& A, const ushort2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const ushort2& A, const ushort2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const ushort2& A, const ushort2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const ushort2& A, const ushort2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const ushort2& A, const ushort2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const ushort2& A, const ushort2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const int2& A, const int2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const int2& A, const int2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const int2& A, const int2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const int2& A, const int2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const int2& A, const int2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const int2& A, const int2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const uint2& A, const uint2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const uint2& A, const uint2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const uint2& A, const uint2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const uint2& A, const uint2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const uint2& A, const uint2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const uint2& A, const uint2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const float2& A, const float2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const float2& A, const float2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const float2& A, const float2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const float2& A, const float2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const float2& A, const float2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const float2& A, const float2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const longlong2& A, const longlong2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const longlong2& A, const longlong2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const longlong2& A, const longlong2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const longlong2& A, const longlong2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const longlong2& A, const longlong2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const longlong2& A, const longlong2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const ulonglong2& A, const ulonglong2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const ulonglong2& A, const ulonglong2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const ulonglong2& A, const ulonglong2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const ulonglong2& A, const ulonglong2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const ulonglong2& A, const ulonglong2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const ulonglong2& A, const ulonglong2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const double2& A, const double2& B) {
    return A.x == B.x && A.y == B.y;
}

HOST_DEVICE bool operator!= (const double2& A, const double2& B) {
    return A.x != B.x || A.y != B.y;
}

HOST_DEVICE bool operator< (const double2& A, const double2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

HOST_DEVICE bool operator<= (const double2& A, const double2& B) {
    return A.x <= B.x && A.y <= B.y;
}

HOST_DEVICE bool operator>= (const double2& A, const double2& B) {
    return A.x >= B.x && A.y >= B.y;
}

HOST_DEVICE bool operator> (const double2& A, const double2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const char4& A, const char4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const char4& A, const char4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const char4& A, const char4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const char4& A, const char4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const char4& A, const char4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const char4& A, const char4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const uchar4& A, const uchar4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const uchar4& A, const uchar4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const uchar4& A, const uchar4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const uchar4& A, const uchar4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const uchar4& A, const uchar4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const uchar4& A, const uchar4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const short4& A, const short4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const short4& A, const short4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const short4& A, const short4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const short4& A, const short4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const short4& A, const short4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const short4& A, const short4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const ushort4& A, const ushort4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const ushort4& A, const ushort4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const ushort4& A, const ushort4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const ushort4& A, const ushort4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const ushort4& A, const ushort4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const ushort4& A, const ushort4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const int4& A, const int4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const int4& A, const int4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const int4& A, const int4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const int4& A, const int4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const int4& A, const int4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const int4& A, const int4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const uint4& A, const uint4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const uint4& A, const uint4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const uint4& A, const uint4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const uint4& A, const uint4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const uint4& A, const uint4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const uint4& A, const uint4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

//------------------------------------------------------------------------------

HOST_DEVICE bool operator== (const float4& A, const float4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

HOST_DEVICE bool operator!= (const float4& A, const float4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

HOST_DEVICE bool operator< (const float4& A, const float4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

HOST_DEVICE bool operator<= (const float4& A, const float4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

HOST_DEVICE bool operator>= (const float4& A, const float4& B) {
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}

HOST_DEVICE bool operator> (const float4& A, const float4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}

// =============================================================================

namespace xlib {

template<>
struct Make2Str<char> {
    using type = char2;

    __host__ __device__ __forceinline__
    static type get(char a, char b) {
        return make_char2(a, b);
    }
};

template<>
struct Make2Str<unsigned char> {
    using type = uchar2;

    __host__ __device__ __forceinline__
    static type get(unsigned char a, unsigned char b) {
        return make_uchar2(a, b);
    }
};

template<>
struct Make2Str<short> {
    using type = short2;

    __host__ __device__ __forceinline__
    static type get(short a, short b) {
        return make_short2(a, b);
    }
};

template<>
struct Make2Str<unsigned short> {
    using type = ushort2;

    __host__ __device__ __forceinline__
    static type get(unsigned short a, unsigned short b) {
        return make_ushort2(a, b);
    }
};

template<>
struct Make2Str<int> {
    using type = int2;

    __host__ __device__ __forceinline__
    static type get(int a, int b) {
        return make_int2(a, b);
    }
};

template<>
struct Make2Str<unsigned> {
    using type = float2;

    __host__ __device__ __forceinline__
    static type get(unsigned a, unsigned b) {
        return make_float2(a, b);
    }
};

template<>
struct Make2Str<long long> {
    using type = longlong2;

    __host__ __device__ __forceinline__
    static type get(long long a, long long b) {
        return make_longlong2(a, b);
    }
};

template<>
struct Make2Str<long long unsigned> {
    using type = ulonglong2;

    __host__ __device__ __forceinline__
    static type get(long long unsigned a, long long unsigned b) {
        return make_ulonglong2(a, b);
    }
};

template<>
struct Make2Str<float> {
    using type = float2;

    __host__ __device__ __forceinline__
    static type get(float a, float b) {
        return make_float2(a, b);
    }
};

template<> struct Make2Str<double> {
    using type = double2;

    __host__ __device__ __forceinline__
    static type get(double a, double b) {
        return make_double2(a, b);
    }
};

//------------------------------------------------------------------------------

template<>
struct Make4Str<char> {
    using type = char4;

    __host__ __device__ __forceinline__
    static type get(char a, char b, char c, char d) {
        return make_char4(a, b, c, d);
    }
};

template<>
struct Make4Str<unsigned char> {
    using type = uchar4;

    __host__ __device__ __forceinline__
    static type get(unsigned char a, unsigned char b,
                    unsigned char c, unsigned char d) {
        return make_uchar4(a, b, c, d);
    }
};

template<>
struct Make4Str<short> {
    using type = short4;

    __host__ __device__ __forceinline__
    static type get(short a, short b, short c, short d) {
        return make_short4(a, b, c, d);
    }
};

template<>
struct Make4Str<unsigned short> {
    using type = ushort4;

    __host__ __device__ __forceinline__
    static type get(unsigned short a, unsigned short b,
                    unsigned short c, unsigned short d) {
        return make_ushort4(a, b, c, d);
    }
};

template<>
struct Make4Str<int> {
    using type = int4;

    __host__ __device__ __forceinline__
    static type get(int a, int b, int c, int d) {
        return make_int4(a, b, c, d);
    }
};

template<>
struct Make4Str<unsigned> {
    using type = uint4;

    __host__ __device__ __forceinline__
    static type get(unsigned a, unsigned b, unsigned c, unsigned d) {
        return make_uint4(a, b, c, d);
    }
};

template<>
struct Make4Str<float> {
    using type = float4;

    __host__ __device__ __forceinline__
    static type get(float a, float b, float c, float d) {
        return make_float4(a, b, c, d);
    }
};

//==============================================================================

template<typename T>
__host__ __device__ __forceinline__
typename Make2Str<T>::type make2(T a, T b) {
    return Make2Str<T>::get(a, b);
}

template<typename T>
__host__ __device__ __forceinline__
typename Make2Str<T>::type make4(T a, T b, T c, T d) {
    return Make4Str<T>::get(a, b, c, d);
}

} // namespace xlib
