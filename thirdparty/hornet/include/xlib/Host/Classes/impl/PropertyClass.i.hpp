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
#include "Host/Basic.hpp"   //ERROR

namespace xlib {

template<typename Enum, typename CRTP>
PropertyClass<Enum, CRTP>::PropertyClass(const Enum& value) noexcept :
                        _state(static_cast<uint64_t>(value)) {
    if (_is_non_compatible())
        ERROR("Incompatible Enumerator")
}

template<typename Enum, typename CRTP>
CRTP PropertyClass<Enum, CRTP>::operator| (const CRTP& obj) const noexcept {
    return CRTP(static_cast<Enum>(_state | obj._state));
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::operator& (const CRTP& obj)  const noexcept {
    return static_cast<bool>(_state & obj._state);
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::operator== (const CRTP& obj) const noexcept {
    return _state == obj._state;
}

template<typename Enum, typename CRTP>
void PropertyClass<Enum, CRTP>::operator+= (const CRTP& obj) noexcept {
    _state |= obj._state;
    if (_is_non_compatible())
        ERROR("Incompatible Enumerator")
}

template<typename Enum, typename CRTP>
void PropertyClass<Enum, CRTP>::operator-= (const CRTP& obj) noexcept {
    _state &= ~obj._state;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::operator!= (const CRTP& obj) const noexcept {
    return _state != obj._state;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::is_undefined() const noexcept {
    return _state == 0;
}

template<typename Enum, typename CRTP>
bool PropertyClass<Enum, CRTP>::_is_non_compatible() const noexcept {
    return false;
}

} // namespace xlib
