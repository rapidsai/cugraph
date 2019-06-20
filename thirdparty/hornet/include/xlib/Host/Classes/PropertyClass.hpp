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

#include <cstdint>  //uint64_t

namespace xlib {

/**
 * @brief Strongly Typed Enumeraton Class with support for bitwise operators
 * @tparam Enum Enumerator Class (Strongly Typed)
 * @tparam CRTP (Curiously Recurring Template Pattern) the base class itself
 */
template<typename Enum, typename CRTP>
class PropertyClass {
public:
    /**
     * @brief Default Constructor
     * @details It inherited by default in derived class and is used with
     *          no explicit initialization is provided
     */
    explicit PropertyClass() noexcept = default;

    /**
     * @brief Bitwise OR between the current object and the object passed as
     *        parameter
     * @param[in] obj Object on which perform the bitwise OR
     * @return New intance of the derived class with state equal to bitwise OR
     *         between the current object and @p obj
     */
    virtual CRTP  operator|  (const CRTP& obj) const noexcept final;

    /**
     * @brief Bitwise AND between the current object and the object passed as
     *        parameter
     * @param[in] obj Object on which perform the bitwise AND
     * @return `true` if at least one enumerator value is common between the
     *         two instances, `false` otherwise
     */
    virtual bool  operator&  (const CRTP& obj) const noexcept final;

    /**
     * @brief Equal comparison
     * @param[in] obj Object on which perform the equal comparison
     * @return `true` if all enumerator values are common between the
     *         two instances, `false` otherwise
     */
    virtual bool  operator== (const CRTP& obj) const noexcept final;

    /**
     * @brief Not equal comparison
     * @param[in] obj Object on which perform the not equal comparison
     * @return `true` if at least one enumerator value is different between the
     *         two instances, `false` otherwise
     */
    virtual bool  operator!= (const CRTP& obj) const noexcept final;

    /**
     * @brief Update the current object by adding the enumerator values of the
     *        object passed as parameter
     * @param[in] obj Object used to add the enumerator values
     */
    virtual void  operator+= (const CRTP& obj) noexcept final;

    /**
     * @brief Update the current object by subtracting the enumerator values
     *        of the object passed as parameter
     * @param[in] obj Object used to add the enumerator values
     */
    virtual void  operator-= (const CRTP& obj) noexcept final;

    /**
     * @brief Test if undefined
     * @return `true` if the current state is equal to the default value,
     *         `false` otherwise
     */
    virtual bool is_undefined() const noexcept final;

protected:
    ///@brief Internal State
    uint64_t _state { 0 };

    /**
     * @brief Main Construcor. It allows to construt the object by starting from
     *        a Strongly type enumerator
     * @param[in] value Strongly type enumerator
     */
    explicit PropertyClass(const Enum& value) noexcept;

    /**
     * @brief Check for incompatible enumerator values
     * @return `true` if the current state is not compatible,
     *         `false` otherwise
     */
    virtual bool _is_non_compatible() const noexcept;
};

} // namespace xlib

#include "impl/PropertyClass.i.hpp"
