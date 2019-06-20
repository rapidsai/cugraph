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

#include <chrono>               //std::chrono::duration
#include <string>               //std::string
#if defined(__linux__)
    #include <sys/times.h>      //::tms
#endif

#define COLOR

#if defined(COLOR)
    #include "Host/PrintExt.hpp"
#else

namespace xlib {
    enum class Color { FG_DEFAULT };
    struct IosFlagSaver {};
} // namespace xlib

inline std::ostream& operator<<(std::ostream& os, Color mod) noexcept {
    return os;
}

#endif

namespace timer {

/// @brief chrono precision : microseconds
using   micro = typename std::chrono::duration<float, std::micro>;
/// @brief default chrono precision (milliseconds)
using   milli = typename std::chrono::duration<float, std::milli>;
/// @brief chrono precision : seconds
using seconds = typename std::chrono::duration<float, std::ratio<1>>;
/// @brief chrono precision : minutes
using minutes = typename std::chrono::duration<float, std::ratio<60>>;
/// @brief chrono precision : minutes
using   hours = typename std::chrono::duration<float, std::ratio<3600>>;

/**
 * @brief timer types
 */
enum timer_type {  HOST = 0       /// Wall (real) clock host time
                 , CPU  = 1       /// CPU User time
            #if defined(__linux__)
                 , SYS  = 2       /// User/Kernel/System time
            #endif
                 , DEVICE = 3     /// GPU device time
};

namespace detail {

/**
 * @brief Timer class
 * @tparam type Timer type (default = HOST)
 * @tparam ChronoPrecision time precision
 */
template<timer_type type, typename ChronoPrecision>
class TimerBase {
    template<typename>
    struct is_duration : std::false_type {};

    template<typename T, typename R>
    struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

    static_assert(is_duration<ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
public:
    /**
     * @brief Default costructor
     * @param[in] decimals precision to print the time elapsed
     * @param[in] space space for the left alignment
     * @param[in] color color of print
     */
    explicit TimerBase(int decimals, int space, xlib::Color color) noexcept;

    virtual ~TimerBase() noexcept = default;

    /**
     * @brief Start the timer
     */
    virtual void start() noexcept = 0;

    /**
     * @brief Stop the timer
     */
    virtual void stop() noexcept = 0;

    /**
     * @brief Get the time elapsed between start() and stop() calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float duration() const noexcept final;

    /**
     * @brief Get the time elapsed between the first start() and the last stop()
     *        calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float total_duration() const noexcept final;

    /**
     * @brief Get the average time elapsed between the first start() and the
     *        last stop() calls
     * @return average duration
     */
    virtual float average() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float std_deviation() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float min() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float max() const noexcept final;

    /**
     *
     */
    virtual void reset() noexcept final;

    /**
     * @brief Print the time elapsed between start() and stop() calls
     * @param[in] str print string \p str before the time elapsed
     * @warning if start() and stop() not invoked undefined behavior
     */
    virtual void print(const std::string& str) const noexcept;

    virtual void printAll(const std::string& str) const noexcept;

protected:
    ChronoPrecision _time_elapsed    {};
    const int         _space         { 0 };
    const int         _decimals      { 0 };
    const xlib::Color _default_color { xlib::Color::FG_DEFAULT };
    bool              _start_flag    { false };

    /**
     *
     */
    virtual void register_time() noexcept final;

private:
    ChronoPrecision   _time_squared       {};
    ChronoPrecision   _total_time_elapsed {};
    ChronoPrecision   _time_min           {};
    ChronoPrecision   _time_max           {};
    int               _num_executions     { 0 };
};

} // namespace detail
//------------------------------------------------------------------------------

template<timer_type type, typename ChronoPrecision = milli>
class Timer;

template<typename ChronoPrecision>
class Timer<HOST, ChronoPrecision> final :
            public timer::detail::TimerBase<HOST, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

private:
    using timer::detail::TimerBase<HOST, ChronoPrecision>::_time_elapsed;
    using timer::detail::TimerBase<HOST, ChronoPrecision>::_start_flag;

    std::chrono::system_clock::time_point _start_time {};
    std::chrono::system_clock::time_point _stop_time  {};

    using timer::detail::TimerBase<HOST, ChronoPrecision>::register_time;
};

//------------------------------------------------------------------------------

template<typename ChronoPrecision>
class Timer<CPU, ChronoPrecision> final :
            public timer::detail::TimerBase<CPU, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

private:
    using timer::detail::TimerBase<CPU, ChronoPrecision>::_time_elapsed;
    using timer::detail::TimerBase<CPU, ChronoPrecision>::_start_flag;

    std::clock_t _start_clock { 0 };
    std::clock_t _stop_clock  { 0 };

    using timer::detail::TimerBase<CPU, ChronoPrecision>::register_time;
};

//------------------------------------------------------------------------------

#if defined(__linux__)

template<typename ChronoPrecision>
class Timer<SYS, ChronoPrecision> final :
            public timer::detail::TimerBase<SYS, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

    void print(const std::string& str = "") const noexcept override;

private:
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_start_flag;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_start_time;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_stop_time;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_default_color;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_space;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_decimals;

    struct ::tms _start_TMS {};
    struct ::tms _end_TMS   {};
};

#endif

} // namespace timer

#include "impl/Timer.i.hpp"
