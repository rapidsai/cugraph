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
 */
#include <cassert>          //assert
#include <cmath>            //std::sqrt
#include <ctime>            //std::clock
#include <iomanip>          //std::setprecision
#include <ratio>            //std::ratio
#if defined(__linux__)
    #include <sys/times.h>  //::times
    #include <unistd.h>     //::sysconf
#endif

namespace timer {

template<class Rep, std::intmax_t Num, std::intmax_t Denom>
std::ostream& operator<<(std::ostream& os,
                         const std::chrono::duration
                            <Rep, std::ratio<Num, Denom>>&) {
    if (Num == 3600 && Denom == 1)    return os << " h";
    else if (Num == 60 && Denom == 1)      return os << " min";
    else if (Num == 1 && Denom == 1)       return os << " s";
    else if (Num == 1 && Denom == 1000)    return os << " ms";
    else if (Num == 1 && Denom == 1000000) return os << " us";
    else return os << " Unsupported";
}

//==============================================================================
//-------------------------- GENERIC -------------------------------------------
namespace detail {

template<timer_type type, typename ChronoPrecision>
TimerBase<type, ChronoPrecision>
::TimerBase(int decimals, int space, xlib::Color color) noexcept :
                   _decimals(decimals),
                   _space(space),
                   _default_color(color) {}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::duration() const noexcept {
    return _time_elapsed.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::total_duration() const noexcept {
    return _total_time_elapsed.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::average() const noexcept {
    auto num_executions = static_cast<float>(_num_executions);
    return _total_time_elapsed.count() / num_executions;
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::std_deviation() const noexcept {
    auto term1 = _num_executions * _time_squared.count();
    auto term2 = _total_time_elapsed.count() * _total_time_elapsed.count();
    return std::sqrt(term1 - term2) / _num_executions;
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::min() const noexcept {
    return _time_min.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::max() const noexcept {
        return _time_max.count();
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::reset() noexcept {
    _time_min           = ChronoPrecision(0);
    _time_max           = ChronoPrecision(0);
    _total_time_elapsed = ChronoPrecision(0);
    _num_executions     = 0;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::register_time() noexcept {
    assert(_start_flag);
    _time_squared       += _time_elapsed * _time_elapsed.count();
    _total_time_elapsed += _time_elapsed;
    _num_executions++;
    if (_time_elapsed > _time_max)
        _time_max = _time_elapsed;
    else if (_time_elapsed < _time_min)
        _time_min = _time_elapsed;
    _start_flag = false;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::print(const std::string& str)    //NOLINT
                                             const noexcept {
    xlib::IosFlagSaver tmp;
    std::cout << _default_color
              << std::fixed << std::setprecision(_decimals)
              << std::right << std::setw(_space - 2) << str << "  "
              << duration() << ChronoPrecision()
             << xlib::Color::FG_DEFAULT << std::endl;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::printAll(const std::string& str) //NOLINT
                                                const noexcept {
    xlib::IosFlagSaver tmp;
    std::cout << _default_color
              << std::right << std::setw(_space - 2) << str << ":"
              << std::fixed << std::setprecision(_decimals)
              << "\n  min: " << min()           << ChronoPrecision()
              << "\n  max: " << max()           << ChronoPrecision()
              << "\n  avg: " << average()       << ChronoPrecision()
              << "\n  dev: " << std_deviation() << ChronoPrecision()
              << xlib::Color::FG_DEFAULT << std::endl;
}

} // namespace detail

//==============================================================================
//-----------------------  HOST ------------------------------------------------

template<typename ChronoPrecision>
Timer<HOST, ChronoPrecision>::Timer(int decimals, int space,
                                           xlib::Color color) noexcept :
      timer::detail::TimerBase<HOST, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<HOST, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = true;
    _start_time = std::chrono::system_clock::now();
}

template<typename ChronoPrecision>
void Timer<HOST, ChronoPrecision>::stop() noexcept {
    _stop_time     = std::chrono::system_clock::now();
    _time_elapsed  = ChronoPrecision(_stop_time - _start_time);
    register_time();
}

//==============================================================================
//-------------------------- CPU -----------------------------------------------

template<typename ChronoPrecision>
Timer<CPU, ChronoPrecision>::Timer(int decimals, int space, xlib::Color color)
                                   noexcept :
       timer::detail::TimerBase<CPU, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<CPU, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag  = true;
    _start_clock = std::clock();
}

template<typename ChronoPrecision>
void Timer<CPU, ChronoPrecision>::stop() noexcept {
    _stop_clock = std::clock();
    auto clock_time_elapsed = static_cast<float>(_stop_clock - _start_clock) /
                              static_cast<float>(CLOCKS_PER_SEC);
    auto time_seconds = seconds(clock_time_elapsed);
    _time_elapsed  = std::chrono::duration_cast<ChronoPrecision>(time_seconds);
    register_time();
}

//==============================================================================
//-------------------------- SYS -----------------------------------------------

#if defined(__linux__)

template<typename ChronoPrecision>
Timer<SYS, ChronoPrecision>::Timer(int decimals, int space, xlib::Color color)
                                   noexcept :
       timer::detail::TimerBase<SYS, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = true;
    _start_time = std::chrono::system_clock::now();
    ::times(&_start_TMS);
}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::stop() noexcept {
    assert(_start_flag);
    _stop_time = std::chrono::system_clock::now();
    ::times(&_end_TMS);
    _start_flag = false;
}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::print(const std::string& str)  //NOLINT
                                        const noexcept {
    xlib::IosFlagSaver tmp;
    auto  wall_time_ms = std::chrono::duration_cast<ChronoPrecision>(
                                             _stop_time - _start_time ).count();

    auto     user_diff = _end_TMS.tms_utime - _start_TMS.tms_utime;
    auto    user_float = static_cast<float>(user_diff) /
                         static_cast<float>(::sysconf(_SC_CLK_TCK));
    auto     user_time = seconds(user_float);
    auto  user_time_ms = std::chrono::duration_cast<ChronoPrecision>(user_time);

    auto      sys_diff = _end_TMS.tms_stime - _start_TMS.tms_stime;
    auto     sys_float = static_cast<float>(sys_diff) /
                         static_cast<float>(::sysconf(_SC_CLK_TCK));
    auto      sys_time = seconds(sys_float);
    auto   sys_time_ms = std::chrono::duration_cast<ChronoPrecision>(sys_time);

    std::cout << _default_color << std::setw(_space) << str
              << std::fixed << std::setprecision(_decimals)
              << "  Elapsed time: [user " << user_time_ms << ", system "
              << sys_time_ms << ", real "
              << wall_time_ms << ChronoPrecision() << "]"
              << xlib::Color::FG_DEFAULT << std::endl;
}
#endif

} // namespace timer
