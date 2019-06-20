/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
namespace xlib {

inline bool is_integer(const std::string& str) {
    return str.find_first_not_of("0123456789") == std::string::npos;
}

template<int N, unsigned MUL, int INDEX = 0>
class fastStringToIntStr {
public:
    static HOST_DEVICE unsigned aux(const char* str) {
        return static_cast<unsigned>(str[INDEX] - '0') * MUL
                + fastStringToIntStr<N - 1, MUL / 10, INDEX + 1>::aux(str);
    }
};
template<unsigned MUL, int INDEX>
class fastStringToIntStr<1, MUL, INDEX> {
public:
    static HOST_DEVICE unsigned aux(const char* str) {
        return static_cast<unsigned>(str[INDEX] - '0');
    }
};

HOST_DEVICE unsigned fastStringToInt(const char* str, int length) {
    switch(length) {
        case 10: return fastStringToIntStr<10, 1000000000>::aux(str);
        case  9: return fastStringToIntStr<9, 100000000>::aux(str);
        case  8: return fastStringToIntStr<8, 10000000>::aux(str);
        case  7: return fastStringToIntStr<7, 1000000>::aux(str);
        case  6: return fastStringToIntStr<6, 100000>::aux(str);
        case  5: return fastStringToIntStr<5, 10000>::aux(str);
        case  4: return fastStringToIntStr<4, 1000>::aux(str);
        case  3: return fastStringToIntStr<3, 100>::aux(str);
        case  2: return fastStringToIntStr<2, 10>::aux(str);
        case  1: return fastStringToIntStr<1, 1>::aux(str);
        default: return 0;
    }
}

//==============================================================================

template<int LENGHT>
class fastIntToStr {
public:
    static inline void aux(unsigned value, char* buffer) {
        static
        const char digits[201] = "0001020304050607080910111213141516171819"
                                 "2021222324252627282930313233343536373839"
                                 "4041424344454647484950515253545556575859"
                                 "6061626364656667686970717273747576777879"
                                 "8081828384858687888990919293949596979899";

        unsigned i = LENGHT <= 2 ? (value << 1u) : ((value % 100u) << 1u);
        buffer[LENGHT - 1] = digits[i + 1];
        buffer[LENGHT - 2] = digits[i];
        fastIntToStr<LENGHT - 2>::aux(value / 100u, buffer);
    }
};
template<>
class fastIntToStr<1> {
public:
    static inline void aux(unsigned value, char* buffer) {
        static const char digits[11] = "0123456789";
        buffer[0] = digits[value];
    }
};
template<>
class fastIntToStr<0> {
public:
    static inline void aux(unsigned, char*) {}
};

inline void fastIntToString(unsigned value, char** buffer) {
    int length = value < 10 ? 1 :
                (value < 100) ? 2 :
                (value < 1000) ? 3 :
                (value < 10000) ? 4 :
                (value < 100000) ? 5 :
                (value < 1000000) ? 6 :
                (value < 10000000) ? 7 :
                (value < 100000000) ? 8 :
                (value < 1000000000) ? 9 : 10;

    switch(length) {
        case 10: fastIntToStr<10>::aux(value, *buffer); break;
        case  9: fastIntToStr<9>::aux(value, *buffer); break;
        case  8: fastIntToStr<8>::aux(value, *buffer); break;
        case  7: fastIntToStr<7>::aux(value, *buffer); break;
        case  6: fastIntToStr<6>::aux(value, *buffer); break;
        case  5: fastIntToStr<5>::aux(value, *buffer); break;
        case  4: fastIntToStr<4>::aux(value, *buffer); break;
        case  3: fastIntToStr<3>::aux(value, *buffer); break;
        case  2: fastIntToStr<2>::aux(value, *buffer); break;
        case  1: fastIntToStr<1>::aux(value, *buffer); break;
        default: break;
    }
    *buffer += length;
}

} // namespace xlib
