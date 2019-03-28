#pragma once

#include <cusp/complex.h>

#include <string>
#include <iostream>
#include <sstream>

namespace unittest
{

class UnitTestException
{
public:
    std::string message;

    UnitTestException() {}
    UnitTestException(const std::string& msg) : message(msg) {}

    friend std::ostream& operator<<(std::ostream& os, const UnitTestException& e)
    {
        return os << e.message;
    }

    template <typename T>
    UnitTestException& operator<<(const T& t)
    {
        std::ostringstream oss;
        oss << t;
        message += oss.str();
        return *this;
    }

	template <typename T>
	UnitTestException& operator<<(const thrust::device_reference<T>& t)
	{
		T t_(t);
		return operator<<(t_);
	}

#if THRUST_VERSION < 100800
    template <typename T>
    UnitTestException& operator<<(const thrust::complex<T>& t)
    {
        std::ostringstream oss;
        oss << "(" << t.real() << ", " << t.imag() << ")";
        message += oss.str();
        return *this;
    }
#endif

};


class UnitTestError   : public UnitTestException
{
public:
    UnitTestError() {}
    UnitTestError(const std::string& msg) : UnitTestException(msg) {}
};

class UnitTestFailure : public UnitTestException
{
public:
    UnitTestFailure() {}
    UnitTestFailure(const std::string& msg) : UnitTestException(msg) {}
};

class UnitTestKnownFailure : public UnitTestException
{
public:
    UnitTestKnownFailure() {}
    UnitTestKnownFailure(const std::string& msg) : UnitTestException(msg) {}
};


} //end namespace unittest

