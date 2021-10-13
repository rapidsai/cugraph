/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Andrei Schaffer, aschaffer@nvidia.com
//
#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>

namespace cugraph {
namespace visitors {

struct return_t {
  struct base_return_t {
    virtual ~base_return_t(void) {}

    virtual void copy(return_t&&)                            = 0;
    virtual std::unique_ptr<base_return_t> clone(void) const = 0;
  };

  template <typename T>
  struct generic_return_t : base_return_t {
    generic_return_t(T const& t) : return_(t) {}

    generic_return_t(T&& t) : return_(std::move(t)) {}

    void copy(return_t&& r) override
    {
      if constexpr (std::is_copy_constructible_v<T>) {
        base_return_t const* p_B = static_cast<base_return_t const*>(r.p_impl_.get());
        return_                  = *(dynamic_cast<T const*>(p_B));
      } else {
        base_return_t* p_B = static_cast<base_return_t*>(r.p_impl_.get());
        return_            = std::move(*(dynamic_cast<T*>(p_B)));
      }
    }

    std::unique_ptr<base_return_t> clone(void) const override
    {
      if constexpr (std::is_copy_constructible_v<T>)
        return std::make_unique<generic_return_t<T>>(return_);
      else
        throw std::runtime_error("ERROR: cannot clone object that is not copy constructible.");
    }

    T const& get(void) const { return return_; }

   private:
    T return_;
  };

  return_t(void) = default;

  template <typename T>
  return_t(T const& t) : p_impl_(std::make_unique<generic_return_t<T>>(t))
  {
  }

  template <typename T>
  return_t(T&& t) : p_impl_(std::make_unique<generic_return_t<T>>(std::move(t)))
  {
  }

  return_t(return_t const& r) : p_impl_{r.clone()} {}

  return_t& operator=(return_t const& r)
  {
    p_impl_ = r.clone();
    return *this;
  }

  return_t(return_t&& other) : p_impl_(std::move(other.p_impl_)) {}
  return_t& operator=(return_t&& other)
  {
    p_impl_ = std::move(other.p_impl_);
    return *this;
  }

  std::unique_ptr<base_return_t> clone(void) const
  {
    if (p_impl_)
      return p_impl_->clone();
    else
      return nullptr;
  }

  template <typename T>
  T const& get(void) const
  {
    if (p_impl_) {
      generic_return_t<T> const* p = static_cast<generic_return_t<T> const*>(p_impl_.get());
      return p->get();
    } else
      throw std::runtime_error("ERROR: nullptr impl.");
  }

  void const* get_ptr(void) const
  {
    if (p_impl_)
      return static_cast<void const*>(p_impl_.get());
    else
      return nullptr;
  }

  void* release(void) { return static_cast<void*>(p_impl_.release()); }

 private:
  std::unique_ptr<base_return_t> p_impl_;
};

}  // namespace visitors
}  // namespace cugraph
