/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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



#ifndef THRUST_TRAITS_HXX

#define THRUST_TRAITS_HXX



#include <thrust/device_vector.h>

#include <thrust/host_vector.h>



namespace nvgraph

{

  //generic Vector Ptr Type facade:

  //

  template<typename T, typename Vector>

  struct VectorPtrT;



  //partial specialization for device_vector:

  //

  template<typename T>

  struct VectorPtrT<T, thrust::device_vector<T> >

  {

    typedef thrust::device_ptr<T> PtrT;

  };



  //partial specialization for host_vector:

  //

  template<typename T>

  struct VectorPtrT<T, thrust::host_vector<T> >

  {

    typedef typename thrust::host_vector<T>::value_type* PtrT;

  };

}

#endif

