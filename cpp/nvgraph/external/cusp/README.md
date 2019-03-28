<hr>
<h3>CUSP : A C++ Templated Sparse Matrix Library</h3>

Current release    : v0.5.1 (April 28, 2015)

| Linux | Windows | Coverage |
| ----- | ------- | -------- |
| [![Linux](https://travis-ci.org/sdalton1/cusplibrary.png)](https://travis-ci.org/sdalton1/cusplibrary) | [![Windows](https://ci.appveyor.com/api/projects/status/36pf1oqwkfq6xekn?svg=true)](https://ci.appveyor.com/project/StevenDalton/cusplibrary) | [![Coverage](https://coveralls.io/repos/sdalton1/cusplibrary/badge.svg?branch=master)](https://coveralls.io/r/sdalton1/cusplibrary?branch=master) |

View the project at [CUSP Website](http://cusplibrary.github.io) and the [cusp-users discussion forum](http://groups.google.com/group/cusp-users) for information and questions.

<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, cusp::device_memory> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b);

    return 0;
}
```

<br><hr>
<h3>Stable Releases</h3>

CUSP releases are labeled using version identifiers having three fields:

| Date | Version | Date | Version |
| ---- | ------- | ---- | ------- |
|            |                                                                              | 03/13/2015 | [CUSP v0.5.0](https://github.com/cusplibrary/cusplibrary/archive/v0.5.0.zip) |
|            |                                                                              | 08/30/2013 | [CUSP v0.4.0](https://github.com/cusplibrary/cusplibrary/archive/v0.4.0.zip) |
|            |                                                                              | 03/08/2012 | [CUSP v0.3.1](https://github.com/cusplibrary/cusplibrary/archive/v0.3.1.zip) |
|            |                                                                              | 02/04/2012 | [CUSP v0.3.0](https://github.com/cusplibrary/cusplibrary/archive/v0.3.0.zip) |
|            |                                                                              | 05/30/2011 | [CUSP v0.2.0](https://github.com/cusplibrary/cusplibrary/archive/v0.2.0.zip) |
| 04/28/2015 | [CUSP v0.5.1](https://github.com/cusplibrary/cusplibrary/archive/v0.5.1.zip) | 07/10/2010 | [CUSP v0.1.0](https://github.com/cusplibrary/cusplibrary/archive/v0.1.0.zip) |


<br><hr>
<h3>Contributors</h3>

CUSP is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).
[Nathan Bell](http:github.com/wnbell) was the original creator and
[Steven Dalton](http://github.com/sdalton1) is the current primary contributor.

<br><hr>
<h3>Citing</h3>

```shell
@MISC{Cusp,
  author = "Steven Dalton and Nathan Bell and Luke Olson and Michael Garland",
  title = "Cusp: Generic Parallel Algorithms for Sparse Matrix and Graph Computations",
  year = "2014",
  url = "http://cusplibrary.github.io/",
  note = "Version 0.5.0"
}
```

<br><hr>
<h3>Open Source License</h3>

CUSP is available under the Apache open-source license:

```
Copyright 2008-2014 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
