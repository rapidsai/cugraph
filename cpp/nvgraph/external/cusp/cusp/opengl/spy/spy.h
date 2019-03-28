/*
 *  Copyright 2008-2013 Steven Dalton
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#	include <windows.h>
#endif

#if _MSC_VER >= 1400
// disable the warning for "non-safe" functions
#pragma warning ( push )
#pragma warning ( disable : 4996 )
#endif // _MSC_VER >= 1400

#define GL_GLEXT_PROTOTYPES

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <cusp/opengl/spy/glext.h>

#undef max
#undef min

#if _MSC_VER >= 1400
// enable the warning for "non-safe" functions
#pragma warning ( pop )
#endif // _MSC_VER >= 1400

#include <cusp/opengl/spy/gl_util.h>
#include <cusp/opengl/spy/glut_window.h>
#include <cusp/opengl/spy/matrix_data_panel.h>
#include <cusp/opengl/spy/matrix_canvas.h>
#include <cusp/opengl/spy/matrix_data_panel.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

template< typename MatrixType >
int view_matrix(const MatrixType& A)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    int argc = 0;
    glutInit(&argc, NULL);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);

    // create the window
    matrix_canvas<IndexType,ValueType,MemorySpace> wind(800,600);

    // begin the data loading process
    wind.load_matrix(A);

    // this is a work around for a bug with visual c++ where the GLUT
    // calls will crash if we compile in release mode with optimizations
    // enabled
    wind.post_constructor();

    // hacky fix around a GLUI bug
    //glui_reshape_func(600,400);
    glutReshapeWindow(600,400);

    glutMainLoop();

    return 0;
}

} // end spy
} // end opengl
} // end cusp
