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

namespace cusp
{
namespace opengl
{
namespace spy
{

void world_to_screen(float wx, float wy, int *x, int *y);
void screen_to_world(int x, int y, double *wx, double *wy);
float scale_to_world(float val);

void world_extents(float &x1, float &y, float &x2, float &y2);

#ifndef GL_UTIL_GLUT_TEXT_FONT
#define GL_UTIL_GLUT_TEXT_FONT GLUT_BITMAP_HELVETICA_18
#endif // GL_UTIL_GLUT_TEXT_FONT

void output_text_string(float x, float y, const char *string, void* font = GL_UTIL_GLUT_TEXT_FONT);
void text_extends(int *width, int *height, const char *string, void* font = GL_UTIL_GLUT_TEXT_FONT);
void text_extends_world(float *width, float *height, const char *string, void* font = GL_UTIL_GLUT_TEXT_FONT);

const unsigned int GL_U_TEXT_WORLD_COORDS = 0;  // default option
const unsigned int GL_U_TEXT_SCREEN_COORDS = 1;

const unsigned int GL_U_TEXT_LEFT_X = 0;        // default option
const unsigned int GL_U_TEXT_CENTER_X = 2;
const unsigned int GL_U_TEXT_RIGHT_X = 4;

const unsigned int GL_U_TEXT_BOTTOM_Y = 0;
const unsigned int GL_U_TEXT_CENTER_Y = 8;
const unsigned int GL_U_TEXT_TOP_Y = 16;

void draw_text(float x, float y, const char *string, 
               unsigned int flags = 0, 
               void* font = GL_UTIL_GLUT_TEXT_FONT);

void* get_window_data(int window_id);
void set_window_data(int window_id, void* data);

} // end spy
} // end opengl
} // end cusp

#include <cusp/opengl/spy/detail/gl_util.inl>
