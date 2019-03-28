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

/**
 * @file gl_util.cc
 * Helpful functions for OpenGL applications.
 */

/*
 * David Gleich, Matt Rasmussen, Leonid Zhukov
 * 6 April 2006
 */

#pragma once

#include <map>

#include <cusp/opengl/spy/gl_util.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

/**
 * Provide the height of a GLUT bitmap.
 *
 * This function is provided by the OpenGLUT and FreeGLUT libraries
 * but it not in GLUT proper, here we add it...
 *
 * @param font the GLUT font identifier
 * @return the height of the GLUT font in screen pixels.
 */
int glutBitmapHeight(void* font)
{
    if (font == GLUT_BITMAP_HELVETICA_18)
	{
		return (14);
	}
	else if (font == GLUT_BITMAP_9_BY_15)
	{
		return (9);
	}
	else if (font == GLUT_BITMAP_8_BY_13)
	{
		return (8);
	}

    return (0);
}

/**
 * Get the screen coordinates for a point in the world coordinates.
 * This function works for the currently selected OpenGL 
 * context.
 *
 * @param wx x world coordinate
 * @param wy y world coordinate
 * @param x output x screen coordinate
 * @param y output y screen coordinate
 */
void world_to_screen(float wx, float wy, int *x, int *y)
{
    GLdouble window_x;
    GLdouble window_y;
    GLdouble window_z;
    GLint viewport[4];
    GLdouble modelmatrix[16], projmatrix[16];


    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);
    
    gluProject(wx, wy, 0, modelmatrix, projmatrix, viewport, &window_x, &window_y, &window_z);
    *x = int(window_x);
    *y = int(window_y);
}

/**
 * Convert screen coordinates into world/windows/opengl coordinates.
 *
 * This function works for the currently selected OpenGL 
 * context.
 *
 * @param wx output x world coordinate
 * @param wy output y world coordinate
 * @param x x screen coordinate
 * @param y y screen coordinate
 */

void screen_to_world(int x, int y, double *wx, double *wy)
{
    GLint viewport[4];
    GLdouble modelmatrix[16], projmatrix[16];

    //  OpenGL y coordinate position  
    GLint realy;          
    
    // returned world x, y, z coords  
    GLdouble wz;      
    
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);

    int height = glutGet(GLUT_WINDOW_HEIGHT);

    //  note viewport[3] is height of window in pixels  
    realy = viewport[3] - (GLint) y + (height-(viewport[3]-viewport[1]))-viewport[1];
    
    gluUnProject((GLdouble)x, (GLdouble)realy, 0.0, 
                  modelmatrix, projmatrix, viewport, wx, wy, &wz);
}

/**
 * Get the 2d extents of the current screen in world coordinates.
 *
 * The points (x1,y1) and (x2,y2) enclose the viewable area.
 *
 * @param x1 the x coordinate of the world view, upper left of window
 * @param y1 the y coordinate of the world view, upper left of window
 * @param x2 the x coordinate of the world view, lower right of window
 * @param y2 the y coordinate of the world view, lower right of window
 */
void world_extents(float &x1, float &y1, float &x2, float &y2)
{
    GLint viewport[4];
    GLdouble modelmatrix[16], projmatrix[16];

    //  OpenGL y coordinate position  
    GLint x,y;
    GLint realy;          
    
    // returned world x, y, z coords  
    GLdouble wx,wy,wz;
    
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);

    int height = glutGet(GLUT_WINDOW_HEIGHT);
    int width = glutGet(GLUT_WINDOW_WIDTH);

    //
    // compute the upper left corner
    //
    x = 0;
    y = 0;

    //  note viewport[3] is height of window in pixels  
    realy = viewport[3] - (GLint) y + (height-(viewport[3]-viewport[1]))-viewport[1];
    
    gluUnProject((GLdouble)x, (GLdouble)realy, 0.0, 
                  modelmatrix, projmatrix, viewport, &wx, &wy, &wz);

    x1 = (float)wx;
    y1 = (float)wy;

    //
    // compute the lower right corner
    //
    x = width;
    y = height;

    //  note viewport[3] is height of window in pixels  
    realy = viewport[3] - (GLint) y + (height-(viewport[3]-viewport[1]))-viewport[1];
    
    gluUnProject((GLdouble)x, (GLdouble)realy, 0.0, 
                  modelmatrix, projmatrix, viewport, &wx, &wy, &wz);

    x2 = (float)wx;
    y2 = (float)wy;
}

/**
 * Scale distance (val) on the screen into world coordinates.  This 
 * function aids finding the world-size of "one-pixel" objects,
 * for example.
 *
 * This function works for the currently selected OpenGL 
 * context.
 *
 * @param val the size on the screen in pixels
 * @return the world size of a val pixels on the screen
 */
float scale_to_world(float val)
{
    GLdouble wx, wy, wx2, wy2;
    screen_to_world(0, 0, &wx, &wy);
    screen_to_world(0, int(val), &wx2, &wy2);
    return float(wy2 - wy);
}

#ifndef GL_UTIL_GLUT_TEXT_FONT
#define GL_UTIL_GLUT_TEXT_FONT GLUT_BITMAP_HELVETICA_18
#endif // GL_UTIL_GLUT_TEXT_FONT

/**
 * Draw a text string using GLUT.
 *
 * To change the DEFAULT font, simply define 
 * GL_UTIL_GLUT_TEXT_FONT
 *
 * @param x the x world coordinate of the text string
 * @param y the y world coordinate of the text string
 * @param string the string to draw
 * @param font an optional parameter for the text font.
 */
void output_text_string(float x, float y, const char *string, void* font)
{
    int len, i;

    glRasterPos2f(x, y);
    len = (int) strlen(string);
    for (i = 0; i < len; i++) 
    {
        glutBitmapCharacter(font, string[i]);
    }
}

/**
 * Get the screen extent of the text string with the current font.
 *
 * To change the DEFAULT font, simply define 
 * GL_UTIL_GLUT_TEXT_FONT
 *
 * @param width the width of the text rendering
 * @param height the height of the text rendering
 * @param string the string to draw
 * @param font an optional parameter for the text font.
 */
void text_extends_screen(int *width, int *height, const char *string, void* font)
{
    int w, h;

    w = glutBitmapLength(font, (const unsigned char*)string);
    h = glutBitmapHeight(font);

    *width = w;
    *height = h;
}

/**
 * Get the world extent of the text string with the current font.
 *
 * To change the DEFAULT font, simply define 
 * GL_UTIL_GLUT_TEXT_FONT
 *
 * @param width the width of the text rendering
 * @param height the height of the text rendering
 * @param string the string to draw
 * @param font an optional parameter for the text font.
 */
void text_extends_world(float *width, float *height, const char *string, void* font)
{
    int w, h;

    w = glutBitmapLength(font, (const unsigned char*)string);
    h = glutBitmapHeight(font);

    double ww,wh;
    double wx0,wy0;

    screen_to_world(0,0,&wx0, &wy0);
    screen_to_world(w,h,&ww, &wh);
    *width = (float)(ww-wx0);
    *height = (float)(wh-wy0);
}


/**
 * Draw a text string to the screen.
 *
 * This function provides a series of options about how to render a text
 * string and is more useful than the output_text_string function.
 *
 * By default, the x and y coordinates are world coordinates, and not
 * screen coordinates.  Also, the flags specify the orientation
 * of the string with respect to the point.
 *
 * GL_U_TEXT_WORLD_COORDS -- default, x and y are world coordinates
 * GL_U_TEXT_SCREEN_COORDS -- x and y are screen coordinates
 *
 * GL_U_TEXT_LEFT_X -- default, x is the left horizatonal coordinate of the text string
 * GL_U_TEXT_CENTER_X -- x is the center horizatonal coordinate of the text string
 * GL_U_TEXT_RIGHT_X -- x is the right horizatonal coordinate of the text string
 *
 * GL_U_TEXT_BOTTOM_Y -- default, y is the bottom vertical coordinate of the text string
 * GL_U_TEXT_CENTER_Y -- y is the center vertical coordinate of hte text string
 * GL_U_TEXT_TOP_Y -- default, y is the top vertical coordinate of the text string
 *
 * @param x the x coordinate of the text string
 * @param y the y coordinate of the text string
 * @param string the text string
 * @param flags how to interpret the text string
 * @param font an optional parameter for the text font.
 */
void draw_text(float x, float y, const char *string, unsigned int flags, void* font)
{
    float wx, wy;

    // first, convert everything to screen coordinates
    if (flags & GL_U_TEXT_SCREEN_COORDS)
    {
        double wx8, wy8;
        screen_to_world((int)x, (int)y, &wx8, &wy8);
        wx = (float)wx8;
        wy = (float)wy8;
    }
    else
    {
        wx = x;
        wy = y;
    }
    
    float str_wf, str_hf;
    text_extends_world(&str_wf, &str_hf, string, font);

    // adjust the screen coordinate based on the text alignment options
    if (flags & GL_U_TEXT_CENTER_X)
    {
        wx -= str_wf/2.f;
    }
    else if (flags & GL_U_TEXT_RIGHT_X)
    {
        wx -= str_wf;
    }

    if (flags & GL_U_TEXT_CENTER_Y)
    {
        wy += str_hf/2.f;
    }
    else if (flags & GL_U_TEXT_TOP_Y)
    {
        wy += str_hf;
    }

    output_text_string(wx, wy, string, font);
}


static std::map<int, void*> gl_util_window_to_data;

/**
 * Get the window data pointer for the glut window with window_id.
 *
 * This pointer must previously have been set, otherwise, you will
 * get NULL returned.
 *
 * @param window_id the glut window_id
 * @return the pointer previously registered with set_window_data
 */
void* get_window_data(int window_id)
{
    return (gl_util_window_to_data[window_id]);
}

/**
 * Set the window data pointer for a glut window.
 *
 * This function allows you to associate a pointer with a GLUT
 * window based on the window id.
 *
 * An easy way to use this function is to keep each window
 * as a class, and register the class "this" pointer as the window
 * data.  Then, you can always convert from a static callback
 * to a class-specific call-back.
 *
 * @param window_id the glut window id
 * @param data the data pointer 
 */
void set_window_data(int window_id, void* data)
{
    gl_util_window_to_data[window_id] = data;
}

} // end spy
} // end opengl
} // end cusp
