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
 * @file glut_2d_canvas.cc
 * Implement a simple 2d canvas using glut.
 */

#pragma once

#include <cusp/opengl/spy/glut_2d_canvas.h>
#include <cusp/opengl/spy/gl_util.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

/**
 * Simple constructor that just takes a width, height, and title.
 *
 * @param w the width of the window
 * @param h the height of the window
 * @param title the window title
 */
glut_2d_canvas::glut_2d_canvas(int w, int h, const char* title)
    : width(w), height(h),
      natural_width((float)w), natural_height((float)h),
      zoom(1.0f),
      trans_x(0.0f), trans_y(0.0f),
      center_x(0.0f), center_y(0.0f),
      aspect(1.0f),
      mouse_state(STATE_NONE),
      mouse_begin_x(0), mouse_begin_y(0),
      virtual_width((float)w), virtual_height((float)h),
      display_finished(true),
      frame(0)
{
    set_background_color(0.0f,0.0f,0.0f);
    glutInitWindowSize(w,h);
    glut_id = glutCreateWindow(title);
    register_with_glut();
}

void glut_2d_canvas::draw()
{
}


void glut_2d_canvas::display()
{
    if (display_finished) {
        frame = 0;
        glClearColor(background_color[0],background_color[1],background_color[2],0.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    } else {
        frame++;
    }

    display_finished = true;

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(center_x, center_y, 0);
    glScalef(zoom,zoom,1);
    glTranslatef(-center_x, -center_y, 0);
    glTranslatef(trans_x, trans_y,0);

    draw();

    glutSwapBuffers();

    if (!display_finished) {
        glutPostRedisplay();
    }
}

void glut_2d_canvas::reshape(int w, int h)
{
    width = w;
    height = h;

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float scale1 = (float)h/natural_height;
    float scale2 = (float)w/natural_width;
    float scalew, scaleh;

    if (scale1 <= scale2) {
        scalew = scale2/scale1;
        //scaleh = 1;
        scaleh = aspect;
    } else {
        //scalew = 1;
        scalew = 1/aspect;
        scaleh = scale1/scale2;
    }

    float x1 = center_x - natural_width*scalew/2;
    float x2 = center_x + natural_width*scalew/2;
    float y1 = center_y - natural_height*scaleh/2;
    float y2 = center_y + natural_height*scaleh/2;

    glOrtho(x1, x2, y2, y1, -1.0, 1.0);
    virtual_width = natural_width*scalew;
    virtual_height = natural_height*scaleh;
}

void glut_2d_canvas::key(unsigned char key, int x, int y)
{
}

void glut_2d_canvas::special_key(int key, int x, int y)
{
}

void glut_2d_canvas::motion(int x, int y)
{
    switch (mouse_state)
    {
    case STATE_MOVE:
        trans_x += (x - mouse_begin_x)*virtual_width/(float)width/zoom;
        trans_y += (y - mouse_begin_y)*virtual_height/(float)height/zoom;

        mouse_begin_x = x;
        mouse_begin_y = y;
        break;
    case STATE_ZOOM:
        // normal zoom
        if(mouse_begin_y > y)
            zoom *= 1.0f/.95f;
        if(mouse_begin_y < y)
            zoom *= 0.95f;

        mouse_begin_y = y;
        break;
    default :
        break;
    }

    glutPostRedisplay();
}


void glut_2d_canvas::mouse_click(int button, int state, int x, int y)
{
    int mod_state = glutGetModifiers();

    switch (state) {
    case GLUT_UP:
        mouse_state = STATE_NONE;

        /*switch (button) {
            case GLUT_WHEEL_DOWN:
                zoom *= 0.75f;
                break;

            case GLUT_WHEEL_UP:
                zoom *= 1.0f/0.75f;
                break;
        }*/

        break;


    case GLUT_DOWN:
        if (button == GLUT_LEFT_BUTTON && mod_state == GLUT_ACTIVE_SHIFT)
        {
            mouse_state = STATE_ZOOM;
        }
        else if (button == GLUT_LEFT_BUTTON)
        {
            mouse_state = STATE_MOVE;
        }

        break;
    }

    begin_mouse_click(x,y);

    glutPostRedisplay();
}

} // end spy
} // end opengl
} // end cusp
