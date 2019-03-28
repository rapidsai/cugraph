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
 * @file matrix_data_cursor.cc
 * The definition file for the matrix_data_cursor class.
 */

/*
 * David Gleich
 * 22 November 2006
 * Copyright, Stanford University
 */
#pragma once

#include <cusp/opengl/spy/gl_util.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

/**
 * The matrix_data_cursor class handles drawing a data_cursor on the
 * screen.  The cursor is a set of two lines identifying a row and
 * a column of the matrix.
 */
class matrix_data_cursor
{
public:
    matrix_data_cursor(int in_nrows, int in_ncols)
        : nrows(in_nrows), ncols(in_ncols),
          cursor_x((float)in_nrows/2),
          cursor_y((float)in_nrows/2),
          state(STATE_NONE)
    {
        color[0]=1.0f;
        color[1]=1.0f;
        color[2]=0.0f;
    }

    void set_matrix_size(int in_nrows, int in_ncols)
    {
        nrows = in_nrows;
        ncols = in_ncols;
    }

    void draw()
    {
        int m = nrows;
        int n = ncols;
        int vx = (int)cursor_x;
        int vy = (int)cursor_y;

        glColor3f(color[0],color[1],color[2]);
        glBegin(GL_LINES);
        glVertex2f(-0.5f,(GLfloat)vy);
        glVertex2f(n-0.5f,(GLfloat)vy);
        glEnd();
        glBegin(GL_LINES);
        glVertex2f((GLfloat)vx,-0.5f);
        glVertex2f((GLfloat)vx,(GLfloat)m-0.5f);
        glEnd();
    }

    bool scaled_motion(float x, float y)
    {
        if (state == STATE_MOVING)
        {
            cursor_x += x;
            cursor_y += y;

            bound_cursor();

            glutPostRedisplay();

            return (true);
        }

        return (false);
    }

    bool mouse_click(int button, int button_state, int x, int y)
    {
        if (button_state == GLUT_UP)
        {
            state = STATE_NONE;

            return (false);
        }

        if (button_state != GLUT_DOWN)
        {
            return (false);
        }

        // henceforth, we can assume that button_state == GLUT_DOWN

        int mod_state = glutGetModifiers();

        if ((button == GLUT_LEFT_BUTTON && mod_state == GLUT_ACTIVE_CTRL) ||
                (button == GLUT_MIDDLE_BUTTON))
        {
            state = STATE_MOVING;

            double wx,wy;
            screen_to_world(x,y,&wx, &wy);

            cursor_x = (float)(wx+0.5);
            cursor_y = (float)(wy+0.5);

            bound_cursor();
            glutPostRedisplay();
            return (true);
        }

        return (false);
    }

    int get_x()
    {
        return (int)cursor_x;
    }
    int get_y()
    {
        return (int)cursor_y;
    }

    void set_position(int x, int y) {
        cursor_x = (float)x;
        cursor_y = (float)y;
        bound_cursor();
        glutPostRedisplay();
    }

    void set_color(float r, float g, float b)  {
        color[0]=r;
        color[1]=g;
        color[2]=b;
    }

private:
    /*int matrix_data_cursor_nrows;
    int matrix_data_cursor_ncols;

    int matrix_data_cursor_x;
    int matrix_data_cursor_y;

    enum {
        MOUSE_DATA_CURSOR_STATE_MOVING,
        MOUSE_DATA_CURSOR_NONE
    } mouse_data_cursor_state;*/

    int nrows;
    int ncols;

    float cursor_x;
    float cursor_y;

    enum {
        STATE_MOVING,
        STATE_NONE
    } state;

    void bound_cursor()
    {
        if(cursor_x < 0) cursor_x = 0;
        if(cursor_y < 0) cursor_y = 0;
        if(cursor_x > ncols-1) cursor_x = (float)(ncols-1);
        if(cursor_y > nrows-1) cursor_y = (float)(nrows-1);
    }

    float color[3];
};

} // end spy
} // end opengl
} // end cusp

#include <cusp/opengl/spy/detail/matrix_data_cursor.inl>

