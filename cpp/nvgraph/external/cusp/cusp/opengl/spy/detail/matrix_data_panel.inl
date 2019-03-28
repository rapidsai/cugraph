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
 * @file matrix_data_panel.cc
 * The implementation file attached to the matrix_data_panel class.
 */

/*
 * David Gleich
 * 22 November 2006
 * Copyright, Stanford University
 */

#pragma once

#include <cusp/opengl/spy/matrix_data_panel.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

matrix_data_panel::matrix_data_panel(int parent_id,
                                     int w, int h, int x, int y)
    : parent_glut_id(parent_id)
{
    set_background_color(0.25f,0.25f,0.25f);
    set_border_color(0.0f,1.0f,0.0f);
    set_text_color(1.0f,1.0f,1.0f);
    super::glut_id = glutCreateSubWindow(parent_glut_id,
                                         x,y,w,h);
    super::register_with_glut();
}

void matrix_data_panel::display()
{
    //glClearColor(0.0f, 0.0f, 1.0f, 0.8f);
    glutSetWindow(super::get_glut_window_id());

    glClearColor (background_color[0], background_color[1], background_color[2], 0.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //char str[300];
    //snprintf(str, 300, "row [%d]:    ", matrix->rperm[vy]);
    //string rs = str + matrix->rlabels[matrix->rperm[vy]];

    //std::string rs = "row:    " +
    std::ostringstream row_oss;
    row_oss << row;

    std::ostringstream col_oss;
    col_oss << col;

    std::ostringstream val_oss;
    val_oss.precision(3);
    val_oss << val;

    glColor3f(text_color[0],text_color[1],text_color[2]);

    draw_text(5,15,"row", GL_U_TEXT_SCREEN_COORDS);
    draw_text(5,30,"column", GL_U_TEXT_SCREEN_COORDS);
    draw_text(5,45,"value", GL_U_TEXT_SCREEN_COORDS);

    draw_text(70,15,row_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);
    draw_text(70,30,col_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);
    draw_text(70,45,val_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);

    draw_text(200,15,rlabel.c_str(), GL_U_TEXT_SCREEN_COORDS);
    draw_text(200,30,clabel.c_str(), GL_U_TEXT_SCREEN_COORDS);

    //draw_text(5,15, row_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);
    //draw_text(5,30, col_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);
    //draw_text(5,45, val_oss.str().c_str(), GL_U_TEXT_SCREEN_COORDS);

    glColor3f (border_color[0],border_color[1],border_color[2]);
    glBegin (GL_LINE_LOOP);
    glVertex2f (0.0F, 0.0F);
    glVertex2f (0.0F, 0.99F);
    glVertex2f (0.999F, 0.99F);
    glVertex2f (0.999F, 0.0F);
    glEnd ();

    glutSwapBuffers();
}

void matrix_data_panel::reshape(int w, int h)
{
    glViewport (0, 0, w, h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluOrtho2D (0.0F, 1.0F, 0.0F, 1.0F);

    width = w;
    height = h;
}

} // end spy
} // end opengl
} // end cusp
