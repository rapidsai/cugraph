/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#include <cusp/opengl/spy/gl_util.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

template <class Derived>
class glut_window
{
public:
    void register_with_glut()
    {
        set_window_data(glut_id, this);

        glutSetWindow(glut_id);
        glutDisplayFunc(Derived::glut_display);
        glutReshapeFunc(Derived::glut_reshape);
        glutKeyboardFunc(Derived::glut_key);
        glutSpecialFunc(Derived::glut_special_key);
        glutMotionFunc(Derived::glut_motion);
        glutMouseFunc(Derived::glut_mouse_click);
    }

    int get_glut_window_id()
    {
        return (glut_id);
    }

protected:
    int glut_id;

    /**
     * Empty prototypes...
     */
    void display()
    {
    }

    void reshape(int w, int h)
    {
    }

    void key(unsigned char key, int x, int y)
    {
    }

    void special_key(int key, int x, int y)
    {
    }

    void motion(int x, int y)
    {
    }

    void mouse_click(int button, int state, int x, int y)
    {
    }

    void menu(int value)
    {
    }

public:
    static void glut_display()
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->display();
    }

    static void glut_reshape(int w, int h)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->reshape(w,h);
    }

    static void glut_key(unsigned char key, int x, int y)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->key(key,x,y);
    }
    static void glut_special_key(int key, int x, int y)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->special_key(key,x,y);
    }
    static void glut_motion(int x, int y)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->motion(x,y);
    }
    static void glut_mouse_click(int button, int state, int x, int y)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->mouse_click(button,state,x,y);
    }

    static void glut_menu(int value)
    {
        Derived* gw = get_class_pointer();
        if (gw)
            gw->menu(value);
    }


private:
    static Derived* get_class_pointer()
    {
        int w_id = glutGetWindow();
        void* d = get_window_data(w_id);
        glut_window<Derived>* gw = (glut_window<Derived>*)d;
        if (gw && gw->glut_id==w_id)
            return ((Derived*)gw);
        else
            return (0);
    }
};

} // end spy
} // end opengl
} // end cusp

