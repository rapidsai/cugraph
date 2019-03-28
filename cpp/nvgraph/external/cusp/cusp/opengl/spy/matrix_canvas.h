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

#include <string>
#include <limits>
#include <cmath>
#include <vector>

#include <cusp/csr_matrix.h>
#include <cusp/opengl/spy/glut_2d_canvas.h>
#include <cusp/opengl/spy/matrix_data_panel.h>
#include <cusp/opengl/spy/matrix_data_cursor.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

/**
 * The matrix_canvas assumes that the matrix
 * can be stored in memory.  It loads a matrix_data_cursor and a
 * matrix_data_panel to handle other details of the implementation.
 */
template< typename IndexType, typename ValueType, typename MemorySpace >
class matrix_canvas
    : public glut_2d_canvas
{
protected:
    typedef glut_2d_canvas super;

    // large_scale_nz controls when we build the matrix _m_fast to draw
    // quickly and then ``fill in'' later
    const static int large_scale_nz = 524288;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> _m;

    std::vector<IndexType> irperm;
    std::vector<IndexType> cperm;
    std::vector<IndexType> icperm;
    bool rperm_loaded;
    bool cperm_loaded;

    std::vector<std::string> rlabel;
    std::vector<std::string> clabel;
    std::vector<ValueType> rnorm;
    std::vector<ValueType> cnorm;

    struct {
        ValueType min_val;
        ValueType max_val;
        IndexType max_degree;
        IndexType min_degree;
    } matrix_stats;

    std::string matrix_filename;
    bool matrix_loaded;

    GLuint matrix_display_list;

public:
    matrix_canvas(int w, int h);

    void post_constructor();

    // glut functions
    virtual void draw();
    virtual void reshape(int w, int h);

    void motion(int w, int h);
    void mouse_click(int button, int state, int x, int y);

    void menu(int value);

    virtual void special_key(int key, int x, int y);
    virtual void key(unsigned char key, int x, int y);

    //
    // control functions
    //

    enum permutation_state_type {
        no_permutation=0,
        row_permutation=1,
        column_permutation=2,
        row_column_permutation=3
    };

    enum normalization_state_type {
        no_normalization=0,
        row_normalization=1,
        column_normalization=2,
        row_column_normalization=3
    };

    enum colormap_state_type {
        first_colormap=1,
        user_colormap=1,
        rainbow_colormap=2,
        bone_colormap=3,
        spring_colormap=4,
        last_colormap,
    };

    void show_data_panel();
    void hide_data_panel();

    void set_point_alpha(float a) {
        if (a >= 0. && a <= 1.) point_alpha = a;
    }
    float get_point_alpha() {
        return (point_alpha);
    }

    float get_aspect() {
        return aspect;
    }
    void set_aspect(float r) {
        if (r > 0) aspect = r;
    }

    void home();

    void set_permutation(permutation_state_type p) {
        permutation_state = p;
    }
    void set_normalization(normalization_state_type n) {
        normalization_state = n;
    }

    colormap_state_type get_colormap();
    void set_colormap(colormap_state_type c);
    void set_next_colormap();

    void set_border_color(float r, float g, float b)
    {
        border_color[0]=r;
        border_color[1]=g;
        border_color[2]=b;
    }

    // data loading
    template< typename MatrixType >
    bool load_matrix(const MatrixType& A);
    bool load_permutations(const std::string& rperm_filename,
                           const std::string& cperm_filename);
    bool load_labels(const std::string& rlabel_filename,
                     const std::string& clabel_filename);

protected:
    // matrix drawing
    void draw_full_matrix();
    void draw_partial_matrix(int r1, int c1, int r2, int c2);

    template <bool partial, class NRMap, class NCMap, class PRMap, class PCMap>
    void draw_matrix(int r1, int c1, int r2, int c2,
                     ValueType min, ValueType inv_val_range, float alpha,
                     NRMap nrv, NCMap ncv, PRMap iprm, PCMap pcm);

    template <bool partial, class NRMap, class NCMap>
    void draw_matrix_dispatch(int r1, int c1, int r2, int c2,
                              ValueType min, ValueType inv_val_range, float alpha,
                              NRMap nrv, NCMap ncv);

    template <bool partial>
    void draw_matrix_dispatch(int r1, int c1, int r2, int c2);

    void write_svg();

    float alpha_from_zoom();

    // control variables
    float point_alpha;

    bool data_panel_visible;
    bool control_visible;

    permutation_state_type permutation_state;
    normalization_state_type normalization_state;
    colormap_state_type colormap_state;
    bool colormap_invert;

    struct colormap_type {
        float *map;
        int size;

        colormap_type(float *m, int s) : map(m), size(s) {}
    } colormap;

    float border_color[3];

    // internal functions
    void init_window();
    void init_display_list();
    void init_menu();

    ValueType matrix_value(IndexType r, IndexType c);
    const std::string& row_label(IndexType r);
    const std::string& column_label(IndexType r);

    const std::string empty_label;

    const static int menu_file_id = 1;
    const static int menu_exit_id = 2;
    const static int menu_toggle_cursor_id = 3;

    const static int menu_aspect_normal = 4;
    const static int menu_aspect_wide_12 = 5;
    const static int menu_aspect_wide_14 = 6;
    const static int menu_aspect_tall_21 = 7;
    const static int menu_aspect_tall_41 = 8;

    const static int menu_colormap_rainbow = 1001;
    const static int menu_colormap_bone = 1002;
    const static int menu_colormap_spring = 1003;
    const static int menu_colormap_invert = 1101;

    const static int menu_colors_white_bkg = 2001;
    const static int menu_colors_black_bkg = 2002;


    const static int panel_offset = 5;
    const static int panel_height = 50;

    // workaround for stupid gcc bug
    int p_offset;
    int p_height;

    // put everything I want constructed last here
    matrix_data_panel data_panel;
    matrix_data_cursor data_cursor;
};

} // end spy
} // end opengl
} // end cusp

#include <cusp/opengl/spy/detail/matrix_canvas.inl>
#include <cusp/opengl/spy/detail/matrix_canvas_svg_output.inl>
