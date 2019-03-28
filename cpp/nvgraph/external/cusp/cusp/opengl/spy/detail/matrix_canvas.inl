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
 * @file matrix_canvas.cc
 * An implementation of a matrix canvas class to display a sparse matrix
 * for the spy program.
 */

/*
 * David Gleich
 * 21 November 2006
 * Copyright, Stanford University
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <string>
#include <utility>

#include <cusp/opengl/spy/glut_2d_canvas.h>
#include <cusp/opengl/spy/matrix_data_panel.h>
#include <cusp/opengl/spy/matrix_data_cursor.h>
#include <cusp/opengl/spy/matrix_canvas.h>
#include <cusp/opengl/spy/colormaps.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

template< typename IndexType, typename ValueType, typename MemorySpace >
matrix_canvas<IndexType,ValueType,MemorySpace>::matrix_canvas(int w, int h)
    :
    glut_2d_canvas(w,h,"spy"),  // initialize the window first
    // the order of the rest is the order in which the variables are
    // declared in the class declaration
    rperm_loaded(false),
    cperm_loaded(false),
    matrix_filename(""),
    matrix_loaded(false),
    matrix_display_list(0),
    point_alpha(0.5f),
    data_panel_visible(false),
    permutation_state(no_permutation),
    normalization_state(no_normalization),
    colormap_state(rainbow_colormap),
    colormap_invert(false),
    colormap((float *)spring_color_map,
             sizeof(spring_color_map)/sizeof(spring_color_map[0])),
    p_offset(panel_offset), p_height(panel_height),
    data_panel(glut_id, width-2*p_offset, std::max(height/2,p_height), p_offset, p_offset),
    data_cursor(0,0)
{
    border_color[0]=1.0f;
    border_color[1]=1.0f;
    border_color[2]=1.0f;
}


template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::post_constructor()
{
    init_window();
    init_menu();
    init_display_list();

    set_zoom(0.95f);

    show_data_panel();
}

/**
 * This function draws the matrix.
 */
template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw()
{
    if (!matrix_loaded) {
        return;
    }

    int m = _m.num_rows;
    int n = _m.num_cols;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // border
    glPointSize(1.0);
    glColor3f(border_color[0], border_color[1], border_color[2]);
    glBegin(GL_LINE_LOOP);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(-0.5f, m-0.5f);
    glVertex2f(n-0.5f, m-0.5f);
    glVertex2f(n-0.5f, -0.5f);
    glEnd();

    glPointSize(zoom / (virtual_width*aspect/(float)width));

    float x1,y1,x2,y2;
    world_extents(x1,y1,x2,y2);

    int r1=(int)floor(y1),r2=(int)floor(y2);
    int c1=(int)floor(x1),c2=(int)floor(x2);

    if (std::max(x2-x1,y2-y1) < 16384)
    {
        //std::cout << "drawing partial matrix (" << r1 << ", " << c1 << ") - ("
        //     << r2 << ", " << c2 << ")" << std::endl;

        draw_partial_matrix(r1,c1,r2,c2);
    }
    else if (_m.num_rows < 32768) {
        //std::cout << "drawing full matrix" << std::endl;
        // only draw the full matrix if there isn't a very small portion
        draw_full_matrix();
    }
    else {
        r1=(std::max)(r1,0);
        int r2end=(std::min)(r2,m);

        r1=r1+16384*frame;
        r2=(std::min)(r1+16384,r2end);
        //std::cout << "drawing matrix iteratively " << frame << " (" << r1 << ", " << c1 << ") - ("
        //     << r2 << ", " << c2 << ")" << " " << r2end << std::endl;

        draw_partial_matrix(r1,c1,r2,c2);

        if (r2 != r2end) {
            display_finished = false;
        }
    }

    if (data_panel_visible)
    {
        int r = data_cursor.get_y();
        int c = data_cursor.get_x();

        if (permutation_state == row_permutation ||
                permutation_state == row_column_permutation) {
            r = irperm[r];
        }
        if (permutation_state == column_permutation ||
                permutation_state == row_column_permutation) {
            c = icperm[c];
        }

        // workaround for stupid VC++ bug
        //data_panel.update(r, c, yasmic::value(r, c,
        //    static_cast<const Matrix&>(_m)));
        data_panel.update(r,c,(float)matrix_value(r,c),
                          row_label(r), column_label(c));

        data_cursor.draw();
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
template <bool partial, class NRMap, class NCMap, class PRMap, class PCMap>
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw_matrix(int r1, int c1, int r2, int c2,
                                ValueType min_val, ValueType inv_val_range, float alpha,
                                NRMap nrv, NCMap ncv, PRMap iprm, PCMap pcm)
{
    int colormap_entry;
    ValueType v;

    int m = _m.num_rows;
    //int n = _m.num_cols;

    glBegin( GL_POINTS );
    {
        for (int pi = std::max(0,r1); pi < std::min(r2, m); ++pi)
        {
            // i is the real row in the matrix for the pith row
            // of the display
            int i = iprm[pi];

            for (IndexType ri = _m.row_offsets[i]; ri < _m.row_offsets[i+1]; ++ri)
            {
                // j is the real column in the matrix for the pjth
                // column of the display
                int j = _m.column_indices[ri];
                int pj = pcm[j];

                // skip all the columns outside
                if (partial && (pj < c1 || pj > c2)) {
                    continue;
                }

                v = _m.values[ri]*nrv[i]*ncv[j];

                // scale v to the range [0,1]
                v = v - min_val;
                v = v*inv_val_range;

                if (!colormap_invert) {
                    colormap_entry = (int)(v*(colormap.size-1));
                } else {
                    colormap_entry = (int)(v*(colormap.size-1));
                    colormap_entry=colormap.size-1-colormap_entry;
                }

                glColor4f(colormap.map[colormap_entry*3],
                          colormap.map[colormap_entry*3+1],
                          colormap.map[colormap_entry*3+2],
                          alpha);
                glVertex2f((GLfloat)pj, (GLfloat)pi);
            }
        }
    }
    glEnd();
}

template< typename IndexType, typename ValueType, typename MemorySpace >
template <bool partial, class NRMap, class NCMap>
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw_matrix_dispatch(int r1, int c1, int r2, int c2,
        ValueType min_val, ValueType inv_val_range, float alpha,
        NRMap nrv, NCMap ncv)
{
    typedef typename cusp::array1d_view< thrust::counting_iterator<ValueType> > CountingView;
    CountingView counting_view(thrust::counting_iterator<ValueType>(0),
				thrust::counting_iterator<ValueType>(_m.num_entries));

    switch (permutation_state) {
    case no_permutation:
        draw_matrix<partial>(r1,c1,r2,c2,min_val,inv_val_range,alpha,nrv,ncv,
                             counting_view,counting_view);
        break;

    case row_permutation:
        draw_matrix<partial>(r1,c1,r2,c2,min_val,inv_val_range,alpha,nrv,ncv,
                             &irperm[0],counting_view);
        break;

    case column_permutation:
        draw_matrix<partial>(r1,c1,r2,c2,min_val,inv_val_range,alpha,nrv,ncv,
                             counting_view,&cperm[0]);
        break;

    case row_column_permutation:
        draw_matrix<partial>(r1,c1,r2,c2,min_val,inv_val_range,alpha,nrv,ncv,
                             &irperm[0],&cperm[0]);
        break;
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
template <bool partial>
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw_matrix_dispatch(int r1, int c1, int r2, int c2)
{
    ValueType max_val = matrix_stats.max_val;
    ValueType min_val = matrix_stats.min_val;

    if (max_val - min_val <= 0)
    {
        // this sets min_val to something reasonable, and
        // shows the high end of the colormap if the values
        // are all equal
        min_val = max_val - 1.0;
    }
    ValueType inv_val_range = 1.0/(max_val - min_val);

    float alpha = alpha_from_zoom();

    typedef typename cusp::array1d_view< thrust::constant_iterator<ValueType> > ConstantView;
    ConstantView constant_view(thrust::constant_iterator<ValueType>(1),
				thrust::constant_iterator<ValueType>(1) + _m.num_entries);

    switch (normalization_state) {
    case no_normalization:
        draw_matrix_dispatch<partial>(r1,c1,r2,c2,min_val,inv_val_range,alpha,
                                      constant_view, constant_view);
        break;

    case row_normalization:
        draw_matrix_dispatch<partial>(r1,c1,r2,c2,0.0,1.0,alpha,
                                      &rnorm[0],constant_view);
        break;

    case column_normalization:
        draw_matrix_dispatch<partial>(r1,c1,r2,c2,0.0,1.0,alpha,
                                      constant_view,&cnorm[0]);
        break;

    case row_column_normalization:
        draw_matrix_dispatch<partial>(r1,c1,r2,c2,0.0,1.0,alpha,&rnorm[0],&cnorm[0]);
        break;
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw_full_matrix()
{
    draw_matrix_dispatch<false>(0,0,_m.num_rows,_m.num_cols);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::draw_partial_matrix(int r1, int c1, int r2, int c2)
{
    draw_matrix_dispatch<true>(r1,c1,r2,c2);
}

/**
 * Compute a point_alpha from the current zoom level.
 *
 * zoom < 1: alpha = point_alpha
 * zoom s.t. point_size >= 2.0 => alpha = 1
 */
template< typename IndexType, typename ValueType, typename MemorySpace >
float matrix_canvas<IndexType,ValueType,MemorySpace>::alpha_from_zoom()
{
    float point_size = zoom / (virtual_width*aspect/(float)width);
    if (zoom < 1) {
        return (point_alpha);
    }
    else if (point_size >= 2.0f) {
        return (1.0f);
    }
    else
    {
        float zoom_1 = 1.0;
        float zoom_2 = 2.0*(virtual_width*aspect/(float)width);

        return 1.0f - (zoom_2 - zoom)/(zoom_2 - zoom_1)*(1.0f-point_alpha);
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::reshape(int w, int h)
{
    super::reshape(w,h);

    // handle the subwindow...
    glutSetWindow(data_panel.get_glut_window_id());
    if (data_panel_visible && ((w < 4*p_offset) || (h < 4*p_offset)))
    {
        // if the window is too small, then hide it!
        hide_data_panel();
    }
    else if (data_panel_visible)
    {
        glutPositionWindow(p_offset, p_offset);
        glutReshapeWindow(w-2*p_offset, std::min(h/2,p_height));
    }

    glutSetWindow(get_glut_window_id());

    /*int x,y,w_glui,h_glui;
    GLUI_Master.get_viewport_area(&x,&y,&w_glui,&h_glui);
    if (data_panel_visible)
    {
        //y += panel_height;
        h_glui -= p_height;
    }

    super::reshape(w_glui,h_glui);
    glViewport(x,y,w_glui,h_glui);*/
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::motion(int x, int y)
{
    display_finished = true;

    // rescale the mouse click
    int move_x = x-mouse_begin_x;
    int move_y = y-mouse_begin_y;

    double scaled_x, scaled_y;
    scaled_x = scale_to_world((float)move_x);
    scaled_y = scale_to_world((float)move_y);

    double norm_x=scaled_x, norm_y=scaled_y;
    norm_x /= aspect;

    if (data_cursor.scaled_motion(norm_x, norm_y))
    {
        begin_mouse_click(x,y);
    }
    else
    {
        super::motion(x, y);
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::mouse_click(int button, int state, int x, int y)
{
    if (data_cursor.mouse_click(button, state, x, y))
    {
        begin_mouse_click(x,y);
    }
    else
    {
        super::mouse_click(button, state, x, y);
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::key(unsigned char key, int x, int y)
{
    switch (key) {
    case 'o':
    case 'O':
        write_svg();
        break;
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::special_key(int key, int x, int y)
{
    switch (key) {
    case GLUT_KEY_UP:
        data_cursor.set_position(data_cursor.get_x(),data_cursor.get_y()-1);
        glutPostRedisplay();
        break;

    case GLUT_KEY_DOWN:
        data_cursor.set_position(data_cursor.get_x(),data_cursor.get_y()+1);
        glutPostRedisplay();
        break;

    case GLUT_KEY_LEFT:
        data_cursor.set_position(data_cursor.get_x()-1,data_cursor.get_y());
        glutPostRedisplay();
        break;

    case GLUT_KEY_RIGHT:
        data_cursor.set_position(data_cursor.get_x()+1,data_cursor.get_y());
        glutPostRedisplay();
        break;
    }
}


/**
 * This function initializes the window and assures it is
 * sized correctly, etc.
 */
template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::init_window()
{
    // make sure we are the current window
    glutSetWindow(get_glut_window_id());

    set_center(_m.num_cols/2, _m.num_rows/2);
    set_natural_size(_m.num_cols, _m.num_rows);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#if !defined (__APPLE__)
    // POINT_SMOOTHING doesn't work quite right on OS X when I tried it
    glEnable(GL_POINT_SMOOTH);
#endif
}

/**
 * This function creates the display list for the matrix.
 */
template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::init_display_list()
{
    glutSetWindow(get_glut_window_id());
}

/*
 * =================
 * matrix functions
 * =================
 */

template< typename IndexType, typename ValueType, typename MemorySpace >
ValueType matrix_canvas<IndexType,ValueType,MemorySpace>::matrix_value(
    IndexType r, IndexType c)
{
    int m = _m.num_rows;
    int n = _m.num_cols;

    if (r < 0 || r >= m || c < 0 || c >= n)
    {
        return ValueType(0);
    }

    IndexType ri,riend;
    ri = _m.row_offsets[r];
    riend = _m.row_offsets[r+1];

    while (ri < riend)
    {
        if (c == _m.column_indices[ri]) {
            return _m.values[ri];
        }
        ++ri;
    }
    return ValueType(0);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
const std::string& matrix_canvas<IndexType,ValueType,MemorySpace>::row_label(IndexType r)
{
    if (r >= 0 && r < IndexType(rlabel.size())) {
        return rlabel[r];
    } else {
        return empty_label;
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
const std::string& matrix_canvas<IndexType,ValueType,MemorySpace>::column_label(IndexType c)
{
    if (c >= 0 && c < IndexType(clabel.size())) {
        return clabel[c];
    } else {
        return empty_label;
    }
}

template< typename IndexType, typename ValueType, typename MemorySpace >
template< typename MatrixType >
bool matrix_canvas<IndexType,ValueType,MemorySpace>::load_matrix(const MatrixType& A)
{
    matrix_loaded = false;

    _m = A;

    int m = A.num_rows;
    int n = A.num_cols;

    matrix_loaded = true;
    init_window();
    data_cursor.set_matrix_size(_m.num_rows, _m.num_cols);

    //
    // compute matrix stats
    //

    matrix_stats.max_degree = 0;
    matrix_stats.min_degree = std::numeric_limits<IndexType>::max();
    matrix_stats.max_val = std::numeric_limits<ValueType>::min();
    matrix_stats.min_val = std::numeric_limits<ValueType>::max();

    rnorm.resize(m);
    cnorm.resize(n);

    for (IndexType r = 0; r < m; ++r)
    {
        IndexType deg = _m.row_offsets[r+1] - _m.row_offsets[r];
        matrix_stats.max_degree = std::max(matrix_stats.max_degree, deg);
        matrix_stats.min_degree = std::min(matrix_stats.min_degree, deg);

        for (IndexType ri = _m.row_offsets[r]; ri < _m.row_offsets[r+1]; ++ri)
        {
            ValueType val = _m.values[ri];
            matrix_stats.max_val = std::max(matrix_stats.max_val, val);
            matrix_stats.min_val = std::min(matrix_stats.min_val, val);

            rnorm[r] += val*val;
            cnorm[_m.column_indices[ri]] += val*val;
        }
    }
    for (IndexType r=0; r<m; ++r) {
        rnorm[r]=1.0/std::sqrt(rnorm[r]);
    }
    for (IndexType r=0; r<n; ++r) {
        cnorm[r]=1.0/std::sqrt(cnorm[r]);
    }

    return (true);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
bool matrix_canvas<IndexType,ValueType,MemorySpace>::load_permutations(const std::string& rperm_filename,
                                      const std::string& cperm_filename)
{
    /*if (!rperm_filename.empty() && !util::file_exists(rperm_filename)) {
        std::cerr << rperm_filename << " does not exist." << std::endl;
        return (false);
    }

    if (!cperm_filename.empty() && !util::file_exists(cperm_filename)) {
        std::cerr << cperm_filename << " does not exist." << std::endl;
        return (false);
    }

    // use all the boost iostreams stuff loaded from the load_crm_matrix.hpp
    // header file
    typedef boost::iostreams::filtering_stream<
    boost::iostreams::input_seekable>
    filtered_ifstream;

    if (!rperm_filename.empty())
    {
        YASMIC_VERBOSE( std::cerr << "reading " << rperm_filename << std::endl; )
        filtered_ifstream ios_fifs;
        ifstream ifs(rperm_filename.c_str());

        if (util::gzip_header(ifs))
        {
            ifs.seekg(0, ios::beg);
            ios_fifs.push(boost::iostreams::gzip_decompressor());
            YASMIC_VERBOSE(  std::cerr << "detected gzip" << std::endl; )
        }

        ios_fifs.push(ifs);

        string line;

        irperm.resize(_m.nrows);

        IndexType r;
        for (r=0; r<_m.nrows && !ios_fifs.eof(); ++r) {
            IndexType p;
            ifs >> p;
            irperm[r]=p;
        }

        if (r!=_m.nrows) {
            std::cerr << rperm_filename << " only contains " << r << " entries, "
                 << " not " << _m.nrows << std::endl;
            return (false);
        }

        rperm_loaded = true;
    }

    if (!cperm_filename.empty())
    {
        YASMIC_VERBOSE( std::cerr << "reading " << cperm_filename << std::endl; )
        filtered_ifstream ios_fifs;
        ifstream ifs(cperm_filename.c_str());

        if (util::gzip_header(ifs))
        {
            ifs.seekg(0, ios::beg);
            ios_fifs.push(boost::iostreams::gzip_decompressor());
            YASMIC_VERBOSE(  std::cerr << "detected gzip" << std::endl; )
        }

        ios_fifs.push(ifs);

        string line;

        cperm.resize(_m.ncols);
        icperm.resize(_m.ncols);

        IndexType c;
        for (c=0; c<_m.ncols && !ios_fifs.eof(); ++c) {
            IndexType p;
            ifs >> p;
            icperm[c]=p;
            cperm[p]=c;
        }

        if (c!=_m.ncols) {
            std::cerr << cperm_filename << " only contains " << c << " entries, "
                 << " not " << _m.ncols << std::endl;
            return (false);
        }

        cperm_loaded = true;
    }*/

    return (true);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
bool matrix_canvas<IndexType,ValueType,MemorySpace>::load_labels(const std::string& rlabel_filename,
                                const std::string& clabel_filename)
{
    /*if (!rlabel_filename.empty() && !util::file_exists(rlabel_filename)) {
        std::cerr << rlabel_filename << " does not exist." << std::endl;
        return (false);
    }

    if (!clabel_filename.empty() && !util::file_exists(clabel_filename)) {
        std::cerr << clabel_filename << " does not exist." << std::endl;
        return (false);
    }

    // use all the boost iostreams stuff loaded from the load_crm_matrix.hpp
    // header file
    typedef boost::iostreams::filtering_stream<
    boost::iostreams::input_seekable>
    filtered_ifstream;

    if (!rlabel_filename.empty())
    {
        YASMIC_VERBOSE( std::cerr << "reading " << rlabel_filename << std::endl; )
        filtered_ifstream ios_fifs;
        ifstream ifs(rlabel_filename.c_str());

        if (util::gzip_header(ifs))
        {
            ifs.seekg(0, ios::beg);
            ios_fifs.push(boost::iostreams::gzip_decompressor());
            YASMIC_VERBOSE(  std::cerr << "detected gzip" << std::endl; )
        }

        ios_fifs.push(ifs);

        string line;

        while (!ios_fifs.eof()) {
            getline(ios_fifs, line);
            rlabel.push_back(line);
        }
    }

    if (!clabel_filename.empty())
    {
        YASMIC_VERBOSE(  std::cerr << "reading " << clabel_filename << std::endl; )
        filtered_ifstream ios_fifs;
        ifstream ifs(clabel_filename.c_str());

        if (util::gzip_header(ifs))
        {
            ifs.seekg(0, ios::beg);
            ios_fifs.push(boost::iostreams::gzip_decompressor());
            YASMIC_VERBOSE(  std::cerr << "detected gzip" << std::endl; )
        }

        ios_fifs.push(ifs);

        string line;

        while (!ios_fifs.eof()) {
            getline(ios_fifs, line);
            clabel.push_back(line);
        }
    }*/

    return (true);
}

/*
 * =================
 * control functions
 * =================
 */

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::show_data_panel()
{
    glutSetWindow(data_panel.get_glut_window_id());
    glutShowWindow();
    glutPostRedisplay();
    data_panel_visible = true;
    reshape(width, height);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::hide_data_panel()
{
    glutSetWindow(data_panel.get_glut_window_id());
    glutHideWindow();
    data_panel_visible = false;
    reshape(width, height);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::set_colormap(colormap_state_type c)
{
    // at the moment, we don't have any support for a user colormap
    if (c == user_colormap) {
        c = rainbow_colormap;
    }
    switch (c)
    {
    case rainbow_colormap:
        colormap.map = (float*)rainbow_color_map;
        colormap.size = sizeof(rainbow_color_map)/sizeof(rainbow_color_map[0]);
        break;

    case bone_colormap:
        colormap.map = (float*)bone_color_map;
        colormap.size = sizeof(bone_color_map)/sizeof(bone_color_map[0]);
        break;

    case spring_colormap:
        colormap.map = (float*)spring_color_map;
        colormap.size = sizeof(spring_color_map)/sizeof(spring_color_map[0]);
        break;

    default :
	break;
    }

    glutPostRedisplay();
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::set_next_colormap()
{
    int next = colormap_state+1;
    if (next == last_colormap) {
        next = first_colormap;
    }
    set_colormap((colormap_state_type)next);
}

/*
 * =================
 * menu functions
 * =================
 */

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::init_menu()
{
    int submenu_aspect_ratio = glutCreateMenu(glut_menu);
    glutAddMenuEntry("1:1",menu_aspect_normal);
    glutAddMenuEntry("4:1 (Tall)",menu_aspect_tall_41);
    glutAddMenuEntry("2:1 (Tall)",menu_aspect_tall_21);
    glutAddMenuEntry("1:2 (Wide)",menu_aspect_wide_12);
    glutAddMenuEntry("1:4 (Wide)",menu_aspect_wide_14);

    int submenu_colormap = glutCreateMenu(glut_menu);
    glutAddMenuEntry("Rainbow",menu_colormap_rainbow);
    glutAddMenuEntry("Bone",menu_colormap_bone);
    glutAddMenuEntry("Spring",menu_colormap_spring);
    glutAddMenuEntry("Invert Colormap",menu_colormap_invert);

    int submenu_colors = glutCreateMenu(glut_menu);
    glutAddMenuEntry("White Background",menu_colors_white_bkg);
    glutAddMenuEntry("Black Background",menu_colors_black_bkg);

    // create the main menu
    glutCreateMenu(glut_menu);

    glutAddMenuEntry("Toggle Cursor", menu_toggle_cursor_id);
    glutAddSubMenu("Aspect Ratio", submenu_aspect_ratio);
    glutAddSubMenu("Colormap", submenu_colormap);
    glutAddSubMenu("Colors", submenu_colors);

    glutAddMenuEntry("Exit", menu_exit_id);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::menu(int value)
{
    bool update_screen = false;
    switch (value)
    {
    case menu_file_id:
        break;

    case menu_toggle_cursor_id:
        data_panel_visible ? hide_data_panel() : show_data_panel();
        update_screen = true;
        break;

    case menu_exit_id:
        exit(0);
        break;

    case menu_aspect_normal:
        set_aspect(1.0);
        reshape(width,height);
        update_screen = true;
        break;

    case menu_aspect_wide_12:
        set_aspect(2.0);
        reshape(width,height);
        update_screen = true;
        break;

    case menu_aspect_wide_14:
        set_aspect(4.0);
        reshape(width,height);
        update_screen = true;
        break;

    case menu_aspect_tall_21:
        set_aspect(0.5);
        reshape(width,height);
        update_screen = true;
        break;

    case menu_aspect_tall_41:
        set_aspect(0.25);
        reshape(width,height);
        update_screen = true;
        break;

    case menu_colormap_rainbow:
        set_colormap(rainbow_colormap);
        break;

    case menu_colormap_bone:
        set_colormap(bone_colormap);
        break;

    case menu_colormap_spring:
        set_colormap(spring_colormap);
        break;

    case menu_colormap_invert:
        colormap_invert = !colormap_invert;
        update_screen = true;
        break;

    case menu_colors_white_bkg:
        set_background_color(1.0f,1.0f,1.0f);
        data_panel.set_background_color(0.75f,0.75f,0.75f);
        data_panel.set_text_color(0.0f,0.0f,0.0f);
        data_panel.set_border_color(0.0f,0.0f,1.0f);
        set_border_color(0.0f,0.0f,0.0f);
        data_cursor.set_color(1.0f,0.0f,0.0f);
        update_screen = true;
        break;

    case menu_colors_black_bkg:
        set_background_color(0.0f,0.0f,0.0f);
        data_panel.set_background_color(0.25f,0.25f,0.25f);
        data_panel.set_text_color(1.0f,1.0f,1.0f);
        data_panel.set_border_color(0.0f,1.0f,0.0f);
        set_border_color(1.0f,1.0f,1.0f);
        data_cursor.set_color(1.0f,1.0f,0.0f);
        update_screen = true;
        break;

    }

    if (update_screen) {
        display_finished = true;
        glutPostRedisplay();
    }
}

} // end spy
} // end opengl
} // end cusp
