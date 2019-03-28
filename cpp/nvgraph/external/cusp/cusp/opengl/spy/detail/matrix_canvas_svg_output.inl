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
 * @file matrix_canvas_svg_output.cc
 * Functions to save the current view as an SVG file.
 */

/*
 * David Gleich
 * 1 August 2007
 * Copyright, Stanford University
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cusp/opengl/spy/matrix_canvas.h>

namespace cusp
{
namespace opengl
{
namespace spy
{

static void color2rgb(float *color, int& r, int &g, int &b)
{
    r = (int)(color[0]*255.0f);
    g = (int)(color[1]*255.0f);
    b = (int)(color[2]*255.0f);
}

template< typename IndexType, typename ValueType, typename MemorySpace >
void matrix_canvas<IndexType,ValueType,MemorySpace>::write_svg()
{
    std::cout << "writing matrix to spy.svg ... " << std::endl;
    FILE *svgfile = fopen("spy.svg", "wt");
    if (!svgfile) {
        printf("spy.svg not writable\n");
        return;
    }

    fprintf(svgfile, "<?xml version=\"1.0\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");

    float onepx = scale_to_world(1.0f);
    int r,g,b;

    {
        GLint     view[4];
   
        glGetIntegerv(GL_VIEWPORT, view);
        color2rgb(background_color, r, g, b);
        fprintf(svgfile, "<svg viewbox=\"%i %i %i %i\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"%d\" height=\"%d\">\n", 
            0, 0, view[2], view[3], view[2], view[3]);
        fprintf(svgfile, "<rect x=\"%i\" y=\"%i\" width=\"%i\" height=\"%i\" fill=\"rgb(%i,%i,%i)\" />\n", 
            0, 0, view[2], view[3], r,g,b);
    }

    {
        GLint     view[4];
        GLdouble  model[16], proj[16], total[16];

        glGetIntegerv(GL_VIEWPORT, view);
        glGetDoublev(GL_MODELVIEW_MATRIX, model);
        glGetDoublev(GL_PROJECTION_MATRIX, proj);

        fprintf(svgfile, "<g transform=\"translate(%lf,%lf)\">\n",
            (double)view[2]/2.0, (double)view[3]/2.0);

        fprintf(svgfile, "<g transform=\"scale(%lf,%lf)\">\n",
            (double)view[2]/2.0, (double)view[3]/-2.0);

        // compute the matrix product, matrices in column-major
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                // i is the column, j is the row
                GLdouble sum = 0.0;
                for (int k = 0; k < 4; k++)
                {
                    // dot product between row j in proj, and col i in model
                    sum += proj[4*k+j]*model[4*i+k];
                }
                total[i*4+j] = sum;
            }
        }

        fprintf(svgfile, "<g transform=\"matrix(%lf,%lf,%lf,%lf,%lf,%lf)\">\n",
            total[0], total[1], total[4], total[5], total[12], total[13]);
    }

    //
    // write the border
    // 
    int m = _m.num_rows;
    int n = _m.num_cols;

    color2rgb(border_color, r, g, b);    
    fprintf(svgfile, "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"%f\" stroke-opacity=\"%f\" stroke=\"rgb(%i,%i,%i)\" />\n",
                    -0.5f,-0.5f,-0.5f,m-0.5f,onepx/2,1.0,r,g,b);
    fprintf(svgfile, "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"%f\" stroke-opacity=\"%f\" stroke=\"rgb(%i,%i,%i)\" />\n",
                    -0.5f,m-0.5f,n-0.5f,m-0.5f,onepx/2,1.0,r,g,b);
    fprintf(svgfile, "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"%f\" stroke-opacity=\"%f\" stroke=\"rgb(%i,%i,%i)\" />\n",
                    n-0.5f,m-0.5f,n-0.5f,-0.5f,onepx/2,1.0,r,g,b);
    fprintf(svgfile, "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" stroke-width=\"%f\" stroke-opacity=\"%f\" stroke=\"rgb(%i,%i,%i)\" />\n",
                    n-0.5f,-0.5f,-0.5f,-0.5f,onepx/2,1.0,r,g,b);

    // 
    // write through the matrix
    //


    {
        ValueType v;
        int colormap_entry;

        //float pts=(zoom / (virtual_width*aspect/(float)width));

        float x1,y1,x2,y2;
        world_extents(x1,y1,x2,y2);

        int r1=(int)floor(y1),r2=(int)floor(y2);
        int c1=(int)floor(x1),c2=(int)floor(x2);

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

        if (normalization_state != no_normalization) {
            min_val = 0.0;
            max_val = 1.0f;
            inv_val_range = 1.0f;
        }

        for (int pi = std::max(0,r1); pi < std::min(r2, m); ++pi)
        {
            if (pi-std::max(0,r1)>0 && (pi-std::max(0,r1)) % 10000 == 0 &&
                std::min(r2, m) - std::max(0,r1) >= 20000) {
                std::cout << "  writing row " << (pi-std::max(0,r1)) 
                          << " of " << std::min(r2, m) - std::max(0,r1)
                          << std::endl;
            }
            // i is the real row in the matrix for the pith row
            // of the display
            int i=pi;
            if (permutation_state == row_permutation || 
                permutation_state == row_column_permutation) {
                i=irperm[i];
            }

            for (IndexType ri = _m.row_offsets[i]; ri < _m.row_offsets[i+1]; ++ri)
            {
                // j is the real column in the matrix for the pjth
                // column of the display
                int j = _m.column_indices[ri];
                int pj = j;
                if (permutation_state == column_permutation ||
                    permutation_state == row_column_permutation) {
                    pj = cperm[pj];
                }

                // skip all the columns outside
                if (pj < c1 || pj > c2) { continue; }

                v = _m.values[ri];
                if (normalization_state == row_normalization ||
                    normalization_state == row_column_normalization) {
                    v*=rnorm[i];
                }
                if (normalization_state == column_normalization ||
                    normalization_state == row_column_normalization) {
                    v*=cnorm[j];
                }

                // scale v to the range [0,1]
                v = v - min_val;
                v = v*inv_val_range;

                if (!colormap_invert) { 
                    colormap_entry = (int)(v*(colormap.size-1)); 
                } else { 
                    colormap_entry = (int)(v*(colormap.size-1)); 
                    colormap_entry=colormap.size-1-colormap_entry;
                }

                color2rgb(&colormap.map[colormap_entry*3],r,g,b);

                fprintf(svgfile, "<circle cx=\"%g\" cy=\"%g\" r=\"%g\" fill=\"rgb(%i,%i,%i)\" opacity=\"%g\"/>\n",
                    (float)pi,(float)pj, (std::max)(0.5f,onepx/2.0f), r,g,b, alpha);
            }
        }

    }

    fprintf(svgfile,"</g>\n");
    fprintf(svgfile,"</g>\n");
    fprintf(svgfile,"</g>\n");
    fprintf(svgfile, "</svg>\n");

    fclose(svgfile);
}

} // end spy
} // end opengl
} // end cusp
