/*! @file structure.h
 * @author Bharath M R
 * @brief Contains details about the structure.
 */
#ifndef _STRUCTURE_H_
#define _STRUCTURE_H_
#include "cuda.h"   
#include<vector>
#include "hostsources.h"
#include "stdio.h"

struct Structure{
    int x_index_dim; //! xdimension of the structure (array size).
    int y_index_dim; //! ydimension of the structure (array size).
    float courant;  //! courant factor used for the structure.
    float dx; //! The spatial step size.
    float dt; //! The time domain step size.
    long total_ticks; //! The number of ticks to stop at
    long save_ticks; //! Number of ticks after which the field values should be saved.
    char * name; //! Prefix of the saved file names.
    size_t pitch; //! row size in bytes after memory allocation
    HostSources * sources; //! array where sources are stored.

    Structure(int xindexdim, int yindexdim, float dxin, float dtin){
        x_index_dim = xindexdim;
        y_index_dim = yindexdim;
        dx = dxin;
        dt = dtin;
    }

    long size(){
        return (long)(x_index_dim * y_index_dim * 4);
    }

    long grid_size(){
        return (long) (x_index_dim * y_index_dim);
    }
};
#endif
