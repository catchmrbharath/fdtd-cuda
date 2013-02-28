#ifndef _STRUCTURE_H_
#define _STRUCTURE_H_
#include "cuda.h"
#include<vector>
#include "hostsources.h"
#include "stdio.h"

struct Structure{
    float xdim;
    float ydim;
    int x_index_dim;
    int y_index_dim;
    float courant;
    float dx;
    float dt;
    long total_ticks;
    long present_ticks;
    char * name;
    HostSources * sources;

    Structure(int xindexdim, int yindexdim, float dxin, float dtin){
        x_index_dim = xindexdim;
        y_index_dim = yindexdim;
        dx = dxin;
        dt = dtin;
    }

    int size(){
        return (long)(x_index_dim * y_index_dim * 4);
    }

    int grid_size(){
        long temp = x_index_dim * y_index_dim;
        printf("The size is %ld\n", temp);
        return (long) (x_index_dim * y_index_dim);
    }

    void set_sources(int x, int y, int source_type){
        //FIXME Add different types of sources.
        sources->x_source_position.push_back(x);
        sources->y_source_position.push_back(y);
        sources->source_type.push_back(0);
        sources->mean.push_back(1);
        sources->variance.push_back(0);
    }
};
#endif
