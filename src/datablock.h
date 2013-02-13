#ifndef __DATABLOCK__
#define __DATABLOCK__
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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
    thrust::host_vector<int> host_x_source_position;
    thrust::host_vector<int> host_y_source_position;
    thrust::device_vector<int> device_x_source_position;
    thrust::device_vector<int> device_y_source_position;
    thrust::host_vector<int> host_source_type;
    thrust::device_vector<int> device_source_type;

    Structure(int xindexdim, int yindexdim, float dxin, float dtin){
        x_index_dim =xindexdim;
        y_index_dim = yindexdim;
        dx = dxin;
        dt = dtin;
    }

    int size(){
        return x_index_dim * y_index_dim * 4;
    }

    void set_sources(int x, int y, int source_type){
        host_x_source_position.push_back(x);
        host_y_source_position.push_back(y);
        host_source_type.push_back(0);
    }
};


// This gives the best results
#define BLOCKSIZE_X 256
#define BLOCKSIZE_Y 1


#define BLOCKSIZE_HX 256
#define BLOCKSIZE_HY 1

#define TE_SIMULATION 0

#define TE_HXFIELD 0
#define TE_HYFIELD 1
#define TE_EZFIELD 2

#define MUINDEX 0
#define EPSINDEX 1
#define SIGMAINDEX 2
#define SIGMA_STAR_INDEX 3

struct Datablock{
    unsigned char *output_bitmap;
    float ** fields;
    float ** constants;
    float * dev_const;
    float *field;
    float ** coefs;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
    int simulationType;
    Structure * structure;

    Datablock(int type){
        simulationType = type;
        if(type == TE_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 3);
            constants = (float **) malloc(sizeof(float *) * 4);
            coefs = (float **) malloc(sizeof(float *) * 4);
        }
    }

    ~Datablock(){
        free(fields);
        free(constants);
    }

};
#endif
