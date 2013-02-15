#ifndef __DATABLOCK__
#define __DATABLOCK__
#include<vector>
#include "structure.h"
#include "devicesources.h"
#include "hostsources.h"
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
    int number_of_sources;
    DeviceSources sources;

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
