#ifndef __DATABLOCK__
#define __DATABLOCK__

#include<vector>
#include "structure.h"
#include "devicesources.h"
#include "hostsources.h"
#include "cpu_anim.h"
// This gives the best results
#define BLOCKSIZE_X 256
#define BLOCKSIZE_Y 1


#define BLOCKSIZE_HX 256
#define BLOCKSIZE_HY 1

#define TM_SIMULATION 0

#define TM_HXFIELD 0
#define TM_HYFIELD 1
#define TM_EZFIELD 2

#define MUINDEX 0
#define EPSINDEX 1
#define SIGMAINDEX 2
#define SIGMA_STAR_INDEX 3


#define TM_PML_SIMULATION 1
#define TM_PML_HXFIELD 0
#define TM_PML_HYFIELD 1
#define TM_PML_EZXFIELD 2
#define TM_PML_EZYFIELD 3
#define TM_PML_EZFIELD 4

#define SIGMAINDEX_X 2
#define SIGMA_STAR_INDEX_X 3
#define SIGMAINDEX_Y 4
#define SIGMA_STAR_INDEX_Y 5

struct Datablock{
    unsigned char *output_bitmap;
    float ** fields;
    float ** constants;
    float *field;
    float ** coefs;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
    int simulationType;
    Structure * structure;
    int number_of_sources;
    DeviceSources * sources;

    Datablock(int type){
        simulationType = type;
        if(type == TM_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 3);
            constants = (float **) malloc(sizeof(float *) * 4);
            coefs = (float **) malloc(sizeof(float *) * 4);
        }

        if(type == TM_PML_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 5);
            constants = (float **) malloc(sizeof(float *) * 6);
            coefs = (float **) malloc(sizeof(float *) * 8);
        }
    }

    ~Datablock(){
        free(fields);
        free(constants);
        free(coefs);
    }

};
#endif
