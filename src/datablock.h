#ifndef __DATABLOCK__
#define __DATABLOCK__
typedef struct {
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
} Structure;


typedef struct {
    unsigned char *output_bitmap;
    float *dev_Ez;
    float *dev_Hx;
    float *dev_Hy;
    float *dev_eps;
    float *dev_mu;
    float *dev_sigma;
    float *dev_sigmastar;
    float *dev_const;
    float *field;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
    Structure * structure;
} Datablock;
#endif
