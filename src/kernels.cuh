#ifndef __KERNELS__
#define __KERNELS__
#include "structure.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "devicesources.h"
#include "hostsources.h"

/* Be extremely careful with the use of constant device variables
   They are static i.e. they cannot be exposed to other functions in
   other files and they will be initiated to zero if not handled properly.
*/
__device__ __constant__ int x_index_dim;
__device__ __constant__ int y_index_dim;
__device__ __constant__ float delta;
__device__ __constant__ float deltat;

__global__ void copy_sources(float * target, int * x_position, int *y_position,
                            int * type, float * mean, float * variance,
                            int sources_size, long time_ticks);
__global__ void update_Hx(float *Hx, float *Ez, float *coef1, float* coef2);
__global__ void update_Hy(float *Hy, float *Ez, float * coef1, float * coef2);
__global__ void update_Ez(float *Hx, float *Hy, float *Ez, float * coef1,
                            float *coef2);
__global__ void tm_getcoeff(float *mu,
                                float * epsilon,
                                float *sigma,
                                float * sigma_star,
                                float * coef1,
                                float * coef2,
                                float * coef3,
                                float * coef4);

__global__ void float_to_color( unsigned char *optr,
                              const float *outSrc);
__device__ unsigned char value( float n1, float n2, int hue );
__global__ void float_to_color( uchar4 *optr,
                              const float *outSrc );

void copy_symbols(Structure *structure);
void copy_sources(HostSources * host_sources, DeviceSources *device_sources);
#endif
