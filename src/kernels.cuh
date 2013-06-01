/*! @file kernels.cuh
    @author Bharath M R
*/

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
__device__ __constant__ int x_index_dim; //! x dimension of the structure
__device__ __constant__ int y_index_dim; //! y dimension of the structure.
__device__ __constant__ float delta; //! The dx of the structure.
__device__ __constant__ float deltat; //! The dt of the structure
__device__ __constant__ size_t pitch; //! The pitch of the structure. Pitch is the row size in bytes.

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

/* PML TM mode definitions */

__global__ void pml_tm_get_coefs(float *mu,
                              float *epsilon,
                              float *sigma_x,
                              float *sigma_y,
                              float *sigma_star_x,
                              float *sigma_star_y,
                              float *coef1,
                              float * coef2,
                              float * coef3,
                              float * coef4,
                              float * coef5,
                              float * coef6,
                              float * coef7,
                              float * coef8);

__global__ void update_pml_ezx(float * Ezx, float *Hy,
                                float * coef1, float *coef2);


__global__ void update_pml_ezy(float * Ezy, float *Hx,
                                float * coef1, float *coef2);

__global__ void update_pml_ez(float * Ezx, float *Ezy, float *Ez);

__global__ void update_drude_ez(float *Ez,
                                float *Hx,
                                float * Hy,
                                float *Jz,
                                float *coef1,
                                float *coef2,
                                float *coef3);

__global__ void drude_get_coefs(float *mu,
                                float * epsilon,
                                float *sigma,
                                float * sigma_star,
                                float *gamma,
                                float *omegap,
                                float * coef1,
                                float * coef2,
                                float * coef3,
                                float * coef4,
                                float * coef5,
                                float * coef6,
                                float * coef7);

__global__ void update_drude_jz(float *Jz,
                                float *Eznew,
                                float *Ezold,
                                float *coefa,
                                float *coefb
                                );

__global__ void make_copy(float * E_old, float * E_new);
__global__ void initialize_array(float * field, float value);
#endif
