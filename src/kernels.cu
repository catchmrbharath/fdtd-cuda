#include "hostsources.h"
#include "kernels.cuh"


// TODO: Add gaussian sources.
__global__ void copy_sources(float * target, int * x_position, int *y_position,
                            int * type, float * mean, float * variance,
                            int sources_size, long time_ticks) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<sources_size){
        int x = x_position[i];
        int y = y_position[i];
        if (type[i] == CONSTANT_SOURCE )
            target[x + y * x_index_dim] = variance[i];
        else if (type[i] == SINUSOID_SOURCE){
            float temp = sinf(mean[i] * time_ticks * deltat);
            float temp2 = variance[i];
            target[x + y * x_index_dim] = temp2 * temp;
        }
        else
            target[x + y * x_index_dim] = 1;
    }
    __syncthreads();
}
__global__ void update_Hx(float *Hx, float *Ez, float *coef1, float* coef2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * x_index_dim;
    int top = offset + x_index_dim;
    if(y < y_index_dim -1)
        Hx[offset] = coef1[offset] * Hx[offset]
                        - coef2[offset] * (Ez[top] - Ez[offset]);
    __syncthreads();
}

__global__ void update_Hy(float *Hy, float *Ez, float * coef1, float * coef2){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * x_index_dim;
    int right = offset + 1;
    if(x < x_index_dim -1)
        Hy[offset] = coef1[offset] * Hy[offset] + 
                        coef2[offset] * (Ez[right] - Ez[offset]);
    __syncthreads();
}

__global__ void update_Ez(float *Hx, float *Hy, float *Ez, float * coef1,
                            float *coef2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int bottom = offset - x_index_dim;

    if (x > 0 && y > 0 && x<x_index_dim - 1 && y < y_index_dim - 1){
        Ez[offset] = coef1[offset] * Ez[offset] +
                    coef2[offset] * ((Hy[offset] - Hy[left]) -
                                    (Hx[offset] - Hx[bottom]));
    }

    __syncthreads();
}



__global__ void tm_getcoeff(float *mu,
                                float * epsilon,
                                float *sigma,
                                float * sigma_star,
                                float * coef1,
                                float * coef2,
                                float * coef3,
                                float * coef4){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float mus = mu[offset];
    float sigmamstar = sigma_star[offset];
    float sigmam = sigma[offset];
    float eps = epsilon[offset];
    coef1[offset] = (2.0 * mus - sigmamstar * deltat) /
                        (2.0 * mus + sigmamstar * deltat);
    coef2[offset] = (2 * deltat) / ((2 * mus + sigmamstar * deltat) * delta);

    coef3[offset] = (2.0 * eps - sigmam * deltat) /
                        (2.0 * eps + sigmam * deltat);
    coef4[offset] = (2.0 * deltat) /
                    ((2 * eps + sigmam * deltat) * delta);
    __syncthreads();
}

__device__ unsigned char value( float n1, float n2, int hue ) {
    if (hue > 360)      hue -= 360;
    else if (hue < 0)   hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
    return (unsigned char)(255 * n1);
}

__global__ void float_to_color( unsigned char *optr,
                              const float *outSrc ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset*4 + 0] = value( m1, m2, h+120 );
    optr[offset*4 + 1] = value( m1, m2, h );
    optr[offset*4 + 2] = value( m1, m2, h -120 );
    optr[offset*4 + 3] = 255;
}

__global__ void float_to_color( uchar4 *optr,
                              const float *outSrc ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset].x = value( m1, m2, h+120 );
    optr[offset].y = value( m1, m2, h );
    optr[offset].z = value( m1, m2, h -120 );
    optr[offset].w = 255;
}

void copy_symbols(Structure *structure){
    checkCudaErrors(cudaMemcpyToSymbol(x_index_dim, &(structure->x_index_dim),
                    sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(y_index_dim, &(structure->y_index_dim),
                    sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(delta, &(structure->dx),
                    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(deltat, &(structure->dt),
                    sizeof(float)));
}

