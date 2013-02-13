__device__ __constant__ int x_index_dim;
__device__ __constant__ int y_index_dim;
__device__ __constant__ float delta;
__device__ __constant__ float deltat;

__global__ void copy_const_kernel(float *iptr, const float *cptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * x_index_dim;
    if(cptr[offset] != 0){
        iptr[offset] = cptr[offset];
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



__global__ void te_getcoeff(float *mu,
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
