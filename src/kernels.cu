/*! @file kernels.cu
    @author Bharath M R
*/
#include "kernels.cuh"


// TODO: Add gaussian sources.
/*! @brief Copies the sources from the sources array to the Ez position */
__global__ void copy_sources(float * target, int * x_position, int *y_position,
                            int * type, float * mean, float * variance,
                            int sources_size, long time_ticks) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<sources_size){
        int x = x_position[i];
        int y = y_position[i];
        int offset = x + y * pitch / sizeof(float);
        if (type[i] == CONSTANT_SOURCE )
            target[offset] = variance[i];
        else if (type[i] == SINUSOID_SOURCE){
            float temp = sinf(mean[i] * time_ticks * deltat);
            float temp2 = variance[i];
            target[offset] = temp2 * temp;
        }
        else
            target[offset] = __expf(time_ticks);
    }
    __syncthreads();
}
/*! @brief Hx updates for the TM mode.

  @params Hx - Hx array
  @params Ez - Ez array
*/
__global__ void update_Hx(float *Hx, float *Ez, float *coef1, float* coef2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    int top = offset + pitch / sizeof(float);
    if(y < y_index_dim - 1)
        Hx[offset] = coef1[offset] * Hx[offset]
                        - coef2[offset] * (Ez[top] - Ez[offset]);
    __syncthreads();
}

/*! @brief Hy updates for TM mode.
  */
__global__ void update_Hy(float *Hy, float *Ez, float * coef1, float * coef2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    int right = offset + 1;
    if(x < x_index_dim -1)
        Hy[offset] = coef1[offset] * Hy[offset] + 
                        coef2[offset] * (Ez[right] - Ez[offset]);
    __syncthreads();
}

/*! @brief Ez updates for TM mode.
  */
__global__ void update_Ez(  float *Hx, 
                            float *Hy, 
                            float *Ez, 
                            float *coef1,
                            float *coef2)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);

    int left = offset - 1;
    int bottom = offset - pitch / sizeof(float);

    if (x > 0 && y > 0 && x<x_index_dim - 1 && y < y_index_dim - 1){
        Ez[offset] = coef1[offset] * Ez[offset] +
                    coef2[offset] * ((Hy[offset] - Hy[left]) -
                                    (Hx[offset] - Hx[bottom]));
    }
    __syncthreads();
}



/*! @brief Calculates tm mode coefficients.
  */
__global__ void tm_getcoeff(float *mu,
                            float * epsilon,
                            float *sigma,
                            float * sigma_star,
                            float * coef1,
                            float * coef2,
                            float * coef3,
                            float * coef4)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
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

/*! @brief Converts HSL value to RGB value */
__device__ unsigned char value( float n1, float n2, int hue ) 
{
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

/*! @brief Converts a floating point value to color */
__global__ void float_to_color( unsigned char *optr,
                              const float *outSrc ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);

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

/*! @brief Converts a floating point value to color */
__global__ void float_to_color( uchar4 *optr,
                              const float *outSrc ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);

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

/* @brief copies the constants to constant memory */
void copy_symbols(Structure *structure){
    checkCudaErrors(cudaMemcpyToSymbol(x_index_dim, &(structure->x_index_dim),
                    sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(y_index_dim, &(structure->y_index_dim),
                    sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(delta, &(structure->dx),
                    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(deltat, &(structure->dt),
                    sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(pitch, &(structure->pitch), sizeof(size_t)));
}


/* PML TM mode functions start here */

// Bad design. Too many arguments to the functions. Can't help it.
// FIXME sometime
/*! @brief TM Mode with PML */
__global__ void pml_tm_get_coefs(float *mu,
                              float *epsilon,
                              float *sigma_x,
                              float *sigma_y,
                              float *sigma_star_x,
                              float *sigma_star_y,
                              float *coef1,
                              float *coef2,
                              float *coef3,
                              float *coef4,
                              float *coef5,
                              float *coef6,
                              float *coef7,
                              float *coef8)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    float mus = mu[offset];
    float eps = epsilon[offset];
    float sigma_x_value = sigma_x[offset];
    float sigma_y_value = sigma_y[offset];
    float sigma_star_x_value = sigma_star_x[offset];
    float sigma_star_y_value = sigma_star_y[offset];
    coef1[offset] = (2.0 * mus - sigma_star_x_value * deltat) /
                    (2.0 * mus + sigma_star_x_value * deltat);

    coef2[offset] = (2.0 * deltat) / ((2 * mus + sigma_star_x_value *deltat)
                    * delta);

    coef3[offset] = (2.0 * mus - sigma_star_y_value * deltat) /
                    (2.0 * mus + sigma_star_y_value * deltat);

    coef4[offset] = (2 * deltat) / ( (2 * mus +
                    sigma_star_y_value *deltat) * delta);

    coef5[offset] = (2.0 * eps - sigma_x_value * deltat) /
                    (2.0 * eps + sigma_x_value * deltat);

    coef6[offset] = (2.0 * deltat) /
                    ((2 * eps + sigma_x_value * deltat) * delta);

    coef7[offset] = (2.0 * eps - sigma_y_value * deltat) /
                    (2.0 * eps + sigma_y_value * deltat);

    coef8[offset] = (2.0 * deltat) /
                    ((2 * eps + sigma_y_value * deltat) * delta);
}

/*! @brief PML Ezx update.

  The pitch is used for handling non multiples of 32 grids.
*/
__global__ void update_pml_ezx(float * Ezx, float *Hy,
                                float * coef1, float *coef2){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    int left = offset - 1;

    if (x > 0 && y > 0 && x<x_index_dim - 1 && y < y_index_dim - 1){
        Ezx[offset] = coef1[offset] * Ezx[offset] +
                      coef2[offset] * (Hy[offset] - Hy[left]);
    }
    __syncthreads();
}

/*!
  @brief PML Ezy update.

  See info about pitch at the top of the file.
*/
__global__ void update_pml_ezy(float * Ezy, float *Hx,
                                float * coef1, float *coef2){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    int bottom = offset - pitch / sizeof(float);

    if (x > 0 && y > 0 && x<x_index_dim - 1 && y < y_index_dim - 1){
        Ezy[offset] = coef1[offset] * Ezy[offset] -
                      coef2[offset] * (Hx[offset] - Hx[bottom]);
    }
    __syncthreads();
}

/*! @brief PML Ez update.
  
  Ez = Ezx + Ezy
*/
__global__ void update_pml_ez(float * Ezx, float *Ezy, float *Ez){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    Ez[offset] = Ezx[offset] + Ezy[offset];
    __syncthreads();
}


/*! @brief Copy data from one source to another.
    @deprecated
  Not really used anywhere. memCpy replaces this.
*/
__global__ void make_copy(float * E_old, float * E_new){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    E_old[offset] = E_new[offset];
    __syncthreads();
}

// FIXME Need better names for coef1. Is confusing.
/*! @brief Drude Ez update.

   Drude model uses the same Hx and Hy updates as that
   of TM mode.
*/

__global__ void update_drude_ez(float *Ez,
                                float *Hx,
                                float * Hy,
                                float *Jz,
                                float *coef1,
                                float *coef2,
                                float *coef3){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    int left = offset - 1;
    int bottom = offset - pitch / sizeof(float);

    if (x > 0 && y > 0 && x < x_index_dim - 1 && y < y_index_dim - 1){
        Ez[offset] = coef1[offset] * Ez[offset] +
                    coef2[offset] * (((Hy[offset] - Hy[left]) -
                                    (Hx[offset] - Hx[bottom])) / delta -
                    coef3[offset] * Jz[offset]);
    }
    __syncthreads();
}

/*!
  @brief: Jz update for drude model.
*/
__global__ void update_drude_jz(float *Jz,
                                float *Eznew,
                                float *Ezold,
                                float *coefa,
                                float *coefb
                                ){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    Jz[offset] = coefa[offset] * Jz[offset] +
                 coefb[offset] * (Eznew[offset] + Ezold[offset]);
    __syncthreads();

}


/* kernels for gain materials */

/*! @brief Calculates coefficients for drude model. */
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
                                float * coef7){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    float mus = mu[offset];
    float sigmamstar = sigma_star[offset];
    float sigmam = sigma[offset];
    float eps = epsilon[offset];
    float gamma_local = gamma[offset];
    float omegap_local = omegap[offset];


    float kp = (1 - gamma_local * 0.5 * deltat) /
                (1 + gamma_local * 0.5 * deltat);

    float betap = (omegap_local * omegap_local * eps * deltat * 0.5) /
                  (1 + gamma_local * 0.5 * deltat);


    coef1[offset] = (2.0 * mus - sigmamstar * deltat) /
                        (2.0 * mus + sigmamstar * deltat);
    coef2[offset] = (2 * deltat) / ((2 * mus + sigmamstar * deltat) * delta);

    coef3[offset] = (2 * eps - deltat * (betap + sigmam)) /
                    (2 * eps + deltat * (betap + sigmam));

    coef4[offset] = (2.0 * deltat) /
                    ((2.0 * eps + deltat * (betap + sigmam)));
    coef5[offset] = 0.5 * (1 + kp);
    coef6[offset] = kp;
    coef7[offset] = betap;

    __syncthreads();
}


/*@brief Initializes an array to a particular value.

  Used to set initial values for fields.
*/

__global__ void initialize_array(float * field, float value){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * pitch / sizeof(float);
    field[offset] = value;

}
