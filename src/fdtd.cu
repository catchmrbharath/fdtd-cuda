#include "cuda.h"
#include "book.h"
#include "cpu_anim.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "h5save.h"
#include<stdio.h>
#include<pthread.h>
#include "datablock.h"
#include "kernels.cu"


#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_VOL 1.0f
#define MIN_VOL 0.00001f
#define LIGHTSPEED 299792458
#define EPSILON 8.8541878176e-12f
#define MU 1.2566370614e-6f





void anim_gpu(Datablock *d, int ticks){
    checkCudaErrors(cudaEventRecord(d->start, 0) );
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;
    for(int i=0;i<100;i++){
        copy_const_kernel<<<blocks, threads>>>(d->fields[TE_EZFIELD],
                                                d->dev_const);

        update_Hx<<<blocks, threads>>>(d->fields[TE_HXFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->constants[SIGMA_STAR_INDEX],
                                        d->constants[MUINDEX]);

        update_Hy<<<blocks, threads>>>(d->fields[TE_HYFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->constants[SIGMA_STAR_INDEX],
                                        d->constants[MUINDEX]);

        update_Ez<<<blocks, threads>>>(d->fields[TE_HXFIELD],
                                        d->fields[TE_HYFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->constants[SIGMAINDEX],
                                        d->constants[EPSINDEX]);
    }
    float_to_color<<<blocks, threads>>> (d->output_bitmap,
                                        d->fields[TE_EZFIELD]);

    checkCudaErrors(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap,
                        bitmap->image_size(), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(d->stop, 0) );
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(Datablock *d){
    cudaFree(d->fields[TE_EZFIELD]);
    cudaFree(d->fields[TE_HYFIELD]);
    cudaFree(d->fields[TE_HXFIELD]);
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
    checkCudaErrors(cudaEventDestroy(d->start) );
    checkCudaErrors(cudaEventDestroy(d->stop) );
}

int main(){
    Datablock data(0);
    Structure structure;
    structure.x_index_dim = 1024;
    structure.y_index_dim = 1024;
    structure.dt= 0.5;
    structure.courant = 0.5;
    structure.dx =  (structure.dt * LIGHTSPEED) / structure.courant; 
    checkCudaErrors(cudaMemcpyToSymbol(x_index_dim, &structure.x_index_dim,
                    sizeof(structure.x_index_dim)));
    checkCudaErrors(cudaMemcpyToSymbol(y_index_dim, &structure.y_index_dim,
                    sizeof(structure.y_index_dim)));
    checkCudaErrors(cudaMemcpyToSymbol(delta, &structure.dx,
                    sizeof(structure.dx)));
    checkCudaErrors(cudaMemcpyToSymbol(deltat, &structure.dt,
                    sizeof(structure.dt)));
    CPUAnimBitmap bitmap(structure.x_index_dim, structure.x_index_dim,
                            &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    checkCudaErrors(cudaEventCreate(&data.start, 1) );
    checkCudaErrors(cudaEventCreate(&data.stop, 1) );

    checkCudaErrors(cudaMalloc( (void **) &data.output_bitmap,
                    bitmap.image_size()));
    checkCudaErrors(cudaMalloc( (void **) &data.fields[TE_EZFIELD], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.fields[TE_HYFIELD], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.fields[TE_HXFIELD], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.constants[MUINDEX], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.constants[EPSINDEX], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.constants[SIGMAINDEX], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.constants[SIGMA_STAR_INDEX], bitmap.image_size() ));
    checkCudaErrors(cudaMalloc( (void **) &data.dev_const, bitmap.image_size() ));

    float *temp = (float *) malloc(bitmap.image_size() );
    float *temp_mu = (float *) malloc(bitmap.image_size() );
    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp_mu[i + j * structure.x_index_dim] = MU;

    checkCudaErrors(cudaMemcpy(data.constants[MUINDEX], temp_mu, bitmap.image_size(),
                    cudaMemcpyHostToDevice));

    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp_mu[i + j * structure.x_index_dim] = EPSILON;

    checkCudaErrors(cudaMemcpy(data.constants[EPSINDEX], temp_mu, bitmap.image_size(),
                    cudaMemcpyHostToDevice));

    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp_mu[i + j * structure.x_index_dim] = 0;

    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp[i + j * structure.x_index_dim] = 0;
    checkCudaErrors(cudaMemcpy(data.constants[SIGMAINDEX], temp_mu, bitmap.image_size(),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(data.constants[SIGMA_STAR_INDEX], temp_mu, bitmap.image_size(),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(data.fields[TE_EZFIELD], temp, bitmap.image_size(),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(data.fields[TE_HXFIELD], temp, bitmap.image_size(),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(data.fields[TE_HYFIELD], temp, bitmap.image_size(),
                    cudaMemcpyHostToDevice));

    for(int i= 125; i< 129;i++)
        for(int j=125; j<129;j++)
        temp[256 * j + i] = 1;

    checkCudaErrors(cudaMemcpy(data.dev_const, temp, bitmap.image_size(),
                    cudaMemcpyHostToDevice));
    free(temp);
    free(temp_mu);
    bitmap.anim_and_exit( (void (*)(void *, int)) anim_gpu,
                            (void (*)(void *)) anim_exit);
}
