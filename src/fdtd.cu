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
#include "constants.h"


void anim_gpu(Datablock *d, int ticks){
    checkCudaErrors(cudaEventRecord(d->start, 0) );
    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);

    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    CPUAnimBitmap *bitmap = d->bitmap;
    for(int i=0;i<100;i++){
        copy_const_kernel<<<blocks, threads>>>(d->fields[TE_EZFIELD],
                                                d->dev_const);

        update_Hx<<<blocks, threads>>>(d->fields[TE_HXFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Hy<<<blocks, threads>>>(d->fields[TE_HYFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Ez<<<blocks, threads>>>(d->fields[TE_HXFIELD],
                                        d->fields[TE_HYFIELD],
                                        d->fields[TE_EZFIELD],
                                        d->coefs[2],
                                        d->coefs[3]);
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
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}

void anim_exit(Datablock *d){
    cudaFree(d->fields[TE_EZFIELD]);
    cudaFree(d->fields[TE_HYFIELD]);
    cudaFree(d->fields[TE_HXFIELD]);
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
    cudaFree(d->coefs[0]);
    cudaFree(d->coefs[1]);
    cudaFree(d->coefs[2]);
    cudaFree(d->coefs[3]);
    cudaFree(d->dev_const);
    checkCudaErrors(cudaEventDestroy(d->start) );
    checkCudaErrors(cudaEventDestroy(d->stop) );
}

void allocateTEMemory(Datablock *data, Structure *structure){
    printf("The size of the structure is %d", structure->size());

    checkCudaErrors(cudaMalloc( (void **) &data->output_bitmap,
                    structure->size()));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_EZFIELD],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_HYFIELD],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_HXFIELD],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[MUINDEX],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[EPSINDEX],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMAINDEX],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMA_STAR_INDEX],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->dev_const,
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[0],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[1],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[2],
                    structure->size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[3],
                    structure->size() ));

}

int main(){
    Datablock data(TE_SIMULATION);
    float dt= 0.5;
// FIXME: check the courant factor for the max epsilon.
    float courant = 0.5;
    float dx =  (dt * LIGHTSPEED) / courant;
    Structure structure(1024, 1024, dx, dt);


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
    data.structure = &structure;
    checkCudaErrors(cudaEventCreate(&data.start, 1) );
    checkCudaErrors(cudaEventCreate(&data.stop, 1) );
    allocateTEMemory(&data, &structure);


    float *temp = (float *) malloc(bitmap.image_size() );
    float *temp_mu = (float *) malloc(bitmap.image_size() );
    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp_mu[i + j * structure.x_index_dim] = MU;

    checkCudaErrors(cudaMemcpy(data.constants[MUINDEX], temp_mu, bitmap.image_size(),
                    cudaMemcpyHostToDevice));

    for(int i=0;i<structure.x_index_dim;i++)
        for(int j=0;j<structure.y_index_dim;j++)
            temp_mu[i + j * structure.x_index_dim] = EPSILON * 20;

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

//  get the coefficients

    dim3 blocks((structure.x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (structure.y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    te_getcoeff<<<blocks, threads>>>(data.constants[0],
                                     data.constants[1],
                                     data.constants[2],
                                     data.constants[3],
                                     data.coefs[0],
                                     data.coefs[1],
                                     data.coefs[2],
                                     data.coefs[3]);
    cudaFree(data.constants[0]);
    cudaFree(data.constants[1]);
    cudaFree(data.constants[2]);
    cudaFree(data.constants[3]);


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
