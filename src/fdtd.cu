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
#include <thrust/fill.h>
#include<algorithm>


void anim_gpu(Datablock *d, int ticks){
    checkCudaErrors(cudaEventRecord(d->start, 0) );
    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    dim3 source_threads(64, 1);
    dim3 source_blocks((d->sources->size + 63) / 64, 1);

    CPUAnimBitmap *bitmap = d->bitmap;

    for(int i=0;i<100;i++){
        copy_sources<<<source_blocks, source_threads>>>(
                d->fields[TE_EZFIELD],
                d->sources->x_source_position,
                d->sources->y_source_position,
                d->sources->source_type,
                d->sources->mean,
                d->sources->variance,
                d->sources->size);

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
    cudaFree(d->sources->x_source_position);
    cudaFree(d->sources->y_source_position);
    cudaFree(d->sources->source_type);
    cudaFree(d->sources->mean);
    cudaFree(d->sources->variance);
    checkCudaErrors(cudaEventDestroy(d->start) );
    checkCudaErrors(cudaEventDestroy(d->stop) );
}

void allocateTEMemory(Datablock *data, Structure structure){
    printf("The size of the structure is %ld", structure.size());

    checkCudaErrors(cudaMalloc( (void **) &data->output_bitmap,
                    structure.size()));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_EZFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_HYFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[TE_HXFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[MUINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[EPSINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMAINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMA_STAR_INDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[0],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[1],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[2],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[3],
                    structure.size() ));
}


void initializeArrays(Datablock *data, Structure structure){
    int size = structure.grid_size();
    printf("%ld\n", size);
    printf("%ld\n", structure.x_index_dim);
    printf("%ld\n", structure.y_index_dim);

    // FIXME: Temporary fix for populating values.

    float * temp = (float *) malloc(structure.size());
    std::fill_n(temp, size, MU);
    cudaMemcpy(data->constants[MUINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, EPSILON * 20);
    cudaMemcpy(data->constants[EPSINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMAINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMA_STAR_INDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    thrust::device_ptr<float> hx_field_ptr(data->fields[TE_HXFIELD]);
    thrust::fill(hx_field_ptr, hx_field_ptr + size, 0);

    thrust::device_ptr<float> hy_field_ptr(data->fields[TE_HYFIELD]);
    thrust::fill(hy_field_ptr, hy_field_ptr + size, 0);

    thrust::device_ptr<float> ez_field_ptr(data->fields[TE_EZFIELD]);
    thrust::fill(ez_field_ptr, ez_field_ptr + size, 0);

}

void copy_sources(HostSources * host_sources, DeviceSources *device_sources){
    int number_of_sources = host_sources->get_size();
    device_sources->size = number_of_sources;
    checkCudaErrors(cudaMalloc((void**)&device_sources->x_source_position,
                number_of_sources * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&device_sources->y_source_position,
                number_of_sources * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&device_sources->source_type,
                number_of_sources * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&device_sources->mean,
                number_of_sources * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_sources->variance,
                number_of_sources * sizeof(float)));

    if(number_of_sources != 0){
    int *host_source_ptr = &(host_sources->x_source_position[0]);
    checkCudaErrors(cudaMemcpy(device_sources->x_source_position, host_source_ptr,
                sizeof(int) * number_of_sources,cudaMemcpyHostToDevice));


    host_source_ptr = &(host_sources->y_source_position[0]);
    checkCudaErrors(cudaMemcpy(device_sources->y_source_position, host_source_ptr,
                sizeof(int) * number_of_sources,cudaMemcpyHostToDevice));


    host_source_ptr = &(host_sources->source_type[0]);
    checkCudaErrors(cudaMemcpy(device_sources->source_type, host_source_ptr,
                sizeof(int) * number_of_sources, cudaMemcpyHostToDevice));

    float * mean_ptr = &(host_sources->mean[0]);
    checkCudaErrors(cudaMemcpy(device_sources->mean, mean_ptr,
                sizeof(float) * number_of_sources, cudaMemcpyHostToDevice));

    float * variance_ptr = &(host_sources->variance[0]);
    checkCudaErrors(cudaMemcpy(device_sources->mean, mean_ptr,
                sizeof(float) * number_of_sources, cudaMemcpyHostToDevice));
    }
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

    allocateTEMemory(&data, structure);
    initializeArrays(&data, structure);


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

// set the sources
    HostSources host_sources;
    DeviceSources device_sources;
    host_sources.add_source(512, 512, 0, 0, 1);
    data.sources = &device_sources;
    copy_sources(&host_sources, &device_sources);

    bitmap.anim_and_exit( (void (*)(void *, int)) anim_gpu,
                            (void (*)(void *)) anim_exit);
}
