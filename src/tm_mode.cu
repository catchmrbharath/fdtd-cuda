/*! @file tm_mode.cu
    @brief TM mode simulation file

    Contains all the methods except the kernels for working
    on a TM mode simulation.
*/

#include "tm_mode.h"
#include "constants.h"
#include "h5save.h"
#include "fdtd.h"
#include "common_functions.h"

/*! @brief The wrapper function which updates all the kernels.
  */
void anim_gpu_tm(Datablock *d, int ticks){
    checkCudaErrors(cudaEventRecord(d->start, 0) );
    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    dim3 source_threads(64, 1);
    dim3 source_blocks((d->sources->size + 63) / 64, 1);

    CPUAnimBitmap *bitmap = d->bitmap;
    static long time_ticks = 0;
    printf("time ticks = %ld  ", time_ticks);

    for(int i=0;i<100;i++){
        time_ticks += 1;
        copy_sources<<<source_blocks, source_threads>>>(
                d->fields[TM_EZFIELD],
                d->sources->x_source_position,
                d->sources->y_source_position,
                d->sources->source_type,
                d->sources->mean,
                d->sources->variance,
                d->sources->size,
                time_ticks);

        update_Hx<<<blocks, threads>>>(d->fields[TM_HXFIELD],
                                        d->fields[TM_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Hy<<<blocks, threads>>>(d->fields[TM_HYFIELD],
                                        d->fields[TM_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Ez<<<blocks, threads>>>(d->fields[TM_HXFIELD],
                                        d->fields[TM_HYFIELD],
                                        d->fields[TM_EZFIELD],
                                        d->coefs[2],
                                        d->coefs[3]);
    }

    if(d->outputType == OUTPUT_HDF5)
    {
        // removed cudaMemcpy w/ d->fields[TM_EZFIELD]
        pthread_t thread;
        pthread_mutex_lock(&mutexcopy);
        checkCudaErrors(cudaMemcpy2D(d->save_field,
                                    sizeof(float) * d->structure->x_index_dim,
                                    d->fields[TM_EZFIELD], // changed from TM_PML_EZFIELD 
                                    d->structure->pitch,
                                    sizeof(float) * d->structure->x_index_dim,
                                    d->structure->y_index_dim,
                                    cudaMemcpyDeviceToHost));
        pthread_mutex_unlock(&mutexcopy);
        pthread_create(&thread, NULL, &create_new_dataset, (void *)d);
    }

    if(d->outputType == OUTPUT_ANIM)
    {
        float_to_color<<<blocks, threads>>> (d->output_bitmap,
                d->fields[TM_EZFIELD]);
        checkCudaErrors(cudaMemcpy2D(bitmap->get_ptr(),
                sizeof(float) * d->structure->x_index_dim,
                d->output_bitmap,
                d->structure->pitch,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyDeviceToHost));
    }

    d->present_ticks = time_ticks;
    printf("time ticks = %ld  ", time_ticks);
    checkCudaErrors(cudaEventRecord(d->stop, 0));
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}


void tm_clear_memory_constants(Datablock *d){
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
}

void clear_memory_TM_simulation(Datablock *d){
    cudaFree(d->fields[TM_EZFIELD]);
    cudaFree(d->fields[TM_HYFIELD]);
    cudaFree(d->fields[TM_HXFIELD]);
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

size_t allocateTMMemory(Datablock *data, Structure structure){
    size_t pitch;
    data->save_field = (float *) malloc(structure.size());
    checkCudaErrors(cudaMallocPitch( (void **) &data->output_bitmap,
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));

    checkCudaErrors(cudaMallocPitch( (void **) &data->fields[TM_EZFIELD],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->fields[TM_HYFIELD],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->fields[TM_HXFIELD],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->constants[MUINDEX],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->constants[EPSINDEX],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->constants[SIGMAINDEX],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->constants[SIGMA_STAR_INDEX],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[0],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[1],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[2],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[3],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    return pitch;
}


void initialize_TM_arrays(Datablock *d, Structure structure, ifstream &fs){
    long size = structure.grid_size();

    string epsname, muname, sigmaname;
    fs>>epsname;
    fs>>muname;
    fs>>sigmaname;

    initialize_eps_array(d, epsname);
    initialize_mu_array(d, muname);
    float * temp = (float *)malloc(sizeof(float) * size);
    parse_csv(sigmaname, temp, size);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMAINDEX], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMA_STAR_INDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    initialize_array<<<blocks, threads>>>(d->fields[TM_HXFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_HYFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_EZFIELD], 0);
}




void copy_sources_device_to_host(HostSources * host_sources, DeviceSources *device_sources){
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
    checkCudaErrors(cudaMemcpy(device_sources->variance, variance_ptr,
                sizeof(float) * number_of_sources, cudaMemcpyHostToDevice));
    }
}


