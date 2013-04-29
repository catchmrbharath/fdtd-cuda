#include "pml_mode.h"

void anim_gpu_pml_tm(Datablock *d, int ticks){
    checkCudaErrors(cudaEventRecord(d->start, 0) );
    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    dim3 source_threads(64, 1);
    dim3 source_blocks((d->sources->size + 63) / 64, 1);
    CPUAnimBitmap *bitmap = d->bitmap;
    static long time_ticks = 0;
    printf("time ticks = %ld", time_ticks);
    printf("time ticks = %ld", time_ticks);

    for(int i=0;i<100;i++){
        time_ticks += 1;
        copy_sources<<<source_blocks, source_threads>>>(
                d->fields[TM_PML_EZFIELD],
                d->sources->x_source_position,
                d->sources->y_source_position,
                d->sources->source_type,
                d->sources->mean,
                d->sources->variance,
                d->sources->size,
                time_ticks);

        update_Hx<<<blocks, threads>>>(d->fields[TM_PML_HXFIELD],
                                        d->fields[TM_PML_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Hy<<<blocks, threads>>>(d->fields[TM_PML_HYFIELD],
                                        d->fields[TM_PML_EZFIELD],
                                        d->coefs[2],
                                        d->coefs[3]);

        update_pml_ezx<<<blocks, threads>>>(d->fields[TM_PML_EZXFIELD],
                                            d->fields[TM_PML_HYFIELD],
                                            d->coefs[4],
                                            d->coefs[5]);

        update_pml_ezy<<<blocks, threads>>>(d->fields[TM_PML_EZYFIELD],
                                            d->fields[TM_PML_HXFIELD],
                                            d->coefs[6],
                                            d->coefs[7]);

        update_pml_ez<<<blocks, threads>>>(d->fields[TM_PML_EZXFIELD],
                                            d->fields[TM_PML_EZYFIELD],
                                            d->fields[TM_PML_EZFIELD]);
    }

    float_to_color<<<blocks, threads>>> (d->output_bitmap,
                                        d->fields[TM_PML_EZFIELD]);

    checkCudaErrors(cudaMemcpy2D(bitmap->get_ptr(),
                                sizeof(float) * d->structure->x_index_dim,
                                d->output_bitmap,
                                d->structure->pitch,
                                sizeof(float) * d->structure->x_index_dim,
                                d->structure->y_index_dim,
                                cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaEventRecord(d->stop, 0) );
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}

void tm_pml_clear_memory_constants(Datablock *d){
    cudaFree(d->constants[SIGMAINDEX_X]);
    cudaFree(d->constants[SIGMAINDEX_Y]);
    cudaFree(d->constants[SIGMA_STAR_INDEX_X]);
    cudaFree(d->constants[SIGMA_STAR_INDEX_Y]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
}

void clear_memory_TM_PML_simulation(Datablock *d){
    cudaFree(d->fields[TM_PML_EZFIELD]);
    cudaFree(d->fields[TM_PML_EZXFIELD]);
    cudaFree(d->fields[TM_PML_EZYFIELD]);
    cudaFree(d->fields[TM_HYFIELD]);
    cudaFree(d->fields[TM_HXFIELD]);
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
    cudaFree(d->coefs[0]);
    cudaFree(d->coefs[1]);
    cudaFree(d->coefs[2]);
    cudaFree(d->coefs[3]);
    cudaFree(d->coefs[4]);
    cudaFree(d->coefs[5]);
    cudaFree(d->coefs[6]);
    cudaFree(d->coefs[7]);
    cudaFree(d->sources->x_source_position);
    cudaFree(d->sources->y_source_position);
    cudaFree(d->sources->source_type);
    cudaFree(d->sources->mean);
    cudaFree(d->sources->variance);
    checkCudaErrors(cudaEventDestroy(d->start) );
    checkCudaErrors(cudaEventDestroy(d->stop) );
}

size_t tm_pml_allocate_memory(Datablock *data, Structure structure){
    size_t pitch;
    checkCudaErrors(cudaMallocPitch( (void **) &data->output_bitmap,
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));

    for(int i = 0;i < 5;i++){
        checkCudaErrors(cudaMallocPitch( (void **) &data->fields[i],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    }

    for(int i = 0; i < 6; i++){
        checkCudaErrors(cudaMallocPitch( (void **) &data->constants[i],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    }

    for(int i = 0;i < 8; i++){
        checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[i],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    }

    return pitch;

}

void tm_pml_initialize_arrays(Datablock *data, Structure structure){
    int size = structure.grid_size();
    printf("%ld\n", size);
    printf("%ld\n", structure.x_index_dim);
    printf("%ld\n", structure.y_index_dim);

    // FIXME: Temporary fix for populating values.
    float * temp = (float *) malloc(structure.size());
    std::fill_n(temp, size, MU);
    checkCudaErrors(cudaMemcpy2D(data->constants[MUINDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, EPSILON * 20);
    checkCudaErrors(cudaMemcpy2D(data->constants[EPSINDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMAINDEX_X], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMAINDEX_Y], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));


    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMA_STAR_INDEX_X], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMA_STAR_INDEX_Y], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    dim3 blocks((data->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (data->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    initialize_array<<<blocks, threads>>>(data->fields[TM_PML_HXFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[TM_PML_HYFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[TM_PML_EZFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[TM_PML_EZXFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[TM_PML_EZYFIELD], 0);
}
