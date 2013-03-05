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

void tm_pml_allocate_memory(Datablock *data, Structure structure){

    checkCudaErrors(cudaMalloc( (void **) &data->output_bitmap,
                    structure.size()));
    for(int i = 0;i < 5;i++){
    checkCudaErrors(cudaMalloc( (void **) &data->fields[i],
                    structure.size() ));
    }

    for(int i = 0; i < 6; i++){
    checkCudaErrors(cudaMalloc( (void **) &data->constants[i],
                    structure.size() ));
    }

    for(int i = 0;i < 8; i++){
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[i],
                    structure.size() ));
    }

}

void tm_pml_initialize_arrays(Datablock *data, Structure structure){
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
    cudaMemcpy(data->constants[SIGMAINDEX_X],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMAINDEX_Y],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMA_STAR_INDEX_X],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMA_STAR_INDEX_Y],temp,structure.size(),
                cudaMemcpyHostToDevice);

    thrust::device_ptr<float> hx_field_ptr(data->fields[TM_PML_HXFIELD]);
    thrust::fill(hx_field_ptr, hx_field_ptr + size, 0);

    thrust::device_ptr<float> hy_field_ptr(data->fields[TM_PML_HYFIELD]);
    thrust::fill(hy_field_ptr, hy_field_ptr + size, 0);

    thrust::device_ptr<float> ez_field_ptr(data->fields[TM_PML_EZFIELD]);
    thrust::fill(ez_field_ptr, ez_field_ptr + size, 0);

    thrust::device_ptr<float> ezx_field_ptr(data->fields[TM_PML_EZXFIELD]);
    thrust::fill(ezx_field_ptr, ezx_field_ptr + size, 0);

    thrust::device_ptr<float> ezy_field_ptr(data->fields[TM_PML_EZYFIELD]);
    thrust::fill(ezy_field_ptr, ezy_field_ptr + size, 0);
}
