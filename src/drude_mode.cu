#include "drude_mode.h"
#include "constants.h"
/**
  Main entry point for the fdtd calculations

  @param d The datablock structure
  @param ticks Represents the number of times the function has run.
*/
void anim_gpu_drude(Datablock *d, int ticks){
    assert(d != NULL);
    float err = cudaEventRecord(d->start, 0);

    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    dim3 source_threads(64, 1);
    dim3 source_blocks((d->sources->size + 63) / 64, 1);
    CPUAnimBitmap *bitmap = d->bitmap;
    static long time_ticks = 0;

    for(int i=0;i<2;i++){
        time_ticks += 1;
        make_copy<<<blocks, threads>>>(d->fields[DRUDE_EZOLD],
                                       d->fields[DRUDE_EZFIELD]);

        update_drude_ez<<<blocks, threads>>>(d->fields[DRUDE_EZFIELD],
                                             d->fields[DRUDE_HXFIELD],
                                             d->fields[DRUDE_HYFIELD],
                                             d->fields[DRUDE_JFIELD],
                                             d->coefs[2],
                                             d->coefs[3],
                                             d->coefs[4]);

        copy_sources<<<source_blocks, source_threads>>>(
                d->fields[DRUDE_EZFIELD],
                d->sources->x_source_position,
                d->sources->y_source_position,
                d->sources->source_type,
                d->sources->mean,
                d->sources->variance,
                d->sources->size,
                time_ticks);

        update_drude_jz<<<blocks, threads>>>(d->fields[DRUDE_JFIELD],
                                             d->fields[DRUDE_EZFIELD],
                                             d->fields[DRUDE_EZOLD],
                                             d->coefs[5],
                                             d->coefs[6]);

        update_Hx<<<blocks, threads>>>(d->fields[DRUDE_HXFIELD],
                                        d->fields[DRUDE_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);

        update_Hy<<<blocks, threads>>>(d->fields[DRUDE_HYFIELD],
                                        d->fields[DRUDE_EZFIELD],
                                        d->coefs[0],
                                        d->coefs[1]);
    }
    float_to_color<<<blocks, threads>>> (d->output_bitmap,
            d->fields[DRUDE_EZFIELD]);

    checkCudaErrors(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap,
                bitmap->image_size(), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(d->stop, 0) );
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime = 1;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}

void drude_clear_memory_constants(Datablock *d){
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
    cudaFree(d->constants[GAMMA_INDEX]);
    cudaFree(d->constants[OMEGAP_INDEX]);
}

void clear_memory_drude_simulation(Datablock *d){
    cudaFree(d->fields[DRUDE_EZFIELD]);
    cudaFree(d->fields[DRUDE_HXFIELD]);
    cudaFree(d->fields[DRUDE_HYFIELD]);
    cudaFree(d->fields[DRUDE_JFIELD]);
    cudaFree(d->fields[DRUDE_EZOLD]);
    cudaFree(d->coefs[0]);
    cudaFree(d->coefs[1]);
    cudaFree(d->coefs[2]);
    cudaFree(d->coefs[3]);
    cudaFree(d->coefs[4]);
    cudaFree(d->coefs[5]);
    cudaFree(d->coefs[6]);
    cudaFree(d->sources->x_source_position);
    cudaFree(d->sources->y_source_position);
    cudaFree(d->sources->source_type);
    cudaFree(d->sources->mean);
    cudaFree(d->sources->variance);
    checkCudaErrors(cudaEventDestroy(d->start) );
    checkCudaErrors(cudaEventDestroy(d->stop) );
}

void allocate_drude_memory(Datablock *data, Structure structure){
    printf("The size of the structure is %ld \n", structure.size());
    printf("Allocation Memory\n");

    checkCudaErrors(cudaMalloc( (void **) &data->output_bitmap,
                    structure.size()));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[DRUDE_EZFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[DRUDE_HXFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[DRUDE_HYFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[DRUDE_JFIELD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->fields[DRUDE_EZOLD],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[MUINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[EPSINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMAINDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[SIGMA_STAR_INDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[OMEGAP_INDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->constants[GAMMA_INDEX],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[0],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[1],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[2],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[3],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[4],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[5],
                    structure.size() ));
    checkCudaErrors(cudaMalloc( (void **) &data->coefs[6],
                    structure.size() ));
}

void initialize_drude_arrays(Datablock *data, Structure structure){
    int size = structure.grid_size();
    printf("%ld\n", size);
    printf("%ld\n", structure.x_index_dim);
    printf("%ld\n", structure.y_index_dim);
    printf("Initializing arrays\n");

    // FIXME: Temporary fix for populating values.

    float * temp = (float *) malloc(structure.size());
    std::fill_n(temp, size, MU);
    cudaMemcpy(data->constants[MUINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, EPSILON);
    cudaMemcpy(data->constants[EPSINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMAINDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 0.0);
    cudaMemcpy(data->constants[SIGMA_STAR_INDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 2.0 * PI * 2e15);
    cudaMemcpy(data->constants[OMEGAP_INDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    std::fill_n(temp, size, 57e12);
    cudaMemcpy(data->constants[GAMMA_INDEX],temp,structure.size(),
                cudaMemcpyHostToDevice);

    thrust::device_ptr<float> hx_field_ptr(data->fields[DRUDE_EZFIELD]);
    thrust::fill(hx_field_ptr, hx_field_ptr + size, 0);

    thrust::device_ptr<float> ezold_field_ptr(data->fields[DRUDE_EZOLD]);
    thrust::fill(ezold_field_ptr, ezold_field_ptr + size, 0);

    thrust::device_ptr<float> hy_field_ptr(data->fields[DRUDE_HXFIELD]);
    thrust::fill(hy_field_ptr, hy_field_ptr + size, 0);

    thrust::device_ptr<float> ez_field_ptr(data->fields[DRUDE_HYFIELD]);
    thrust::fill(ez_field_ptr, ez_field_ptr + size, 0);

    thrust::device_ptr<float> j_field_ptr(data->fields[DRUDE_JFIELD]);
    thrust::fill(j_field_ptr, j_field_ptr + size, 0);
}
