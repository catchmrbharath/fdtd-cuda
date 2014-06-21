/*! @file drude_mode.cu
   @author Bharath M R

   @brief Contains the functions for the drude mode.
*/
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

    for(int i=0;i<100;i++){
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

    checkCudaErrors(cudaMemcpy2D(bitmap->get_ptr(),
                                sizeof(float) * d->structure->x_index_dim,
                                d->output_bitmap,
                                d->structure->pitch,
                                sizeof(float) * d->structure->x_index_dim,
                                d->structure->y_index_dim,
                                cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(d->stop, 0) );
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime = 1;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}

/*! @brief Clears the constants after the coefficients are calculated
  */
void drude_clear_memory_constants(Datablock *d){
    cudaFree(d->constants[SIGMAINDEX]);
    cudaFree(d->constants[SIGMA_STAR_INDEX]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
    cudaFree(d->constants[GAMMA_INDEX]);
    cudaFree(d->constants[OMEGAP_INDEX]);
}

/*! @ brief Clears the memory after the simulation. */

//TODO: Replace the calls with a for loop.
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

/*! @brief Allocates the memory for the simulation */
size_t allocate_drude_memory(Datablock *data, Structure structure){
    printf("The size of the structure is %ld \n", structure.size());
    printf("Allocation Memory\n");
    size_t pitch; //! pitch is the row size in bytes.

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

    for(int i = 0;i < 7; i++){
        checkCudaErrors(cudaMallocPitch( (void **) &data->coefs[i],
                    &pitch, sizeof(float) * structure.x_index_dim,
                    sizeof(float) * structure.y_index_dim ));
    }
    return pitch;
}

/*! @brief Allocates the memory for the simulation */
void initialize_drude_arrays(Datablock *data, Structure structure){
    long size = structure.grid_size();
    printf("%ld\n", size);
    printf("%d\n", structure.x_index_dim);
    printf("%d\n", structure.y_index_dim);
    printf("Initializing arrays\n");

    // FIXME: Temporary fix for populating values.
    float * temp = (float *)malloc(sizeof(float) * size);
    std::fill_n(temp, size, MU);
    checkCudaErrors(cudaMemcpy2D(data->constants[MUINDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, EPSILON);
    checkCudaErrors(cudaMemcpy2D(data->constants[EPSINDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMAINDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) * structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 0.0);
    checkCudaErrors(cudaMemcpy2D(data->constants[SIGMA_STAR_INDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));


    std::fill_n(temp, size, 2.0 * PI * 2e15);
    checkCudaErrors(cudaMemcpy2D(data->constants[OMEGAP_INDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));

    std::fill_n(temp, size, 57e12);
    checkCudaErrors(cudaMemcpy2D(data->constants[GAMMA_INDEX], structure.pitch,
                temp, sizeof(float) * structure.x_index_dim,
                sizeof(float) *  structure.x_index_dim,
                structure.y_index_dim,
                cudaMemcpyHostToDevice));
   dim3 blocks((data->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (data->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    initialize_array<<<blocks, threads>>>(data->fields[DRUDE_EZFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[DRUDE_EZOLD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[DRUDE_HXFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[DRUDE_HYFIELD], 0);
    initialize_array<<<blocks, threads>>>(data->fields[DRUDE_JFIELD], 0);
}
