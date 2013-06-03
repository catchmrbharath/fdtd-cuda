#include "pml_mode.h"
#include "h5save.h"
#include "fdtd.h"
#include<pthread.h>
/*! @brief TM mode iteration function
  This is the function which runs all the updates. It runs
  the updates for a certain number of iterations and returns
  the results
*/

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

    for(int i=0;i<50;i++){
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


    d->present_ticks = time_ticks;
    pthread_t thread;
    /*Copy back to cpu memory */
    /*Create a lock */
    pthread_mutex_lock(&mutexcopy);
    checkCudaErrors(cudaMemcpy2D(d->save_field,
                                sizeof(float) * d->structure->x_index_dim,
                                d->fields[TM_PML_EZFIELD],
                                d->structure->pitch,
                                sizeof(float) * d->structure->x_index_dim,
                                d->structure->y_index_dim,
                                cudaMemcpyDeviceToHost));
    pthread_mutex_unlock(&mutexcopy);

    pthread_create(&thread, NULL, &create_new_dataset, (void *)d);
    create_new_dataset(d);
    checkCudaErrors(cudaEventRecord(d->stop, 0) );
    checkCudaErrors(cudaEventSynchronize(d->stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", elapsedTime);
}

/*! @brief: Frees cuda memory for constants TM PML mode */
void tm_pml_clear_memory_constants(Datablock *d){
    cudaFree(d->constants[SIGMAINDEX_X]);
    cudaFree(d->constants[SIGMAINDEX_Y]);
    cudaFree(d->constants[SIGMA_STAR_INDEX_X]);
    cudaFree(d->constants[SIGMA_STAR_INDEX_Y]);
    cudaFree(d->constants[EPSINDEX]);
    cudaFree(d->constants[MUINDEX]);
}

/*! @brief Frees the cuda memory for everything */
//FIXME Convert this to a for loop.
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

/*! @brief Allocates memory for simulation

  Returns
  @params Returns the pitch which is the row length in bytes. Used in
  accessing two dimensional data.
*/
size_t tm_pml_allocate_memory(Datablock *data, Structure structure){
    size_t pitch;
    data->save_field = (float *) malloc(structure.size());
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

/*! @brief Initializes the arrays from the configuration file.

    @param d The datablock which stores the fields and coeficients
    @param structure Stores the dimensions of the file.
    @param fs The input file stream of the configuration file.
*/
void tm_pml_initialize_arrays(Datablock *d, Structure structure, ifstream &fs){
    int size = structure.grid_size();
    printf("%ld\n", size);
    printf("%ld\n", structure.x_index_dim);
    printf("%ld\n", structure.y_index_dim);

    string epsname, muname, sigma_x_name, sigma_y_name;
    string sigma_star_x_name, sigma_star_y_name;
    fs>>epsname;
    fs>>muname;
    fs>>sigma_x_name;
    fs>>sigma_y_name;
    fs>>sigma_star_x_name;
    fs>>sigma_star_y_name;

    initialize_eps_array(d, epsname);
    initialize_mu_array(d, muname);
    float * temp = (float *)malloc(sizeof(float) * size);
    /* Read the csv file and get the sigma array */
    parse_csv(sigma_x_name, temp, size);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMAINDEX_X], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));

    // Similarly parse every csv file.
    parse_csv(sigma_y_name, temp, size);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMAINDEX_Y], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));

    parse_csv(sigma_star_x_name, temp, size);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMA_STAR_INDEX_X], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));

    parse_csv(sigma_star_y_name, temp, size);
    checkCudaErrors(cudaMemcpy2D(d->constants[SIGMA_STAR_INDEX_Y], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));

    dim3 blocks((d->structure->x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (d->structure->y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    // Initialize all the fields to zero.
    initialize_array<<<blocks, threads>>>(d->fields[TM_PML_HXFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_PML_HYFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_PML_EZFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_PML_EZXFIELD], 0);
    initialize_array<<<blocks, threads>>>(d->fields[TM_PML_EZYFIELD], 0);
}
