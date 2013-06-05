/*! @file fdtd.cu

  @brief This is the entry point of the file.
  */
#include "fdtd.h"
#include "cuda.h"
#include "cpu_anim.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "h5save.h"
#include<stdio.h>
#include<pthread.h>
#include "datablock.h"
#include "kernels.cuh"
#include "constants.h"
#include <thrust/fill.h>
#include<algorithm>
#include "tm_mode.h"
#include "pml_mode.h"
#include "drude_mode.h"
#include<fstream>
#include<assert.h>
#include<string>
#include "common_functions.h"
using namespace std;
pthread_mutex_t mutexcopy;
/** @brief Calls the gpu kernels in order.
  * Different types of simulation.
  */
void anim_gpu(Datablock *d, int ticks){
    if(d->simulationType == TM_SIMULATION)
        anim_gpu_tm(d, ticks);
    else if(d->simulationType == TM_PML_SIMULATION)
        anim_gpu_pml_tm(d, ticks);
    else if(d->simulationType == DRUDE_SIMULATION)
        anim_gpu_drude(d, ticks);
}

/*! @brief Clears memory when the simulation is done. */
void anim_exit(Datablock *d){
    if(d->simulationType == TM_SIMULATION)
        clear_memory_TM_simulation(d);
    else if(d->simulationType == TM_PML_SIMULATION)
        clear_memory_TM_PML_simulation(d);
    else if(d->simulationType == DRUDE_SIMULATION)
        clear_memory_drude_simulation(d);

}

/*! @brief Allocates memory for the simulation depending on the type
   of the simulation.
*/
size_t allocate_memory(Datablock *data, Structure structure){
    if(data->simulationType == TM_SIMULATION)
        return allocateTMMemory(data, structure);
    else if(data->simulationType == TM_PML_SIMULATION)
        return tm_pml_allocate_memory(data, structure);
    else if(data->simulationType == DRUDE_SIMULATION)
        return allocate_drude_memory(data, structure);
    return 0;
}

/*! @brief Initializes the memory for simulation.*/
void initializeArrays(Datablock *data, Structure structure, ifstream &fs){
    if(data->simulationType == TM_SIMULATION)
        initialize_TM_arrays(data, structure, fs);
    else if(data->simulationType == TM_PML_SIMULATION)
        tm_pml_initialize_arrays(data, structure, fs);
    else if(data->simulationType == DRUDE_SIMULATION)
        initialize_drude_arrays(data, structure);
}

/*! @brief Clears all the constants initially declared.
  
  This method is called once all the coefficients are
  calculated.
*/
void clear_memory_constants(Datablock *data){
    if(data->simulationType == TM_SIMULATION)
        tm_clear_memory_constants(data);
    else if(data->simulationType == TM_PML_SIMULATION)
        tm_pml_clear_memory_constants(data);
    else if(data->simulationType == DRUDE_SIMULATION)
        drude_clear_memory_constants(data);

}

/*!
  @brief Calculates the coefficients for each simulation.
*/
void calculate_coefficients(Datablock *data, Structure structure){
    dim3 blocks((structure.x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (structure.y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);
    if(data->simulationType == TM_SIMULATION)
        tm_getcoeff<<<blocks, threads>>>(data->constants[MUINDEX],
                                         data->constants[EPSINDEX],
                                         data->constants[SIGMAINDEX],
                                         data->constants[SIGMA_STAR_INDEX],
                                         data->coefs[0],
                                         data->coefs[1],
                                         data->coefs[2],
                                         data->coefs[3]
                                         );

    else if(data->simulationType == TM_PML_SIMULATION)
        pml_tm_get_coefs<<<blocks, threads>>>(data->constants[MUINDEX],
                                              data->constants[EPSINDEX],
                                              data->constants[SIGMAINDEX_X],
                                              data->constants[SIGMAINDEX_Y],
                                              data->constants[SIGMA_STAR_INDEX_X],
                                              data->constants[SIGMA_STAR_INDEX_Y],
                                              data->coefs[0],
                                              data->coefs[1],
                                              data->coefs[2],
                                              data->coefs[3],
                                              data->coefs[4],
                                              data->coefs[5],
                                              data->coefs[6],
                                              data->coefs[7]);

    else if(data->simulationType == DRUDE_SIMULATION)
        drude_get_coefs<<<blocks, threads>>>(data->constants[MUINDEX],
                                         data->constants[EPSINDEX],
                                         data->constants[SIGMAINDEX],
                                         data->constants[SIGMA_STAR_INDEX],
                                         data->constants[GAMMA_INDEX],
                                         data->constants[OMEGAP_INDEX],
                                         data->coefs[0],
                                         data->coefs[1],
                                         data->coefs[2],
                                         data->coefs[3],
                                         data->coefs[4],
                                         data->coefs[5],
                                         data->coefs[6]
                                         );

}

/*! @brief entry point */
int main(int argc, char **argv){
    assert(argc == 2);
    ifstream fs;
    fs.open(argv[1]);
    assert(fs.is_open());
    string simulation_name;
    fs>>simulation_name;
    int simulation_type;
    fs>>simulation_type;
    Datablock data(simulation_type);
    data.simulation_name = simulation_name;
    float dx;
    fs>>dx;
    float courant = 0.5;
    float dt =  courant * dx / LIGHTSPEED;
    printf("dt = %f", dt);

    int xdim, ydim;
    fs>>xdim>>ydim;
    Structure structure(xdim, ydim, dx, dt);


    CPUAnimBitmap bitmap(structure.x_index_dim, structure.x_index_dim,
                            &data); /* bitmap structure */
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    checkCudaErrors(cudaEventCreate(&data.start, 1) );
    checkCudaErrors(cudaEventCreate(&data.stop, 1) );

    size_t pitch;
    pitch = allocate_memory(&data, structure);
    structure.pitch = pitch;
    copy_symbols(&structure);
    printf("pitch = %d", pitch);
    data.structure = &structure;
    initializeArrays(&data, structure, fs);

//  get the coefficients
    calculate_coefficients(&data, structure);
clear_memory_constants(&data);

// set the sources
    HostSources host_sources;
    DeviceSources device_sources;
    long long x, y, source_type;
    float mean, variance;
    while(!fs.eof()){
        fs>>x>>y>>source_type>>mean>>variance;
        cout<<mean<<endl;
        host_sources.add_source(x, y, source_type, mean, variance);
    }

    data.sources = &device_sources;
    copy_sources_device_to_host(&host_sources, &device_sources);
    pthread_mutex_init(&mutexcopy, NULL);

    for(long i=0; i < 3; i++){
        anim_gpu(&data, 0);
    }
}
