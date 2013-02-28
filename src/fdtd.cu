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


void anim_gpu(Datablock *d, int ticks){
    if(d->simulationType == TM_SIMULATION)
        anim_gpu_tm(d, ticks);
}


void anim_exit(Datablock *d){
    if(d->simulationType == TM_SIMULATION)
        clear_memory_TM_simulation(d);

}
void allocate_memory(Datablock *data, Structure structure){
    if(data->simulationType == TM_SIMULATION)
        allocateTMMemory(data, structure);
}

void initializeArrays(Datablock *data, Structure structure){
    if(data->simulationType == TM_SIMULATION)
        initialize_TM_arrays(data, structure);
}


int main(){
    Datablock data(TM_SIMULATION);
    float dt= 0.5;
// FIXME: check the courant factor for the max epsilon.
    float courant = 0.5;
    float dx =  (dt * LIGHTSPEED) / courant;
    Structure structure(1024, 1024, dx, dt);
    copy_symbols(&structure);


    CPUAnimBitmap bitmap(structure.x_index_dim, structure.x_index_dim,
                            &data);

    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    data.structure = &structure;
    checkCudaErrors(cudaEventCreate(&data.start, 1) );
    checkCudaErrors(cudaEventCreate(&data.stop, 1) );

    allocate_memory(&data, structure);
    initializeArrays(&data, structure);


//  get the coefficients

    dim3 blocks((structure.x_index_dim + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
                (structure.y_index_dim + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y);

    tm_getcoeff<<<blocks, threads>>>(data.constants[0],
                                     data.constants[1],
                                     data.constants[2],
                                     data.constants[3],
                                     data.coefs[0],
                                     data.coefs[1],
                                     data.coefs[2],
                                     data.coefs[3]);
clear_memory_constants(&data);


// set the sources
    HostSources host_sources;
    DeviceSources device_sources;
    host_sources.add_source(512, 512, SINUSOID_SOURCE, 0.05, 1);
    host_sources.add_source(256, 512, SINUSOID_SOURCE, 0.1, 1);
    host_sources.add_source(1, 0, SINUSOID_SOURCE, 0.1, 1);

    data.sources = &device_sources;
    copy_sources_device_to_host(&host_sources, &device_sources);

    bitmap.anim_and_exit( (void (*)(void *, int)) anim_gpu,
                            (void (*)(void *)) anim_exit);
}
