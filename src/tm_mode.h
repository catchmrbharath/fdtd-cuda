#ifndef _TE_MODE_
#define _TE_MODE_

#include "datablock.h"
#include <thrust/fill.h>
#include<algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"

extern "C" void anim_gpu_tm(Datablock *d, int ticks);
extern "C" void clear_memory_TM_simulation(Datablock *d);
extern "C" void allocateTMMemory(Datablock *d, Structure structure);
extern "C" void initialize_TM_arrays(Datablock *data, Structure structure);

#endif
