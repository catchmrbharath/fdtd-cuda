#ifndef _DRUDE_MODE_
#define _DRUDE_MODE_

#include "datablock.h"
#include <thrust/fill.h>
#include<algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include "hostsources.h"

extern "C" void anim_gpu_drude(Datablock *d, int ticks);
extern "C" void clear_memory_drude_simulation(Datablock *d);
extern "C" void allocate_drude_memory(Datablock *d, Structure structure);
extern "C" void initialize_drude_arrays(Datablock *data, Structure structure);
extern "C" void drude_clear_memory_constants(Datablock *d);
#endif
