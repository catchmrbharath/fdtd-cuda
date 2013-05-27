#ifndef _TE_MODE_
#define _TE_MODE_

#include "datablock.h"
#include <thrust/fill.h>
#include<algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include<string>
using namespace std;


extern "C" void anim_gpu_tm(Datablock *d, int ticks);
extern "C" void clear_memory_TM_simulation(Datablock *d);
extern "C" size_t allocateTMMemory(Datablock *d, Structure structure);
#include "hostsources.h"
extern "C" void initialize_TM_arrays(Datablock *data, Structure structure);
extern "C" void tm_clear_memory_constants(Datablock *d);
extern "C" void copy_sources_device_to_host(HostSources *host_sources,
                                            DeviceSources *device_sources);
#endif
