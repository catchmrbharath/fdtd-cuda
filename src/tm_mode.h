/*! @file tm_mode.h
 * @author Bharath M R
 *
 * @brief Header file for TM mode simulation
 */
#ifndef _TM_MODE_
#define _TM_MODE_

#include "datablock.h"
#include <thrust/fill.h>
#include <algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include "common_functions.h"
#include <string>
#include "hostsources.h"
using namespace std;


extern "C" void anim_gpu_tm(Datablock *d, int ticks);
extern "C" void clear_memory_TM_simulation(Datablock *d);
extern "C" size_t allocateTMMemory(Datablock *d, Structure structure);
extern "C" void initialize_TM_arrays(Datablock *data, Structure structure, ifstream& fs);
extern "C" void tm_clear_memory_constants(Datablock *d);
extern "C" void copy_sources_device_to_host(HostSources *host_sources,
                                            DeviceSources *device_sources);
#endif
