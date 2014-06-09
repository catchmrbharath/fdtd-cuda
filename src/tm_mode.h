/*! @file tm_mode.h
 * @author Bharath M R
 *
 * @brief Header file for TM mode simulation
 */
#ifndef _TM_MODE_
#define _TM_MODE_

#include "datablock.h"
#include <thrust/fill.h>
#include<algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include "common_functions.h"
#include<string>
#include "hostsources.h"
using namespace std;


extern "C" void anim_gpu_tm(Datablock *d, int ticks);
extern "C" void clear_memory_TM_simulation(Datablock *d);
extern "C" size_t allocateTMMemory(Datablock *d, Structure structure);
extern "C" void initialize_TM_arrays(Datablock *data, Structure structure, ifstream& fs);
extern "C" void tm_clear_memory_constants(Datablock *d);
extern "C" void copy_sources_device_to_host(HostSources *host_sources,
                                            DeviceSources *device_sources);
extern "C" void initialize_eps_array(Datablock *d, string epsname);
extern "C" void initialize_mu_array(Datablock *d, string muname);
extern "C" void parse_csv(string file_name, float * mem, long long size);                                            
#endif
