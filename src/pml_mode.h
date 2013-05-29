#ifndef _PML_MODE_
#define _PML_MODE_
#include "datablock.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include "constants.h"
#include <thrust/fill.h>
#include<algorithm>
#include<fstream>
#include "common_functions.h"
using namespace std;

extern "C" void anim_gpu_pml_tm(Datablock *d, int ticks);
extern "C" void clear_memory_TM_PML_simulation(Datablock *d);
extern "C" void tm_pml_clear_memory_constants(Datablock *d);
extern "C" size_t tm_pml_allocate_memory(Datablock *d, Structure structure);
extern "C" void tm_pml_initialize_arrays(Datablock *data, Structure structure, ifstream& fs);
#endif
