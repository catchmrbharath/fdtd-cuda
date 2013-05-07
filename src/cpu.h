#ifndef _CPU_MODE_
#define _CPU_MODE_

#include "datablock.h"
#include<algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "constants.h"


void anim_cpu(Datablock *d, int ticks);
void allocate_cpu_memory(Datablock *d);
void initialize_cpu_memory(Datablock *d);
void cpu_clear_memory_constants(Datablock *d);
void free_constant_cpu(Datablock *d);
void anim_exit_cpu(Datablock *d);


#endif
