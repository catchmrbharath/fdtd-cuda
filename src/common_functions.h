#ifndef _COMMON_FUNCTIONS_
#define _COMMON_FUNCTIONS_
#include "fdtd.h"
#include "datablock.h"
#include <thrust/fill.h>
#include <algorithm>
#include "helper_cuda.h"
#include "helper_functions.h"
#include "kernels.cuh"
#include <string>
#include "hostsources.h"
using namespace std;

extern "C" void initialize_eps_array(Datablock *d, string epsname);
extern "C" void initialize_mu_array(Datablock *d, string muname);
extern "C" void parse_csv(string file_name, float *mem, long long size);     
#endif