#include <iostream>
#include <fstream>
#include "common_functions.h"
using namespace std;

void initialize_eps_array(Datablock *d, string epsname)
{
    float * temp = (float *) malloc(sizeof(float) * d->structure->grid_size());
    parse_csv(epsname, temp, d->structure->grid_size());
    checkCudaErrors(cudaMemcpy2D(d->constants[EPSINDEX], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));
    free(temp);
}

void initialize_mu_array(Datablock *d, string muname)
{
    float * temp = (float *) malloc(sizeof(float) * d->structure->grid_size());
    parse_csv(muname, temp, d->structure->grid_size());
    checkCudaErrors(cudaMemcpy2D(d->constants[MUINDEX], d->structure->pitch,
                temp, sizeof(float) * d->structure->x_index_dim,
                sizeof(float) * d->structure->x_index_dim,
                d->structure->y_index_dim,
                cudaMemcpyHostToDevice));
    free(temp);
}

void parse_csv(string file_name, float * mem, long long size)
{
    ifstream in(file_name.c_str());
    for(int i=0; i < size; i++)
        in >> mem[i];
}        
