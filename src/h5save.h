#ifndef __H5_SAVE__
#define __H5_SAVE__

#include "hdf5.h"

typedef struct h5data{
    int x_index_dim;
    int y_index_dim;
    char *name;
    long int ticks;
    float* field;
}H5block;
int createfile(char * name);
int create_new_dataset(H5block *d);
#endif
