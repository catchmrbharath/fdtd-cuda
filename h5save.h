#ifndef _H5_SAVE_
#define _H5_SAVE_

#include "hdf5.h"

int createfile(char * name);
int create_new_dataset(char * name, int ticks, float * fields, int xdim,
                        int ydim);
#endif
