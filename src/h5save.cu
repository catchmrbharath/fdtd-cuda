#include "datablock.h"
#include <stdlib.h>
#include "h5save.h"
#define FILE "file.h5"

int create_file(char * name){
    hid_t file_id;
    file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    return 0;
}

int create_new_dataset(Datablock *d){
    hid_t file_id, dataset_id, dataspace_id, status, dcpl, datatype;
    file_id = H5Fopen(d->structure->name, H5F_ACC_RDWR, H5P_DEFAULT);
    hsize_t dims[2];
    dims[0] = d->structure->x_index_dim;
    dims[1] = d->structure->y_index_dim;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
     status = H5Tset_order(datatype, H5T_ORDER_LE);
    char buffer[50];
    sprintf(buffer, "/dset%ld", d->structure->present_ticks);

    dataset_id = H5Dcreate(file_id, buffer, datatype, dataspace_id, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        d->field);
    status = H5Dclose(dataset_id);
    status = H5Tclose(datatype);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    return 1;
}


int test_hdf5(){
    char filename[] = "temp";
    int i, j;
    int ticks = 1;
    int xdim = 1024;
    int ydim = 1024;
    float * data = (float *) malloc(sizeof(float) * xdim * ydim);
    for(i = 0; i < xdim; i++)
        for(j=0; j< ydim; j++){
            data[j * xdim + i] = i * j * 0.7;
        }

    Datablock *d = (Datablock *)malloc(sizeof(Datablock) );
    Structure *s = (Structure *) malloc(sizeof(Structure));
    s->name = "temp";
    s->x_index_dim = 1024;
    s->y_index_dim = 1024;
    d->field = data;
    s->present_ticks = 1;
    d->structure = s;
    create_file(s->name);
    create_new_dataset(d);
    return 1;
}

int main(){
    test_hdf5();
}
