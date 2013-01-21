#include "hdf5.h"
#include <stdlib.h>
#define FILE "file.h5"

int create_file(char * name){
    hid_t file_id;
    file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    return 0;
}

int create_new_dataset(char * name, int ticks, float * fields, int xdim, int ydim){
    hid_t file_id, dataset_id, dataspace_id, status, dcpl, datatype;
    file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT);
    hsize_t dims[2];
    dims[0] = xdim;
    dims[1] = ydim;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    status = H5Tset_order(datatype, H5T_ORDER_LE);
    char buffer[50];
    sprintf(buffer, "/dset%d", ticks);

    dataset_id = H5Dcreate(file_id, buffer, datatype, dataspace_id, H5P_DEFAULT);
    printf("%d\n", dataset_id);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        fields);
    status = H5Dclose(dataset_id);
    status = H5Tclose(datatype);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
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

    create_file(filename);
    create_new_dataset(filename, ticks, data, xdim, ydim);
}
int main(){
    test_hdf5();
}
