#include "datablock.h"
#include <stdlib.h>
#include <string>
#include "h5save.h"
#include "fdtd.h"
#include<pthread.h>
using namespace std;

int create_file(char * name){
    hid_t file_id;
    return 0;
}

void *create_new_dataset(void *data){
    Datablock *d = (Datablock *)data;
    string new_name = d->simulation_name;
    int ticks = d->present_ticks;
    char buffer[200];
    sprintf(buffer, "%s%d%s", new_name.c_str(), ticks, ".h5");
    hid_t file_id, dataset_id, dataspace_id, status, dcpl, datatype;
    pthread_mutex_lock(&mutexcopy);
    file_id = H5Fcreate(buffer, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[2];
    dims[0] = d->structure->x_index_dim;
    dims[1] = d->structure->y_index_dim;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
     status = H5Tset_order(datatype, H5T_ORDER_LE);
    dataset_id = H5Dcreate(file_id, buffer, datatype, dataspace_id, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        d->save_field);
    status = H5Dclose(dataset_id);
    status = H5Tclose(datatype);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    pthread_mutex_unlock(&mutexcopy);
}



/*int main(){*/
    /*test_hdf5();*/
/*}*/
