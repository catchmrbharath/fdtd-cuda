#ifndef __DEVICE_SOURCES__
#define __DEVICE_SOURCES__
struct DeviceSources{
    int * x_source_position;
    int * y_source_position;
    int * source_type;
    float * mean;
    float * variance;
    int size;
};
#endif
