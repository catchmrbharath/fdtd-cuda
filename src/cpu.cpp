#include "cpu.h"
#include "time.h"
#include "cpu_kernel.h"
#include<iostream>
using namespace std;

void allocate_cpu_memory(Datablock *d){
    long size = d->structure->size();
    d->output_bitmap = (unsigned char *)malloc(size);
    for(int i=0;i < 3;i++)
        d->fields[i] = (float *)malloc(size);

    for(int i=0; i < 4;i++)
        d->constants[i] = (float *) malloc(size);

    for(int i=0; i < 4; i++){
        d->coefs[i] = (float *) malloc(size);
    }
}

void initialize_cpu_memory(Datablock *d){
    long long size = d->structure->grid_size();
    std::fill_n(d->constants[MUINDEX], size, MU);
    std::fill_n(d->constants[EPSINDEX], size, EPSILON);
    std::fill_n(d->constants[SIGMAINDEX], size, 0);
    std::fill_n(d->constants[SIGMA_STAR_INDEX],size,  0);
    for(int i=0;i < 3;i++){
        std::fill_n(d->fields[i], size, 0);
    }
}

void anim_cpu(Datablock *d, int ticks){
    CPUAnimBitmap *bitmap = d->bitmap;
    float * Ez = d->fields[TM_EZFIELD];
    static long time_ticks = 0;
    clock_t start = clock();
    clock_t end;
    for(int i=0; i < 2; i++){
        time_ticks += 1;
        Ez[512 + 512 * 1024] = 1;
// ADD sources here;
        update_hx_cpu(d);
        update_hy_cpu(d);
        update_ez_cpu(d);
    }
    float_to_color(d);
    end = clock();
    double cputime = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout<<cputime<<endl;
}

void free_constant_cpu(Datablock *d){
    for(int i=0;i < 4;i++){
        free(d->constants[i]);
    }
}

void anim_exit_cpu(Datablock *d){
    for(int i=0; i < 3; i++){
        free(d->fields[i]);
    }
    for(int j = 0; j < 4; j++){
        free(d->coefs[j]);
    }
}
