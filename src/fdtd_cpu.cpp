#include "datablock.h"
#include "cpu.h"
#include "cpu_kernel.h"
#include<iostream>
using namespace std;

int main(){
    Datablock data(TM_SIMULATION);
    float dx = 1e-6 / 300.0;
    float courant = 0.5;
    float dt = courant * dx / LIGHTSPEED;
    Structure structure(1024, 1024, dx, dt);
    data.structure = &structure;
    CPUAnimBitmap bitmap(structure.x_index_dim, structure.x_index_dim,
                            &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    allocate_cpu_memory(&data);
    initialize_cpu_memory(&data);
    cpu_get_coef(&data);
    free_constant_cpu(&data);
    for(int i=0; i < 10; i++){
        anim_cpu(&data, 0);
    }
}

