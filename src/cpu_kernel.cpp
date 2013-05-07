#include<iostream>
#include "datablock.h"
using namespace std;

void update_hx_cpu(Datablock *d){
    float * Hx = d->fields[TM_HXFIELD];
    float *Ez = d->fields[TM_EZFIELD];
    float *coef1 = d->coefs[0];
    float *coef2 = d->coefs[1];
    long long x_index_dim = d->structure->x_index_dim;
    long long y_index_dim = d->structure->y_index_dim;

    for(long long j=0; j < y_index_dim;j++){
        for(long long i=0; i < x_index_dim;i++){
            long long offset = i + j * x_index_dim;
            long long top = offset + x_index_dim;
            if(j < y_index_dim - 1){
                Hx[offset] = coef1[offset] * Hx[offset] -
                    coef2[offset] * (Ez[top] - Ez[offset]);
            }

        }
    }
}

void update_hy_cpu(Datablock *d){
    float * Hy = d->fields[TM_HXFIELD];
    float *Ez = d->fields[TM_EZFIELD];
    float *coef1 = d->coefs[0];
    float *coef2 = d->coefs[1];
    long long x_index_dim = d->structure->x_index_dim;
    long long y_index_dim = d->structure->y_index_dim;
    for(long long j=0; j < y_index_dim;j++){
        for(long long i=0; i < x_index_dim;i++){
            long long offset = i + j * x_index_dim;
            long long right = offset + 1;
            if(j < x_index_dim - 1){
                Hy[offset] = coef1[offset] * Hy[offset] +
                    coef2[offset] * (Ez[right] - Ez[offset]);
            }

        }
    }
}


void update_ez_cpu(Datablock *d){
    float *Hx = d->fields[TM_HXFIELD];
    float *Hy = d->fields[TM_HYFIELD];
    float *Ez = d->fields[TM_EZFIELD];
    float *coef1 = d->coefs[2];
    float *coef2 = d->coefs[3];
    long long x_index_dim = d->structure->x_index_dim;
    long long y_index_dim = d->structure->y_index_dim;
    for(long long j = 0; j < y_index_dim;j++){
        for(long long i = 0 ; i < x_index_dim;i++){
            if(i > 0 && j > 0 && i < x_index_dim - 1 && j < y_index_dim - 1){
                long long offset = i + j * x_index_dim;
                long long left = offset - 1;
                long long bottom = offset - x_index_dim;
                Ez[offset] = coef1[offset] * Ez[offset] +
                    coef2[offset] * (Hy[offset] - Hy[left]) -
                    (Hx[offset] - Hx[bottom]);
            }
        }
    }
}

void cpu_get_coef(Datablock *d){
    long x_index_dim = d->structure->x_index_dim;
    long y_index_dim = d->structure->y_index_dim;
    float *mu = d->constants[MUINDEX];
    float *epsilon = d->constants[EPSINDEX];
    float *sigma = d->constants[SIGMAINDEX];
    float *sigmastar = d->constants[SIGMA_STAR_INDEX];

    float *coef1 = d->coefs[0];
    float *coef2 = d->coefs[1];
    float *coef3 = d->coefs[2];
    float *coef4 = d->coefs[3];

    long long deltat = d->structure->dt;
    long long delta = d->structure->dx;
    for(int j=0; j < y_index_dim;j++){
        for(int i=0;i < x_index_dim;i++){
            long long offset = i + j * x_index_dim;
            float mus = mu[offset];
            float sigmamstar = sigmastar[offset];
            float eps = epsilon[offset];
            float sigmam = sigma[offset];
            coef1[offset] = (2.0 * mus - sigmamstar * deltat) /
                (2.0 * mus + sigmamstar * deltat);
            coef2[offset] = (2 * deltat) / ((2 * mus + sigmamstar * deltat) * delta);

            coef3[offset] = (2.0 * eps - sigmam * deltat) /
                (2.0 * eps + sigmam * deltat);
            coef4[offset] = (2.0 * deltat) /
                ((2 * eps + sigmam * deltat) * delta);
        }
    }
}

unsigned char value( float n1, float n2, int hue ) {
    if (hue > 360)      hue -= 360;
    else if (hue < 0)   hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
    return (unsigned char)(255 * n1);
}



void float_to_color(Datablock *d) {
    long x_index_dim = d->structure->x_index_dim;
    long y_index_dim = d->structure->y_index_dim;
    unsigned char *optr = (d->bitmap->get_ptr());
    float * outSrc = d->fields[TM_EZFIELD];

    for(int j = 0; j < y_index_dim; j++){
        for(int i=0; i < x_index_dim; i++){
            long long offset = i + j * x_index_dim;
            float l = outSrc[offset];
            float s = 1;
            int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
            float m1, m2;

            if (l <= 0.5f)
                m2 = l * (1 + s);
            else
                m2 = l + s - l * s;
            m1 = 2 * l - m2;

            optr[offset*4 + 0] = value( m1, m2, h+120 );
            optr[offset*4 + 1] = value( m1, m2, h );
            optr[offset*4 + 2] = value( m1, m2, h -120 );
            optr[offset*4 + 3] = 255;
        }
    }
}
