#include "cuda.h"
#include "book.h"
#include "cpu_anim.h"


#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_VOL 1.0f
#define MIN_VOL 0.00001f
#define SPEED 0.25f

struct DataBlock{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

__global__ void copy_const_kernel(float *iptr, const float *cptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if(cptr[offset] != 0){
        iptr[offset] = cptr[offset];
    }
}

__global__ void blend_kernel(float *outSrc, const float *inSrc){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if(x==0) left++;
    if(x==DIM - 1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;

    if (y==0) top += DIM;
    if (y==DIM - 1) bottom -= DIM;

    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top]
            + inSrc[bottom] + inSrc[right] + inSrc[left]
            - 4 * inSrc[offset]);
}
void anim_gpu(DataBlock *d, int ticks){
    HANDLE_ERROR(cudaEventRecord(d->start, 0) );
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;
    for(int i=0;i<90;i++){
        copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks, threads>>> (d->output_bitmap, d->dev_inSrc);
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(d->stop, 0) );
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime +=elapsedTime;
    d->frames +=1;
    printf("Average time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock *d){
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);
    HANDLE_ERROR(cudaEventDestroy(d->start) );
    HANDLE_ERROR(cudaEventDestroy(d->stop) );
}

int main(){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR(cudaEventCreate(&data.start, 1) );
    HANDLE_ERROR(cudaEventCreate(&data.stop, 1) );

    HANDLE_ERROR(cudaMalloc( (void **) &data.output_bitmap, bitmap.image_size() ));
    HANDLE_ERROR(cudaMalloc( (void **) &data.dev_inSrc, bitmap.image_size() ));
    HANDLE_ERROR(cudaMalloc( (void **) &data.dev_outSrc, bitmap.image_size() ));
    HANDLE_ERROR(cudaMalloc( (void **) &data.dev_constSrc, bitmap.image_size() ));

    float *temp = (float *) malloc(bitmap.image_size() );
    int i = 0;

    for (i= 0; i<DIM * DIM; i++){
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if( (x > 300) && (x < 600) && (y> 310) && (y<610) ){
            temp[i] = MAX_VOL;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
    free(temp);
    bitmap.anim_and_exit( (void (*)(void *, int)) anim_gpu, (void (*)(void *)) anim_exit);
}
