#include <cuda_runtime.h>

extern "C" {

#include <stdio.h>
#include <math.h>
#include <time.h>


__host__ void h_add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}


__global__ void d_add(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


float compare(float *a, float *b, int n, float eps) {
    float diff = 0;

    for (int i = 0; i < n; i++) {
        diff = fabs(a[i] - b[i]);
        if (diff >= eps) {
            return diff;
        }
    }

    return diff;
}


void fill(float *a, float *b, int n, float r) {
    for (int i = 0; i < n; i++) {
        a[i] = -r + 2 * r * ((float)rand() / RAND_MAX);
        b[i] = -r + 2 * r * ((float)rand() / RAND_MAX);
    }
}

}


#define VEC_LEN 51200000
#define VEC_LEN_INC 512000
#define VEC_MAX_ABS_VAL 101
#define SEED 27
#define BLOCK_SIZE 128
#define VEC_MEM_SIZE (VEC_LEN * sizeof(float))
#define FILE_NAME "timings.stmp"
#define PRECISION 10e-8
#define ts_to_ms(ts) ((ts.tv_sec * 10e9 + ts.tv_nsec) * 10e-6)


int main() {
    FILE* file = fopen(FILE_NAME, "a");

    float *h_a, *h_b, *h_c, *h_d;
    h_a = (float*)malloc(VEC_MEM_SIZE);
    h_b = (float*)malloc(VEC_MEM_SIZE);
    h_c = (float*)malloc(VEC_MEM_SIZE);
    h_d = (float*)malloc(VEC_MEM_SIZE);

    srand(SEED);
    fill(h_a, h_b, VEC_LEN, VEC_MAX_ABS_VAL);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, VEC_MEM_SIZE);
    cudaMalloc((void**)&d_b, VEC_MEM_SIZE);
    cudaMalloc((void**)&d_c, VEC_MEM_SIZE);

    cudaMemcpy(d_a, h_a, VEC_MEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, VEC_MEM_SIZE, cudaMemcpyHostToDevice);

    float h_time;
    timespec h_start, h_stop;

    float d_time;
    cudaEvent_t d_start, d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);

    int grid_size;

    fprintf(file, "Vector Length, CPU Time, GPU Time\n");

    for (int m = VEC_LEN_INC; m < VEC_LEN; m += VEC_LEN_INC) {
        grid_size = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // or the same "grid_size = ceil((float)m / BLOCK_SIZE)" 

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &h_start);
        h_add(h_a, h_b, h_c, m);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &h_stop);
        h_time = (ts_to_ms(h_stop) - ts_to_ms(h_start)); // time in ms

        cudaEventRecord(d_start);
        d_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, m);
        cudaEventRecord(d_stop);
        cudaEventSynchronize(d_stop);
        cudaEventElapsedTime(&d_time, d_start, d_stop); // time in ms

        cudaMemcpy(h_d, d_c, m * sizeof(float), cudaMemcpyDeviceToHost);

        if (compare(h_c, h_d, m, PRECISION) > PRECISION) {
            printf("Panic!\n");
            break;
        }

        fprintf(file, "%d, %f, %f\n", m, h_time, d_time);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    fclose(file);

    return 0;
}