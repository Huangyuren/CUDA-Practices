#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 64
#define cudaCheckError() {                  \
    cudaError_t e = cudaGetLastError();     \
    if (e != cudaSuccess) {                 \
        printf("CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                 \
    }                                       \
}

inline cudaError_t cudaCheckError_inline(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

__global__ void matrixMultiplication(int* dev_a, int* dev_b, int* dev_c, int row_a, int col_a, int col_b) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    // each (row, col) pair will match on element in resulting matrix -> with shape (row_a, col_b)
    int ret=0;
    if (row < row_a && col < col_b) {
        for(int i=0; i<col_a; ++i) {
            ret += dev_a[row * col_a + i] * dev_b[i * col_b + col_b];
        }
        dev_c[row*col_b + col] = ret;
    }
}

void matrixMultiplication_cpu(int* host_a, int* host_b, int* host_c, int row_a, int col_a, int col_b) {
    for (int i=0; i<row_a; ++i) {
        for (int j=0; j<col_b; ++j) {
            int tmp=0;
            for (int k=0; k<col_a; ++k) {
                tmp += host_a[i*col_a+k] * host_b[k*col_b+j];
            }
            host_c[i*col_b + j] = tmp;
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc != 4){
        fprintf(stderr, "%s", "Usage: ./a.out $row_A $col_A $col_B $thread_count_in_block in 1Dim direction\n");
        exit(-1);
    }
	int row_a = atoi(argv[1]);
	int col_a = atoi(argv[2]);
	int col_b = atoi(argv[3]);
	int* h_a, *h_b, *h_c, *h_c_for_dev;
	cudaCheckError_inline(cudaMallocHost(&h_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError_inline(cudaMallocHost(&h_b, sizeof(int)*(col_a*col_b)));
	cudaCheckError_inline(cudaMallocHost(&h_c, sizeof(int)*(row_a*col_b)));
    cudaCheckError_inline(cudaMallocHost(&h_c_for_dev, sizeof(int)*(row_a*col_b)));
    //Random initialized matrix a on host
    for(int i=0; i<row_a; ++i) {
        for(int j=0; j<col_a; ++j) {
            h_a[i*col_a+j] = rand() % 1024;
        }
    }
    //Random initialized matrix b on host
    for(int i=0; i<col_a; ++i) {
        for(int j=0; j<col_b; ++j) {
            h_a[i*col_b+j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time;
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start to count execution time of device computation
    cudaEventRecord(start, 0);
	int* dev_a, *dev_b, *dev_c;
	cudaCheckError_inline(cudaMalloc((void **) &dev_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError_inline(cudaMalloc((void **) &dev_b, sizeof(int)*(col_a*col_b)));
	cudaCheckError_inline(cudaMalloc((void **) &dev_c, sizeof(int)*(row_a*col_b)));

	cudaCheckError_inline(cudaMemcpy(dev_a, h_a, sizeof(int)*(row_a*col_a), cudaMemcpyHostToDevice));
	cudaCheckError_inline(cudaMemcpy(dev_b, h_b, sizeof(int)*(col_a*col_b), cudaMemcpyHostToDevice));

    int grid_row = (row_a + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_col = (col_b + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_col, grid_row);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	matrixMultiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, row_a, col_a, col_b);
    cudaCheckError();
    cudaCheckError_inline(cudaMemcpy(h_c_for_dev, dev_c, sizeof(int)*row_a*col_b, cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", row_a, col_a, col_a, col_b, gpu_elapsed_time_ms);
    matrixMultiplication_cpu(h_a, h_b, h_c, row_a, col_a, col_b);
    cudaCheckError_inline(cudaFree(dev_a));
    cudaCheckError_inline(cudaFree(dev_b));
    cudaCheckError_inline(cudaFree(dev_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_for_dev);

}
