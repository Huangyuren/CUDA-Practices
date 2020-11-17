#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 32
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
            ret += dev_a[row * col_a + i] * dev_b[i * col_b + col];//original b
            //ret += dev_a[row*col_a + i] * dev_b[col*col_b + i];//transposed b, but NOT speeduped
        }
        dev_c[row*col_b + col] = ret;
    }
}
__global__ void matrixTranspose(int* in_mat, int* out_mat, int dim_rows, int dim_cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < dim_rows && col < dim_cols) {
        unsigned int new_pos = col * dim_cols + row;
        out_mat[new_pos] = in_mat[row*dim_cols + col];
    }
}
void matrixTranspose_cpu(int* in_mat, int* out_mat, int dim_rows, int dim_cols) {
    for (int i=0; i<dim_rows; ++i) {
        for (int j=0; j<dim_cols; ++j) {
            unsigned int new_pos = j*dim_cols + i;
            out_mat[new_pos] = in_mat[i*dim_cols+j];
        }
    }
}
void matrixMultiplication_cpu(int* host_a, int* host_b, int* host_c, int row_a, int col_a, int col_b) {
    for (int i=0; i<row_a; ++i) {
        for (int j=0; j<col_b; ++j) {
            int tmp=0;
            for (int k=0; k<col_a; ++k) {
                //tmp += host_a[i*col_a+k] * host_b[k*col_b+j];//original b
                tmp += host_a[i*col_a+k] * host_b[j*col_b+k];//transposed b for speeduped cpu multiplication
            }
            host_c[i*col_b + j] = tmp;
        }
    }
}
void matrixMultiplication_cpu_cache_friendly(int* host_a, int* host_b, int* host_c, int row_a, int col_a, int col_b) {
    for (int i=0; i<row_a; ++i) {
        for(int k=0; k<col_a; ++k) {
            int tmp = host_a[i*col_a + k];
            for(int j=0; j<col_b; ++j) {
                host_c[i*col_b+j] += tmp * host_b[k*col_b+j];
            }
        }
    }
}
bool verifyResult(int* h_c, int* h_c_result, int rows, int cols) {
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            if(h_c[i*cols + j] != h_c_result[i*cols + j]){
                printf("Host: %d, Device: %d\n", h_c[i*cols + j], h_c_result[i*cols + j]);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if(argc != 4){
        fprintf(stderr, "%s", "Usage: ./a.out $row_A $col_A $col_B $thread_count_in_block in 1Dim direction\n");
        exit(-1);
    }
	int row_a = atoi(argv[1]);
	int col_a = atoi(argv[2]);
	int col_b = atoi(argv[3]);

    float gpu_elapsed_time_ms;
    float cpu_elapsed_time_ms;
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start counting execution time of device computation
    cudaEventRecord(start, 0);
	int* h_a, *h_b, *h_c_result;
    int* h_c;
    //int* h_b_trans;
	cudaCheckError_inline(cudaMallocHost((void **) &h_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError_inline(cudaMallocHost((void **) &h_b, sizeof(int)*(col_a*col_b)));
	cudaCheckError_inline(cudaMallocHost((void **) &h_c, sizeof(int)*(row_a*col_b)));
    cudaCheckError_inline(cudaMallocHost((void **) &h_c_result, sizeof(int)*(row_a*col_b)));
	//cudaCheckError_inline(cudaMallocHost((void **) &h_b_trans, sizeof(int)*(col_a*col_b)));
    //Random initialized matrix a on host
    for(int i=0; i<row_a; ++i) {
        for(int j=0; j<col_a; ++j) {
            h_a[i*col_a+j] = rand() % 1024;
        }
    }
    //Random initialized matrix b on host
    for(int i=0; i<col_a; ++i) {
        for(int j=0; j<col_b; ++j) {
            h_b[i*col_b+j] = rand() % 1024;
        }
    }

	int* dev_a, *dev_b, *dev_c;
    //int* dev_b_trans;
	cudaCheckError_inline(cudaMalloc((void **) &dev_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError_inline(cudaMalloc((void **) &dev_b, sizeof(int)*(col_a*col_b)));
	cudaCheckError_inline(cudaMalloc((void **) &dev_c, sizeof(int)*(row_a*col_b)));
	//cudaCheckError_inline(cudaMalloc((void **) &dev_b_trans, sizeof(int)*(col_a*col_b)));

	cudaCheckError_inline(cudaMemcpy(dev_a, h_a, sizeof(int)*(row_a*col_a), cudaMemcpyHostToDevice));
	cudaCheckError_inline(cudaMemcpy(dev_b, h_b, sizeof(int)*(col_a*col_b), cudaMemcpyHostToDevice));

    int grid_row = (row_a + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_col = (col_b + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_col, grid_row);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    /*
    //First we perform matrix transpose
    matrixTranspose<<<dimGrid, dimBlock>>>(dev_b, dev_b_trans, col_a, col_b);
    cudaCheckError_inline(cudaDeviceSynchronize());
    //Then we perform multiplication on transposed matrix
	matrixMultiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b_trans, dev_c, row_a, col_a, col_b);
    */
    matrixMultiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, row_a, col_a, col_b);
    cudaCheckError_inline(cudaDeviceSynchronize());
    cudaCheckError_inline(cudaMemcpy(h_c_result, dev_c, sizeof(int)*row_a*col_b, cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n", row_a, col_a, col_a, col_b, gpu_elapsed_time_ms);

    
    //Start counting execution time of cpu computation
    //matrixTranspose_cpu(h_b, h_b_trans, col_a, col_b);
    cudaEventRecord(start, 0);
    matrixMultiplication_cpu_cache_friendly(h_a, h_b, h_c, row_a, col_a, col_b);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n", row_a, col_a, col_a, col_b, cpu_elapsed_time_ms);
    bool check = verifyResult(h_c, h_c_result, row_a, col_b);
    if(!check) {
        fprintf(stderr, "Error, result not matched.\n");
        exit(4);
    }else{
        printf("Congratulations, results match !!\n");
    }
    float speedups = cpu_elapsed_time_ms / gpu_elapsed_time_ms;
    printf("Overall speedup = %f\n", speedups);
    

    cudaCheckError_inline(cudaFree(dev_a));
    cudaCheckError_inline(cudaFree(dev_b));
    cudaCheckError_inline(cudaFree(dev_c));

    cudaCheckError_inline(cudaFreeHost(h_a));
    cudaCheckError_inline(cudaFreeHost(h_b));
    cudaCheckError_inline(cudaFreeHost(h_c));
    cudaCheckError_inline(cudaFreeHost(h_c_result));

}
