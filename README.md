# CUDA-Practices
This repo contains some basic cuda examples described below:
* vector_add_1D.cu: Simply adding elements from 2 vectors and check correctness.
* vector_mult_1D.cu: Simply multiplying elements from 2 vectors and check correctness.
* vector_mult_2D.cu: Simply doing matrix multiplication, see below benchmark results.
## Hardware information
* OS: Ubuntu 18.04 LTS
* CPU: Intel i7-9700K
* GPU: NVIDIA GeForce RTX 2080Ti
* RAM: DDR4 3766MHz 16GB
## vector_mult_2D.cu -- Matrix multiplication benchmark
1. **matrix_A**: 1024 * 1024, **matrix_B**: 1024 * 1024
    * Time on GPU (ms): 2.936768
    * Time on CPU (ms): 1964.278809
    * Overall speedup = 668.857300
2. **matrix_A**: 4096 * 4096, **matrix_B**: 4096 * 4096
    * Time on GPU (ms): 126.989983
    * Time on CPU (ms): 130397.976562
    * Overall speedup = 1026.836670
3. **matrix_A**: 8192 * 8192, **matrix_B**: 8192 * 8192
    * Time on GPU (ms): 720.784180
    * Time on CPU (ms): 6992006.500000
    * Overall speedup = 9700.554688
## Additional results -- Unified memory
> Tips: Unified memory would treat GPU memory and CPU memory as one entity, though that is on software point. NVIDIA did prefetching and on-demand paging to boost performance and fully utilize benefits of paged memory.
> For more information, please refer to [this link](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
* Result: 
    * Three mode
        * normal: programmer explicitly specified cudaMalloc & cudaMemcpy
        * Unified memory: managed memory
        * Unified + prefetch: plus explicitly prefetching pages of data
    * X-axis stands for square matrix rows/cols, which is {1, 2, 4, 8, 16, 32} $\times$ 1024
    * Y-axis stands for runtime normalized to normal case
    * Matrix of 32 $\times$ 1024 is too large for GPU, so I normalized runtime to Unified memory's runtime
* ![](https://i.imgur.com/6sAQezt.png)
```
How to compile:
nvcc **.cu -o a.out
Run:
./a.out $row_a $col_a $col_b
```
