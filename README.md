# CUDA-Practices
This repo contains some basic cuda examples
* vector_add_1D.cu: Simply adding elements from 2 vectors and check correctness.
* vector_mult_1D.cu: Simply multiplying elements from 2 vectors and check correctness.
* vector_mult_2D.cu: Simply doing matrix multiplication, see below benchmark results.
### vector_mult_2D.cu -- Matrix multiplication benchmark
#### Hardware information
OS: Ubuntu 18.04 LTS
CPU: Intel i7-9700K
GPU: NVIDIA GeForce RTX 2080Ti
RAM: HyperX fury 3766MHz
#### Result
**matrix_A**: 1024 * 1024, **matrix_B**: 1024 * 1024
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 2.903456 ms.
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on CPU: 3574.231689 ms.
Overall speedup = 1231.026611
**matrix_A**: 8192 * 8192, **matrix_B**: 8192 * 8192
Time elapsed on matrix multiplication of 8192x8192 . 8192x8192 on GPU:  ms.
Time elapsed on matrix multiplication of 8192x8192 . 8192x8192 on CPU:  ms.
Overall speedup = 
