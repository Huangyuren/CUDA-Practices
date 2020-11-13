# CUDA-Practices
This repo contains some basic cuda examples
* vector_add_1D.cu: Simply adding elements from 2 vectors and check correctness.
* vector_mult_1D.cu: Simply multiplying elements from 2 vectors and check correctness.
* vector_mult_2D.cu: Simply doing matrix multiplication, see below benchmark results.
### vector_mult_2D.cu -- Matrix multiplication benchmark
#### Hardware information
OS: Ubuntu 18.04 LTS <br />
CPU: Intel i7-9700K <br />
GPU: NVIDIA GeForce RTX 2080Ti <br />
RAM: HyperX fury 3766MHz <br />
#### Result
**matrix_A**: 1024 * 1024, **matrix_B**: 1024 * 1024 <br />
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 2.903456 ms. <br />
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on CPU: 3574.231689 ms. <br />
Overall speedup = 1231.026611 <br />
**matrix_A**: 8192 * 8192, **matrix_B**: 8192 * 8192 <br />
Time elapsed on matrix multiplication of 8192x8192 . 8192x8192 on GPU:  ms. <br />
Time elapsed on matrix multiplication of 8192x8192 . 8192x8192 on CPU:  ms. <br />
Overall speedup =  <br />
