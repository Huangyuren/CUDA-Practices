# CUDA-Practices
This repo contains some basic cuda examples described below:
* vector_add_1D.cu: Simply adding elements from 2 vectors and check correctness.
* vector_mult_1D.cu: Simply multiplying elements from 2 vectors and check correctness.
* vector_mult_2D.cu: Simply doing matrix multiplication, see below benchmark results.
## Hardware information
* OS: Ubuntu 18.04 LTS
* CPU: Intel i7-9700K
* GPU: NVIDIA GeForce RTX 2080Ti
* RAM: HyperX fury DDR4 3766MHz
## vector_mult_2D.cu -- Matrix multiplication benchmark
1. **matrix_A**: 1024 * 1024, **matrix_B**: 1024 * 1024
    Time on GPU (ms): 903456
    Time on CPU (ms): 3574.231689
    Overall speedup = 1231.02661
2. **matrix_A**: 8192 * 8192, **matrix_B**: 8192 * 8192
    Time on GPU (ms):
    Time on CPU (ms):
    Overall speedup =
```
nvcc **.cu -o a.out
./a.out $row_a $col_a $col_b
```
