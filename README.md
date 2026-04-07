# CUDA C++ Course — Beginner Level

A hands-on CUDA C++ course delivered as Jupyter notebooks. Each notebook builds on the
previous one, taking you from zero to writing real parallel GPU programs.

## Run on Google Colab (Recommended)

No local GPU needed — run everything for free in the browser. Click a badge to open:

| # | Notebook | Open in Colab |
|---|----------|:-------------:|
| 1 | Introduction to CUDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/01_Introduction_to_CUDA.ipynb) |
| 2 | Kernels and Thread Hierarchy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/02_Kernels_and_Thread_Hierarchy.ipynb) |
| 3 | Memory Management | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/03_Memory_Management.ipynb) |
| 4 | Vector Addition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/04_Vector_Addition.ipynb) |
| 5 | Error Handling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/05_Error_Handling.ipynb) |
| 6 | 2D Grids and Matrix Ops | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/06_2D_Grids_and_Matrix_Ops.ipynb) |
| 7 | Performance Measurement | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/cuda-cpp-course/blob/main/07_Performance_Measurement.ipynb) |

### Colab Setup (2 steps)

1. **Enable GPU runtime:** `Runtime` > `Change runtime type` > select **T4 GPU** > `Save`
2. **Run the first cell** in any notebook — it installs `nvcc4jupyter` and loads the CUDA extension

That's it. All `%%cuda` cells will compile and run on the free T4 GPU.

## Topics Covered

| # | Notebook | Topics |
|---|----------|--------|
| 1 | Introduction to CUDA | GPU vs CPU, CUDA model, `__global__`, `<<<blocks, threads>>>`, first kernel |
| 2 | Kernels and Thread Hierarchy | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, grid-stride loops, warps, `__syncthreads()` |
| 3 | Memory Management | `cudaMalloc`, `cudaMemcpy`, `cudaFree`, Unified Memory, pinned memory |
| 4 | Vector Addition | Complete end-to-end program, CPU vs GPU comparison, bandwidth measurement |
| 5 | Error Handling | `CUDA_CHECK` macro, `cudaGetLastError`, debugging strategies, `compute-sanitizer` |
| 6 | 2D Grids and Matrix Ops | `dim3`, row-major layout, 2D indexing, matrix add/transpose/scale |
| 7 | Performance Measurement | CUDA events, bandwidth, arithmetic intensity, roofline model, occupancy |

## Prerequisites

- Basic C/C++ knowledge (pointers, functions, arrays)
- A Google account (for Colab) — **no local GPU required**

## Running Locally (Alternative)

If you prefer to run on your own machine:

```bash
# Requirements
# - NVIDIA GPU with CUDA toolkit installed
# - Python 3.8+ with Jupyter

pip install nvcc4jupyter jupyter
jupyter notebook
```

Each notebook uses the `nvcc4jupyter` extension. The first code cell loads it:

```python
!pip install nvcc4jupyter -q
%load_ext nvcc4jupyter
```

Then CUDA C++ code is written in cells with the `%%cuda` magic:

```python
%%cuda
#include <cstdio>
__global__ void hello() { printf("Hello from GPU!\n"); }
int main() { hello<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
```
