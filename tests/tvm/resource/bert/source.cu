
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_expand_dims_expand_dims_cast_subtract_multiply_kernel(float* __restrict__ T_multiply, int64_t* __restrict__ p0) {
  T_multiply[((int)threadIdx.x)] = ((1.000000e+00f - ((float)p0[((int)threadIdx.x)])) * -1.000000e+04f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_broadcast_to_reshape_kernel(float* __restrict__ T_reshape, float* __restrict__ p0) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = p0[(((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3) * 768) + (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_batch_matmul_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ p0, float* __restrict__ p1) {
  float T_batch_matmul_NN_local[64];
  __shared__ float p0_shared[512];
  __shared__ float p1_shared[512];
  float p0_shared_local[8];
  float p1_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NN_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 96; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      p0_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = p0[(((((((int)blockIdx.y) * 49152) + (((int)threadIdx.y) * 6144)) + (ax1_inner * 768)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      p1_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ax2_inner)] = p1[(((((k_outer * 6144) + (((int)threadIdx.y) * 768)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + ax2_inner)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        p0_shared_local[ax1] = p0_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        p1_shared_local[ax2] = p1_shared[(((k_inner * 64) + (((int)threadIdx.x) * 8)) + ax2)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NN_local[((i_c * 8) + j_c)] = (T_batch_matmul_NN_local[((i_c * 8) + j_c)] + (p0_shared_local[i_c] * p1_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NN[((((((((int)blockIdx.y) * 49152) + (((int)threadIdx.y) * 6144)) + (i_inner_inner * 768)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NN_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_mean_kernel_1(float* __restrict__ T_divide, float* __restrict__ p0_red) {
  T_divide[((int)threadIdx.x)] = (p0_red[((int)threadIdx.x)] * 1.302083e-03f);
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_batch_matmul_3_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ p0, float* __restrict__ p1) {
  float T_batch_matmul_NN_local[64];
  __shared__ float p0_shared[512];
  __shared__ float p1_shared[512];
  float p0_shared_local[8];
  float p1_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NN_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 96; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      p0_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = p0[(((((((int)blockIdx.y) * 49152) + (((int)threadIdx.y) * 6144)) + (ax1_inner * 768)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      p1_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ax2_inner)] = p1[(((((k_outer * 24576) + (((int)threadIdx.y) * 3072)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + ax2_inner)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        p0_shared_local[ax1] = p0_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        p1_shared_local[ax2] = p1_shared[(((k_inner * 64) + (((int)threadIdx.x) * 8)) + ax2)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NN_local[((i_c * 8) + j_c)] = (T_batch_matmul_NN_local[((i_c * 8) + j_c)] + (p0_shared_local[i_c] * p1_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NN[((((((((int)blockIdx.y) * 196608) + (((int)threadIdx.y) * 24576)) + (i_inner_inner * 3072)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NN_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_transpose_reshape_broadcast_to_reshape_kernel(float* __restrict__ T_reshape, float* __restrict__ p0) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = p0[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 6)) % 12) * 8192) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3) * 64)) + (((int)threadIdx.x) & 63))];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_tanh_kernel(float* __restrict__ T_tanh, float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ p2) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 12; ++k_outer) {
    T_matmul_NT_rf[0] = (T_matmul_NT_rf[0] + (p0[((k_outer * 64) + ((int)threadIdx.x))] * p1[(((((int)blockIdx.x) * 768) + (k_outer * 64)) + ((int)threadIdx.x))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[((int)threadIdx.x)] = T_matmul_NT_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 32)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 16)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    float w_8_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 8)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    float w_4_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 4)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    float w_2_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 2)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    float w_1_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 1)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_matmul_NT[0] = ((volatile float*)red_buf0)[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_tanh[((int)blockIdx.x)] = tanhf((T_matmul_NT[0] + p2[((int)blockIdx.x)]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_add_reshape_transpose_broadcast_to_reshape_kernel(float* __restrict__ T_reshape, float* __restrict__ p0, float* __restrict__ p1) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (p0[(((((((int)blockIdx.x) & 7) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) & 63))] + p1[(((((int)blockIdx.x) >> 3) * 64) + (((int)threadIdx.x) & 63))]);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_variance_kernel(float* __restrict__ T_multiply_red, float* __restrict__ p0, float* __restrict__ p1) {
  float T_multiply_red_rf[1];
  float red_buf0[1];
  T_multiply_red_rf[0] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 24; ++k2_outer) {
    T_multiply_red_rf[0] = (T_multiply_red_rf[0] + ((p0[((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 768)) + (k2_outer * 32)) + ((int)threadIdx.x))] - p1[((((int)blockIdx.x) * 32) + ((int)threadIdx.y))]) * (p0[((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 768)) + (k2_outer * 32)) + ((int)threadIdx.x))] - p1[((((int)blockIdx.x) * 32) + ((int)threadIdx.y))])));
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = T_multiply_red_rf[0];
  mask[0] = (__activemask() & ((uint)0 << ((uint)32 * ((uint)((int)threadIdx.y)))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  if (((int)threadIdx.x) == 0) {
    T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.y))] = red_buf0[0];
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_divide_add_kernel(float* __restrict__ T_add, float* __restrict__ p0, float* __restrict__ p1) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * 1.250000e-01f) + p1[(((int)threadIdx.x) & 127)]);
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_batch_matmul_1_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ p0, float* __restrict__ p1) {
  float T_batch_matmul_NN_local[64];
  __shared__ float p0_shared[512];
  __shared__ float p1_shared[512];
  float p0_shared_local[8];
  float p1_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NN_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      p0_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = p0[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (ax1_inner * 64)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      p1_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ax2_inner)] = p1[((((((((int)blockIdx.z) * 8192) + (k_outer * 1024)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + ax2_inner)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        p0_shared_local[ax1] = p0_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        p1_shared_local[ax2] = p1_shared[(((k_inner * 64) + (((int)threadIdx.x) * 8)) + ax2)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NN_local[((i_c * 8) + j_c)] = (T_batch_matmul_NN_local[((i_c * 8) + j_c)] + (p0_shared_local[i_c] * p1_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NN[(((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.y) * 1024)) + (i_inner_inner * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NN_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_add_reshape_transpose_broadcast_to_reshape_1_kernel(float* __restrict__ T_reshape, float* __restrict__ p0, float* __restrict__ p1) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (p0[((((((int)threadIdx.x) & 127) * 768) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7))] + p1[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7))]);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_kernel(float* __restrict__ T_reshape, float* __restrict__ p0) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_softmax_broadcast_to_kernel(float* __restrict__ T_softmax_maxelem, float* __restrict__ p0) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 3) {
    T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 128; ++k) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 3) {
      T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], p0[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 128)) + k)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_softmax_broadcast_to_kernel_4(float* __restrict__ T_broadcast_to, float* __restrict__ T_softmax_exp) {
  T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_softmax_broadcast_to_kernel_2(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 3) {
    T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 128; ++k) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 3) {
      T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_softmax_exp[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 128)) + k)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_variance_kernel_1(float* __restrict__ T_divide, float* __restrict__ T_multiply_red) {
  T_divide[((int)threadIdx.x)] = (T_multiply_red[((int)threadIdx.x)] * 1.302083e-03f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_mean_kernel(float* __restrict__ p0, float* __restrict__ p0_red) {
  float p0_red_rf[1];
  float red_buf0[1];
  p0_red_rf[0] = 0.000000e+00f;
  for (int k2_outer = 0; k2_outer < 24; ++k2_outer) {
    p0_red_rf[0] = (p0_red_rf[0] + p0[((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 768)) + (k2_outer * 32)) + ((int)threadIdx.x))]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = p0_red_rf[0];
  mask[0] = (__activemask() & ((uint)0 << ((uint)32 * ((uint)((int)threadIdx.y)))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  if (((int)threadIdx.x) == 0) {
    p0_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.y))] = red_buf0[0];
  }
}

extern "C" __global__ void __launch_bounds__(768) tvmgen_default_fused_take_kernel(float* __restrict__ T_take, float* __restrict__ p0) {
  T_take[((int)threadIdx.x)] = p0[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_kernel(float* __restrict__ T_add, float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ p2, float* __restrict__ p3, float* __restrict__ p4) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - p1[(((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3)]) * (1.000000e+00f / sqrtf((p2[(((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3)] + 1.000000e-12f)))) * p3[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768)]) + p4[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768)]);
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_batch_matmul_2_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ p0, float* __restrict__ p1) {
  float T_batch_matmul_NN_local[64];
  __shared__ float p0_shared[512];
  __shared__ float p1_shared[512];
  float p0_shared_local[8];
  float p1_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NN_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      p0_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = p0[((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.y) * 1024)) + (ax1_inner * 128)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      p1_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ax2_inner)] = p1[(((((((int)blockIdx.z) * 8192) + (k_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 8)) + ax2_inner)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        p0_shared_local[ax1] = p0_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        p1_shared_local[ax2] = p1_shared[(((k_inner * 64) + (((int)threadIdx.x) * 8)) + ax2)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NN_local[((i_c * 8) + j_c)] = (T_batch_matmul_NN_local[((i_c * 8) + j_c)] + (p0_shared_local[i_c] * p1_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NN[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NN_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_softmax_broadcast_to_kernel_3(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem) {
  T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_softmax_maxelem[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7))]);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_softmax_broadcast_to_kernel_1(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem, float* __restrict__ p0) {
  T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __expf((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7))]));
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_add_multiply_erf_multiply_add_multiply_broadcast_to_reshap_f895a3812fb00bdf__kernel(float* __restrict__ T_reshape, float* __restrict__ p0, float* __restrict__ p1) {
  for (int ax0_ax1_fused_ax2_fused_outer = 0; ax0_ax1_fused_ax2_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer * 2) + (((int)blockIdx.x) >> 7)) < 3) {
      T_reshape[(((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ((p0[(((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + p1[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 3072)]) * (5.000000e-01f + (erff(((p0[(((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + p1[((((ax0_ax1_fused_ax2_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 3072)]) * 7.071068e-01f)) * 5.000000e-01f)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_squeeze_add_add_kernel(float* __restrict__ T_add, float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ p2) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((p0[(((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3) * 768) + (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768))] + p1[(((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768)]) + p2[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_batch_matmul_4_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ p0, float* __restrict__ p1) {
  float T_batch_matmul_NN_local[64];
  __shared__ float p0_shared[512];
  __shared__ float p1_shared[512];
  float p0_shared_local[8];
  float p1_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NN_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 384; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      p0_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = p0[(((((((int)blockIdx.y) * 196608) + (((int)threadIdx.y) * 24576)) + (ax1_inner * 3072)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      p1_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ax2_inner)] = p1[(((((k_outer * 6144) + (((int)threadIdx.y) * 768)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + ax2_inner)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        p0_shared_local[ax1] = p0_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 8; ++ax2) {
        p1_shared_local[ax2] = p1_shared[(((k_inner * 64) + (((int)threadIdx.x) * 8)) + ax2)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NN_local[((i_c * 8) + j_c)] = (T_batch_matmul_NN_local[((i_c * 8) + j_c)] + (p0_shared_local[i_c] * p1_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NN[((((((((int)blockIdx.y) * 49152) + (((int)threadIdx.y) * 6144)) + (i_inner_inner * 768)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NN_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_cast_take_cast_take_add_add_kernel(float* __restrict__ T_add, int64_t* __restrict__ p0, float* __restrict__ p1, int64_t* __restrict__ p2, float* __restrict__ p3, float* __restrict__ p4) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((p1[((min(max(0, ((int)p0[(((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3)])), 30521) * 768) + (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768))] + p3[((min(max(0, ((int)p2[(((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 3)])), 1) * 768) + (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % 768))]) + p4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

