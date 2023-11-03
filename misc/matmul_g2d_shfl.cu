#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using uint = __uint32_t;

template <typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

template <typename scalar_t>
using accessor_1d = torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>;


template <typename scalar_t>
__forceinline__ __device__
void gpuAtomicAdd(scalar_t *acc_ptr, scalar_t part_val) {
    #if __CUDA_ARCH__ >= 700
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            atomicAdd(
                reinterpret_cast<half*>(acc_ptr),
                static_cast<half>(part_val)
            );
        }
        else {
            atomicAdd(acc_ptr, part_val);
        }
    #else
        if constexpr (std::is_same_v<scalar_t, float>) {
            atomicAdd(acc_ptr, part_val);
        }
        else {
            assert(false && "Not supported CUDA device.");
        }
    #endif
}


template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t value) {
    constexpr int warp_size = 32;
    constexpr int half_warp = warp_size / 2;
    constexpr int mask = warp_size - 1;

    if constexpr (std::is_same_v<scalar_t, c10::Half>) {
        half br_value = value;

        #pragma unroll
        for (int offset = half_warp; offset > 0; offset /= 2) {
            br_value = __hadd(br_value, __shfl_down_sync(mask, br_value, offset));
        }
        return static_cast<scalar_t>(br_value);
    } else {
        #pragma unroll
        for (int offset = half_warp; offset > 0; offset /= 2) {
            value += __shfl_down_sync(mask, value, offset);
        }
        return value;
    }
}


template <bool a_transposed = false, bool b_transposed = false, typename scalar_t>
__device__ __forceinline__ bool product_matrix_guard(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    const uint a_row_or_col,
    const uint b_col_or_row) {

    if constexpr (not a_transposed) {
        if (a_row_or_col >= matrix_a.size(0)) {
            return false;
        }
    } else {
        if (a_row_or_col >= matrix_a.size(1)) {
            return false;
        }
    }

    if constexpr (not b_transposed) {
        if (b_col_or_row >= matrix_b.size(1)) {
            return false;
        }
    } else {
        if (b_col_or_row >= matrix_b.size(0)) {
            return false;
        }
    }

    return true;
}


template <bool transposed = false, typename scalar_t>
__device__ __forceinline__ scalar_t ld_matrix_elem(
    const accessor_2d<scalar_t> matrix, uint row, uint col) {

    if constexpr (transposed) {
        return matrix[col][row];
    } else {
        return matrix[row][col];
    }
}


template<uint block_size, typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t value) {
    constexpr uint warp_num = block_size / 32;
    __shared__ scalar_t partial_sums[warp_num];

    uint wid = threadIdx.x / warpSize;
    uint lane = threadIdx.x % warpSize;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        partial_sums[wid] = value;
    }

    __syncthreads();

    if (wid == 0) {
        value = (lane < warp_num) ? partial_sums[lane] : static_cast<scalar_t>(0.0f);
        value = warp_reduce_sum(value);
    }

    return value;
}


template <
    uint block_size,
    bool transpose_a,
    bool transpose_b,
    typename scalar_t
>
__global__ void matmul_kernel_shfl(
    accessor_2d<scalar_t> matrix_a,
    accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto n = blockIdx.y;

    bool guard = product_matrix_guard<transpose_a, transpose_b>(
        matrix_a, matrix_b, t, j, i
    );

    scalar_t partial_sum = block_reduce_sum<block_size>(
        (not guard) ? static_cast<scalar_t>(0.0f):
            (ld_matrix_elem<transpose_a>(matrix_a, t, i) 
                * ld_matrix_elem<transpose_b>(matrix_b, i, j))
    );

    if (guard and threadIdx.x == 0) {
        gpuAtomicAdd(&matrix_c[m][n], partial_sum);
    }
}


__forceinline__ uint div_and_ceil(float x, float y) {
    return ceil(x / y);
}


template <bool transpose_a, bool transpose_b>
torch::Tensor matmul(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b) {

    CHECK_ARG(matrix_a);
    CHECK_ARG(matrix_b);

    auto m = transpose_a ? matrix_a.size(1) : matrix_a.size(0);
    auto n = transpose_b ? matrix_b.size(0) : matrix_b.size(1);
    auto k = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    auto matrix_c = torch::zeros(
        {m, n},
        matrix_a.options()
    );

    constexpr dim3 block_dim = {32 * 4, 1};

    const dim3 grid_dim = {
        div_and_ceil(k, block_dim.x),
        div_and_ceil(n, block_dim.y),
    };

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        matrix_a.scalar_type(),
        "linear_forward",
        ([&] {
            matmul_kernel_shfl<block_dim.x, transpose_a, transpose_b><<<grid_dim, block_dim>>>(
                matrix_a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                matrix_b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                matrix_c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
            );
        })
    );

    return matrix_c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul<false, false>, "Performs C = AB with CUDA");
    m.def("matmul_ta", &matmul<true, false>, "Performs C = A^TB with CUDA");
    m.def("matmul_tb", &matmul<false, true>, "Performs C = AB^T with CUDA");
    m.def("matmul_tatb", &matmul<true, true>, "Performs C = A^TB^T with CUDA");
}