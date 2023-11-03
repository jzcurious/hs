#include <torch/extension.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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


template <
    bool a_transposed = false,
    bool b_transposed = false,
    typename scalar_t
>
__device__ __forceinline__ bool product_matrix_guard(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    const uint a_row_or_col,
    const uint b_col_or_row,
    const uint a_col_or_row) {

    if constexpr (not a_transposed) {
        if (a_row_or_col >= matrix_a.size(0)) {
            return false;
        }
        if (a_col_or_row >= matrix_a.size(1)) {
            return false;
        }
    } else {
        if (a_row_or_col >= matrix_a.size(1)) {
            return false;
        }
        if (a_col_or_row >= matrix_a.size(0)) {
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


template<typename group_t, typename scalar_t>
__device__ __forceinline__ scalar_t group_reduce(
    group_t group, scalar_t *temp, scalar_t value) {

    auto lane = group.thread_rank();

    for (uint i = group.size() / 2; i > 0; i /= 2) {
        temp[lane] = value;
        group.sync();
        if (lane < i) {
            value += temp[lane + i];
        }
        group.sync();
    }

    return value;
}


template<uint block_size, typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(
    cg::thread_block block, scalar_t value) {

    __shared__ scalar_t temp[block_size];
    return group_reduce(block, temp, value);
}


template<uint block_size, typename scalar_t>
__device__ __forceinline__ void reduce_sum(
    scalar_t value, scalar_t *sum) {
    
    auto block = cg::this_thread_block();

    value = block_reduce_sum<block_size>(
        block, value
    );

    if (sum != nullptr and block.thread_rank() == 0) {
        gpuAtomicAdd(sum, value);
    }
}


template <
    uint block_size,
    bool transpose_a,
    bool transpose_b,
    typename scalar_t
>
__global__ void matmul_kernel(
    accessor_2d<scalar_t> matrix_a,
    accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto t = blockIdx.y;
    auto j = blockIdx.z;

    bool guard = product_matrix_guard<transpose_a, transpose_b>(
        matrix_a, matrix_b, t, j, i
    );

    scalar_t product = 0.0f;

    if (guard) {
        product = (
            ld_matrix_elem<transpose_a>(matrix_a, t, i) * 
            ld_matrix_elem<transpose_b>(matrix_b, i, j)
        );
    }

    reduce_sum<block_size>(
        product,
        guard ? &matrix_c[t][j] : nullptr
    );
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

    constexpr dim3 block_dim = {32 * 4, 1, 1};

    const dim3 grid_dim = {
        div_and_ceil(k, block_dim.x),
        div_and_ceil(m, block_dim.y),
        div_and_ceil(n, block_dim.z)
    };

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        matrix_a.scalar_type(),
        "linear_forward",
        ([&] {
            matmul_kernel<block_dim.x, transpose_a, transpose_b><<<grid_dim, block_dim>>>(
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