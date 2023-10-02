#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using uint = __uint32_t;

template <typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;


template <bool a_transposed, bool b_transposed, typename scalar_t>
__device__ __forceinline__ bool matrix_guard(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    const uint a_row_or_col,
    const uint b_col_or_row) {

    if ((not a_transposed) and a_row_or_col >= matrix_a.size(0)) {
        return false;
    }

    if ((not b_transposed) and b_col_or_row >= matrix_b.size(1)) {
        return false;
    }

    if (a_transposed and a_row_or_col >= matrix_a.size(1)) {
        return false;
    }

    if (b_transposed and b_col_or_row >= matrix_b.size(0)) {
        return false;
    }

    return true;
}


template <bool transposed = false, typename scalar_t>
__device__ __forceinline__ scalar_t get_matrix_elem(
    const accessor_2d<scalar_t> matrix, uint row, uint col) {

    if constexpr (transposed) {
        return matrix[col][row];
    } else {
        return matrix[row][col];
    }
}


template <uint tile_size, bool transpose_a, bool transpose_b, typename scalar_t>
__device__ void matmul_smma(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    __shared__ scalar_t matrix_a_frag[tile_size][tile_size];
    __shared__ scalar_t matrix_b_frag[tile_size][tile_size];

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    if (not matrix_guard<transpose_a, transpose_b>(
        matrix_a, matrix_b, m, n)) {
        return;
    }

    scalar_t acc = 0;

    auto sd = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    for (uint t = 0; t < sd; t += tile_size) {
        uint j = threadIdx.x;
        uint i = threadIdx.y;

        matrix_a_frag[i][j] = get_matrix_elem<transpose_a>(
            matrix_a, tile_size * blockIdx.y + i, t + j);
        
        matrix_b_frag[i][j] = get_matrix_elem<transpose_b>(
            matrix_b, t + i, tile_size * blockIdx.x + j);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_size; k++) {
            acc += matrix_a_frag[i][k] * matrix_b_frag[k][j];
        }

        __syncthreads();
    }

    matrix_c[m][n] = acc;
}


template <uint tile_size, bool transpose_a, bool transpose_b, typename scalar_t>
__global__ void matmul_smma_kernel(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    matmul_smma<tile_size, transpose_a, transpose_b, scalar_t>(
        matrix_a, matrix_b, matrix_c);
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

    auto matrix_c = torch::zeros(
        {m, n},
        matrix_a.options()
    );

    constexpr dim3 block_dim = {16, 16};

    dim3 grid_dim = {
        div_and_ceil(n, block_dim.x),
        div_and_ceil(m, block_dim.y)
    };

    AT_DISPATCH_FLOATING_TYPES(
        matrix_a.scalar_type(),
        "matmul",
        ([&] {
            matmul_smma_kernel<block_dim.x, transpose_a, transpose_b><<<grid_dim, block_dim>>>(
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