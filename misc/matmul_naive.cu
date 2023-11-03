#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using uint = __uint32_t;

template <typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;


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
__device__ __forceinline__ scalar_t get_matrix_elem(
    const accessor_2d<scalar_t> matrix, uint row, uint col) {

    if constexpr (transposed) {
        return matrix[col][row];
    } else {
        return matrix[row][col];
    }
}


template <bool transpose_a, bool transpose_b, typename scalar_t>
__device__ void matmul_naive(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    if (not product_matrix_guard<transpose_a, transpose_b>(
        matrix_a, matrix_b, m, n)) {
        return;
    }

    scalar_t acc = 0;
    auto sd = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    for (int k = 0; k < sd; k++) {
        acc += get_matrix_elem<transpose_a>(matrix_a, m, k) \
            * get_matrix_elem<transpose_b>(matrix_b, k, n);
    }

    matrix_c[m][n] = acc;
}


template <bool transpose_a, bool transpose_b, typename scalar_t>
__global__ void matmul_naive_kernel(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c) {

    matmul_naive<transpose_a, transpose_b, scalar_t>(
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
            matmul_naive_kernel<transpose_a, transpose_b><<<grid_dim, block_dim>>>(
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